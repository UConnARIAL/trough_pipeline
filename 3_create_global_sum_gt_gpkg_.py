#!/usr/bin/env python3
"""
3_create_global sun_gt_gpkg.py

is the stage specific logic after refactoring common code to gt_gpkg_common.py,

This script does the global sumerization for the entire mosaic

It can be executed for a range at tile level (sorted ranked order)

It will NOT overite (NOT Idemptoent) the existing gpkgs if they exist.

It will continue to append to the exsiting data and also aggegate values.

Overall objective:
Builds per-tile GeoPackages (components/edges/nodes + per-tile global_stats + XML)
and a master GeoPackage (global_stats rows for all tiles, global_poly polygons + XML per tile),
plus a whole-mosaic aggregation table global_stats_summary (1 row) with properly weighted averages.

Decisions:
  • Orientation: keep ONLY step-weighted mean on [0,180) (drop edge-weighted).
  • global_poly attributes: store averages instead of raw counts for nodes/edges:
      - avg_graph_nodes_per_component = num_graph_nodes / num_components
      - avg_graph_edges_per_component = num_graph_edges / num_components

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera
"""

#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import sqlite3
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from gt_gpkg_common import build_master_gpkg, list_gpkg_layers_sqlite


# ----------------------------- TOML + cfg helpers -----------------------------

def load_toml(path: str | Path) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    try:
        import tomllib  # py3.11+
        return tomllib.loads(raw.decode("utf-8"))
    except Exception:
        import tomli     # py3.10 backport
        return tomli.loads(raw.decode("utf-8"))


def cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def cfg_get_step(cfg: dict, step: str, *keys, default=None):
    return cfg_get(cfg, "steps", step, *keys, default=default)


def apply_thread_env(cfg: dict) -> None:
    # Keep these consistent across steps (and safe on HPC)
    openblas = int(cfg_get(cfg, "runtime", "openblas_threads", default=1) or 1)
    omp = int(cfg_get(cfg, "runtime", "omp_threads", default=1) or 1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(openblas)
    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("OGR_SQLITE_SYNCHRONOUS", "OFF")


def get_id_version(cfg: dict) -> str:
    v = cfg_get(cfg, "project", "id_version", default=None)
    if not v:
        raise ValueError("Missing [project].id_version in config")
    return str(v)


def get_target_epsg(cfg: dict) -> int:
    epsg = cfg_get(cfg, "crs", "target_epsg", default=3338)
    return int(epsg)


def resolve_workers(cfg: dict, step: str, cli_workers: Optional[int]) -> int:
    """
    CLI override > steps.<step>.runtime.workers > runtime.workers
    Convention: 0 means auto.
    """
    if cli_workers is not None:
        return int(cli_workers)
    return int(
        cfg_get_step(
            cfg, step, "runtime", "workers",
            default=cfg_get(cfg, "runtime", "workers", default=0)
        ) or 0
    )


def compute_auto_workers(cfg: dict) -> int:
    import psutil
    total_mem_gb = psutil.virtual_memory().total / 1e9
    hard_max = int(cfg_get(cfg, "runtime", "hard_max_workers", default=32) or 32)
    max_gb_per_worker = float(cfg_get(cfg, "runtime", "max_gb_per_worker", default=64) or 64)

    cpu_cnt = max(1, (os.cpu_count() or 1) - 1)
    mem_based = max(1, int(total_mem_gb // max_gb_per_worker))
    return max(1, min(cpu_cnt, mem_based, hard_max))


# ----------------------------- tile selection helpers -----------------------------

def list_tile_ids(tiles_root: str) -> List[str]:
    return sorted([
        d for d in os.listdir(tiles_root)
        if os.path.isdir(os.path.join(tiles_root, d))
    ])


def apply_rank_partition(tiles: List[str], rank: int, world_size: int) -> List[str]:
    world_size = max(1, int(world_size))
    rank = max(1, int(rank))
    rank0 = rank - 1
    return [t for i, t in enumerate(tiles) if (i % world_size) == rank0]


def apply_slice(tiles: List[str], st: Optional[int], en: Optional[int]) -> List[str]:
    if st is None and en is None:
        return tiles
    n = len(tiles)
    st = 1 if st is None else max(1, int(st))
    en = n if en is None else min(n, int(en))
    if st > en:
        return []
    return tiles[st - 1: en]


# ----------------------------- harvesting global_stats -----------------------------

GLOBAL_STATS_TABLE = "global_stats"

AGG_PATTERNS = [
    "{tile}_TCN_summary.gpkg",
    "{tile}_TCN_summ*.gpkg",
    "*TCN_*summ*.gpkg",
]

INT_FIELDS = {
    "id", "num_components", "num_graph_nodes", "num_graph_edges",
    "end_nodes_count", "junction_nodes_count", "num_edges",
}
FLOAT_FIELDS = {
    "total_tcn_length_m", "largest_component_size_m", "average_component_size_m",
    "average_node_degree", "mean_edge_length_m", "mean_edge_orientation_deg_stepweighted",
}

def find_tile_summary_gpkg(tile_dir: str, tile_id: str) -> Optional[str]:
    candidates: List[str] = []
    for pat in AGG_PATTERNS:
        path_pat = os.path.join(tile_dir, pat.format(tile=tile_id))
        candidates.extend(glob.glob(path_pat))
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def read_global_stats_row(gpkg_path: str) -> Optional[Dict]:
    con = None
    try:
        con = sqlite3.connect(gpkg_path)
        cur = con.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (GLOBAL_STATS_TABLE,),
        )
        if not cur.fetchone():
            return None

        cur.execute(f"SELECT * FROM {GLOBAL_STATS_TABLE} LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

def coerce_types(tile_id: str, row: Dict) -> Dict:
    out = dict(row)
    out["tile_id"] = tile_id
    for k in INT_FIELDS:
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                out[k] = None
    for k in FLOAT_FIELDS:
        if k in out and out[k] is not None:
            try:
                out[k] = float(out[k])
            except Exception:
                out[k] = None
    out["raster_crs_wkt"] = out.get("raster_crs_wkt") or ""
    return out


def _harvest_one_tile(task: Tuple[str, str, bool]) -> Optional[Dict]:
    tiles_root, tile_id, strict = task
    tile_dir = os.path.join(tiles_root, tile_id)
    gpkg = find_tile_summary_gpkg(tile_dir, tile_id)
    if not gpkg or not os.path.exists(gpkg):
        msg = f"[{tile_id}] missing summary gpkg under {tile_dir}"
        if strict:
            raise FileNotFoundError(msg)
        logging.warning(msg)
        return None

    stats = read_global_stats_row(gpkg)
    if not stats:
        msg = f"[{tile_id}] missing/empty '{GLOBAL_STATS_TABLE}' in {Path(gpkg).name}"
        if strict:
            raise RuntimeError(msg)
        logging.warning(msg)
        return None

    return coerce_types(tile_id, stats)


def harvest_per_tile_rows(
    tiles_root: str,
    tile_ids: List[str],
    *,
    n_workers: int,
    strict: bool,
) -> List[Dict]:
    tasks = [(tiles_root, t, strict) for t in tile_ids]
    rows: List[Dict] = []

    if n_workers <= 1:
        for t in tasks:
            r = _harvest_one_tile(t)
            if r:
                rows.append(r)
        return rows

    with mp.Pool(processes=max(1, n_workers)) as pool:
        for r in pool.imap_unordered(_harvest_one_tile, tasks, chunksize=8):
            if r:
                rows.append(r)
    return rows


def _listlayers_safe(path: str) -> set:
    try:
        return set(list_gpkg_layers_sqlite(path))
    except Exception:
        return set()


# ----------------------------- append SubTileStats to master -----------------------------

def _ensure_multi_polygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def _to_multi(geom):
        if geom is None:
            return None
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        return geom
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_to_multi)
    return gdf


def _align_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # exact schema match: drop extras, add missings as NA
    return pd.DataFrame({
        c: (df[c] if c in df.columns else pd.Series([pd.NA] * len(df)))
        for c in cols
    })


def append_subtile_stats_to_master(
    tiles_root: str,
    tile_ids: List[str],
    master_gpkg: str,
    *,
    in_layer: str = "SubTileStats",
    out_layer: str = "SubTileStats",
    target_epsg: int = 3338,
    promote_to_multi: bool = True,
) -> None:
    target_crs = f"EPSG:{int(target_epsg)}"

    schema_cols: Optional[List[str]] = None
    total_written = 0

    for tile_id in tile_ids:
        tile_dir = os.path.join(tiles_root, tile_id)
        gpkg = find_tile_summary_gpkg(tile_dir, tile_id)
        if not gpkg or not os.path.exists(gpkg):
            logging.debug("[%s] no summary gpkg for SubTileStats append; skip", tile_id)
            continue

        layers = _listlayers_safe(gpkg)
        if in_layer not in layers:
            logging.debug("[%s] layer '%s' not present; skip", tile_id, in_layer)
            continue

        try:
            gdf = gpd.read_file(gpkg, layer=in_layer)
        except Exception as e:
            logging.warning("[%s] failed reading %s: %s", tile_id, in_layer, e)
            continue
        if gdf.empty:
            continue

        # enforce target CRS (equal-area EPSG:3338)
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs, allow_override=True)
        elif str(gdf.crs) != str(gpd.GeoSeries(crs=target_crs).crs):
            gdf = gdf.to_crs(target_crs)

        if promote_to_multi:
            gdf = _ensure_multi_polygon(gdf)

        # lock schema on first write
        expected_front = [
            "tile_id","sub_tile_id","id_version","raster_crs_wkt",
            "num_components","total_tcn_length_m","largest_component_size_m","average_component_size_m",
            "num_graph_nodes","num_graph_edges","end_nodes_count","junction_nodes_count",
            "average_node_degree","mean_edge_length_m","mean_edge_orientation_deg_stepweighted",
            "num_edges","norm_input_exp_coverage_weighted","input_coverage",
        ]

        if schema_cols is None:
            front = [c for c in expected_front if c in gdf.columns]
            extras = [c for c in gdf.columns if c not in front and c != "geometry"]
            schema_cols = front + extras + ["geometry"]
        else:
            df = _align_columns(gdf, schema_cols)
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=target_crs)

        # type coercions (avoid driver warnings)
        int_cols = {"num_components","num_graph_nodes","num_graph_edges","end_nodes_count","junction_nodes_count","num_edges"}
        float_cols = {
            "total_tcn_length_m","largest_component_size_m","average_component_size_m",
            "average_node_degree","mean_edge_length_m","mean_edge_orientation_deg_stepweighted",
            "norm_input_exp_coverage_weighted","input_coverage"
        }
        for c in gdf.columns:
            if c in int_cols:
                gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0).astype(np.int64)
            elif c in float_cols:
                gdf[c] = pd.to_numeric(gdf[c], errors="coerce").astype(float)

        mode = "a" if os.path.exists(master_gpkg) else "w"
        try:
            gdf.to_file(master_gpkg, layer=out_layer, driver="GPKG", mode=mode)
            total_written += len(gdf)
        except Exception as e:
            logging.error("[%s] append %s -> master failed: %s", tile_id, out_layer, e)

    logging.info("[SubTileStats] appended %d features to %s (CRS=%s, layer=%s)",
                 total_written, Path(master_gpkg).name, target_crs, out_layer)


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser("Step 3: Global master aggregation")
    p.add_argument("--config", required=True, help="Path to config.toml")

    # Optional overrides
    p.add_argument("--mosaic-gpkg", default=None, help="Override master gpkg output (else io.mosaic_gpkg or output_root/all_mosaic.gpkg)")
    p.add_argument("--tiles-root", default=None, help="Override tiles root (else io.output_tiles_dir)")
    p.add_argument("--imagery-dir", default=None, help="Override imagery dir (else cutlines.root)")
    p.add_argument("--workers", type=int, default=None, help="Override workers (0=auto). If omitted uses TOML.")
    p.add_argument("--rank", type=int, default=1, help="1-based rank for round-robin partition")
    p.add_argument("--world-size", type=int, default=1, help="Total ranks for round-robin partition")
    p.add_argument("--tile-rank-st", type=int, default=None, help="1-based slice start (after partition)")
    p.add_argument("--tile-rank-end", type=int, default=None, help="1-based slice end (after partition, inclusive)")
    p.add_argument("--tiles-per-run", type=int, default=None, help="Cap tiles processed this run (override TOML; 0=unlimited)")
    p.add_argument("--one-tile", type=str, default=None, help="Only process this tile_id")
    p.add_argument("--strict", action="store_true", help="Fail on missing summary gpkg or stats")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()

    cfg = load_toml(args.config)
    apply_thread_env(cfg)

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    id_version = get_id_version(cfg)
    target_epsg = get_target_epsg(cfg)

    tiles_root = args.tiles_root or cfg_get(cfg, "io", "output_tiles_dir")
    if not tiles_root:
        raise ValueError("Need io.output_tiles_dir in config (or pass --tiles-root)")
    tiles_root = str(tiles_root)

    imagery_dir = args.imagery_dir or cfg_get(cfg, "cutlines", "root") or cfg_get(cfg, "io", "imagery_dir")
    if not imagery_dir:
        raise ValueError("Need cutlines.root (or io.imagery_dir) in config (or pass --imagery-dir)")
    imagery_dir = str(imagery_dir)

    # mosaic output path
    mosaic_gpkg = (
        args.mosaic_gpkg
        or cfg_get(cfg, "io", "mosaic_gpkg")
        or (os.path.join(str(cfg_get(cfg, "io", "output_root", default=".")), "all_mosaic.gpkg"))
    )
    mosaic_gpkg = str(mosaic_gpkg)

    # tile list
    tiles = list_tile_ids(tiles_root)
    if args.one_tile:
        tiles = [t for t in tiles if t == args.one_tile]

    tiles = apply_rank_partition(tiles, args.rank, args.world_size)
    tiles = apply_slice(tiles, args.tile_rank_st, args.tile_rank_end)

    if not tiles:
        logging.error("No tiles selected.")
        sys.exit(2)

    # cap tiles this run
    cap = (
        int(args.tiles_per_run)
        if args.tiles_per_run is not None
        else int(cfg_get_step(cfg, "global", "tiles_per_run", default=0) or 0)
    )
    if cap > 0:
        tiles = tiles[:cap]

    # workers
    w = resolve_workers(cfg, "global", args.workers)
    n_workers = compute_auto_workers(cfg) if w == 0 else max(1, w)

    logging.info("Step=global | id_version=%s | target_epsg=%d | tiles=%d | workers=%d",
                 id_version, target_epsg, len(tiles), n_workers)

    t0 = time.time()
    rows = harvest_per_tile_rows(tiles_root, tiles, n_workers=n_workers, strict=args.strict)
    if not rows:
        logging.error("No per-tile rows harvested; aborting.")
        sys.exit(3)

    # stable ordering + sequential ids
    rows = sorted(rows, key=lambda r: r["tile_id"])
    for i, r in enumerate(rows, start=1):
        r["id"] = i

    # Build master (pass id_version so common code never depends on a global constant)
    build_master_gpkg(mosaic_gpkg, rows, imagery_dir, id_version=id_version)

    # Append SubTileStats polygons into master, enforcing EPSG:3338
    tile_ids_in_master = [r["tile_id"] for r in rows]
    append_subtile_stats_to_master(
        tiles_root=tiles_root,
        tile_ids=tile_ids_in_master,
        master_gpkg=mosaic_gpkg,
        in_layer="SubTileStats",
        out_layer="SubTileStats",
        target_epsg=target_epsg,
    )

    dt = time.time() - t0
    logging.info("Master built: %s | tiles=%d | time=%.1f min",
                 mosaic_gpkg, len(rows), dt / 60.0)

if __name__ == "__main__":
    try:
        for envk in ("TMPDIR","CPL_TMPDIR","SQLITE_TMPDIR"):
            os.environ.setdefault(envk, "/tmp")
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)


"""
python gt_gpkg_tile_global_sum_w_sub.py --tiles_root /scratch2/projects/PDG_shared/TCN_gpkgs/  --imagery_dir /scratch2/projects/PDG_shared/AlaskaTundraMosaic/imagery --mosaic_gpkg /scratch2/projects/PDG_shared/TCN_gpkgs/new_all_mosaic_w_sub_1_10.gpkg --tile_rank_st 1 --tile_rank_end 10       --loglevel DEBUG
python 3_create_global_sum_gt_gpkg_.py --config config.toml
"""



