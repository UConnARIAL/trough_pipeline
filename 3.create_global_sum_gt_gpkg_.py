#!/usr/bin/env python3
"""
Builds per-tile GeoPackages (components/edges/nodes + per-tile global_stats + XML)
and a master GeoPackage (global_stats rows for all tiles, global_poly polygons + XML per tile),
plus a whole-mosaic aggregation table global_stats_summary (1 row) with properly weighted averages.

Decisions:
  • Orientation: keep ONLY step-weighted mean on [0,180) (drop edge-weighted).
  • global_poly attributes: store averages instead of raw counts for nodes/edges:
      - avg_graph_nodes_per_component = num_graph_nodes / num_components
      - avg_graph_edges_per_component = num_graph_edges / num_components
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
from contextlib import contextmanager
from math import sin, cos, radians, atan2, degrees

import shapely.geometry as sgeom
from shapely.ops import unary_union

import rasterio
from rasterio.transform import xy as rio_xy
from skimage.morphology import skeletonize
from skan.csr import pixel_graph
import networkx as nx

# ---- SciPy optional (fast path) ----
try:
    from scipy.sparse.csgraph import connected_components as csgraph_connected_components
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


from gt_gpkg_common import StageTimer, _expected_gpkg_name_for_tif, _gpkg_delete_layer, _has_table, _is_valid_gpkg, _records_to_gdf, _table_has_fields, axial_mean_deg, build_master_gpkg, combine_axial_deg, filter_done_tiffs, get_sub_tile_id, has_layer_sqlite, insert_xml_metadata, is_gpkg_complete_by_tail, is_valid_geopackage, list_gpkg_layers_sqlite, logp, pix_to_xy, sanity_check_paths, sqlite_fast_writes, write_table

# ------------------------------ config ------------------------------

DEFAULT_PIXEL_M  = 0.5
HARD_MAX_WORKERS = 64
ID_VERSION = "v1.0-deterministic-rcsorted-neighsorted-compsorted"
COMPS_HEARTBEAT = 20000  # progress log every N components
DEFAULT_MAX_GB_PER_WORKER = 64

# ------------------------------ small logging helpers ------------------------------



# ------------------------------ math helpers ------------------------------





# ------------------------------ sqlite helpers ------------------------------





def _listlayers_safe(path):
    try:
        return set(list_gpkg_layers_sqlite(path))
    except Exception:
        return set()







#---------- resume gpkg code helpers -------------------------------
# --- put near your other utilities ---
from typing import Sequence, Tuple, List


#---------------- Sanity check for gpkg created by pipe line to decide to redo
import sqlite3





#</END gpkg sanity check>----Sanity check for gpkg created by pipe line to decide to redo



# ------------------------------ per-tile processing updated to include the sub_cell gpkg--
# --- NEW: helpers for per-file GPKG writing ---


from lib import add_from_mask_rastors as AdMsk, add_from_gpkg_layers as AL
import update_in_exp_cover as UpExpCov

EXTRA_GPKG_LAYERS = ["input_image_cutlines"]
NEW_LAYER_NAMES = ["InputMosaicCutlinesVector"]


#----- NEW write per file to resume from previous work -------------------

import re
from osgeo import ogr



#------------------------------ per-tile processing updated..............
################# AGGREGATION #################################################

# reuse your helpers:
# - StageTimer, sqlite_fast_writes, write_table, insert_xml_metadata, ID_VERSION






################# </AGGREGATION>#################################################
# ------------------------------ master build ------------------------------


# ------------------------------ startup sanity ------------------------------



# --- imports at top ---
import multiprocessing as mp
import psutil








#!/usr/bin/env python3
import os, sys, glob, argparse, sqlite3, time
from typing import List, Dict, Optional


# --- Assume your existing build_master_gpkg is importable in the same module or path ---
# from your_module import build_master_gpkg


AGG_PATTERNS = [
    "{tile}_TCN_summary.gpkg",
    "{tile}_TCN_summ*.gpkg",       # tolerant to slight typos/variants
    "*TCN_*summ*.gpkg",            # last resort within the tile dir
]
GLOBAL_STATS_TABLE = "global_stats"

INT_FIELDS = {
    "id", "num_components", "num_graph_nodes", "num_graph_edges",
    "end_nodes_count", "junction_nodes_count", "num_edges",
}
FLOAT_FIELDS = {
    "total_tcn_length_m", "largest_component_size_m", "average_component_size_m",
    "average_node_degree", "mean_edge_length_m", "mean_edge_orientation_deg_stepweighted",
}

# 'tile_id', 'id_version', 'raster_crs_wkt' are strings; leave as-is.


def find_tile_agg_gpkg(tile_dir: str, tile_id: str) -> Optional[str]:
    """Return the most likely aggregate GPKG path for this tile (newest if multiple)."""
    candidates: List[str] = []
    for pat in AGG_PATTERNS:
        path_pat = os.path.join(tile_dir, pat.format(tile=tile_id))
        matches = glob.glob(path_pat)
        if matches:
            candidates.extend(matches)
    # Deduplicate and pick newest by mtime if multiple
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def read_global_stats_row(gpkg_path: str) -> Optional[Dict]:
    """
    Read the single row from the non-spatial 'global_stats' table via SQLite.
    Returns a dict or None if missing.
    """
    try:
        con = sqlite3.connect(gpkg_path)
        cur = con.cursor()
        # Ensure the table exists (some stacks may expose as a view)
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
        try:
            con.close()
        except Exception:
            pass


def coerce_types(tile_id: str, row: Dict) -> Dict:
    """Coerce numeric fields to the expected types; leave strings as-is."""
    out = dict(row)
    # Enforce tile_id from directory, overriding any stray content
    out["tile_id"] = tile_id

    for k in INT_FIELDS:
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                logging.warning(f"[{tile_id}] cannot coerce {k}='{out[k]}' to int; setting None")
                out[k] = None

    for k in FLOAT_FIELDS:
        if k in out and out[k] is not None:
            try:
                out[k] = float(out[k])
            except Exception:
                logging.warning(f"[{tile_id}] cannot coerce {k}='{out[k]}' to float; setting None")
                out[k] = None

    # Raster WKT may be NULL
    out["raster_crs_wkt"] = out.get("raster_crs_wkt") or ""

    return out


def harvest_per_tile_rows(tiles_root: str,
                          tile_ids: List[str],
                          strict: bool = False) -> List[Dict]:
    """
    For each tile_id, locate the aggregate GPKG and read the single-row global_stats.
    Returns a list of dicts (without final 'id' sequencing).
    """
    rows: List[Dict] = []
    missing_files, missing_stats = [], []

    for tile_id in tile_ids:
        tile_dir = os.path.join(tiles_root, tile_id)
        gpkg = find_tile_agg_gpkg(tile_dir, tile_id)
        if not gpkg or not os.path.exists(gpkg):
            msg = f"[{tile_id}] aggregate GPKG not found under {tile_dir}"
            if strict:
                raise FileNotFoundError(msg)
            logging.warning(msg)
            missing_files.append(tile_id)
            continue

        stats = read_global_stats_row(gpkg)
        if not stats:
            msg = f"[{tile_id}] '{GLOBAL_STATS_TABLE}' table missing/empty in {os.path.basename(gpkg)}"
            if strict:
                raise RuntimeError(msg)
            logging.warning(msg)
            missing_stats.append(tile_id)
            continue

        rows.append(coerce_types(tile_id, stats))

    if missing_files:
        logging.warning(f"Tiles missing aggregate GPKG: {len(missing_files)} -> {missing_files[:8]}{'...' if len(missing_files)>8 else ''}")
    if missing_stats:
        logging.warning(f"Tiles missing '{GLOBAL_STATS_TABLE}': {len(missing_stats)} -> {missing_stats[:8]}{'...' if len(missing_stats)>8 else ''}")

    return rows

def slice_tile_ids(tiles_root: str,
                   st: Optional[int],
                   en: Optional[int],
                   one_tile: Optional[str]) -> List[str]:
    """Return a sorted, sliced list of tile_ids (1-based inclusive index)."""
    all_tiles = [d for d in sorted(os.listdir(tiles_root))
                 if os.path.isdir(os.path.join(tiles_root, d))]
    if one_tile:
        return [t for t in all_tiles if t == one_tile]

    if (st is None) and (en is None):
        return all_tiles

    n = len(all_tiles)
    st = 1 if st is None else max(1, int(st))
    en = n if en is None else min(n, int(en))
    return all_tiles[st - 1: en] if st <= en else []

#<Adding > subtile layer---------------------------
# --- Add near your other imports ---
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from shapely.geometry import MultiPolygon, Polygon


# --- Reuse your existing helper that finds each tile's aggregated GPKG ---
# def find_tile_agg_gpkg(tile_dir: str, tile_id: str) -> Optional[str]: ...

def _ensure_multi_polygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert Polygon -> MultiPolygon so schema is stable."""
    def _to_multi(geom):
        if geom is None:
            return None
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        return geom  # MultiPolygon or others left as-is
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_to_multi)
    return gdf

def _align_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subset/add columns to match 'cols' exactly; extra columns are dropped, missing filled with NA."""
    out = pd.DataFrame({c: df[c] if c in df.columns else pd.Series([pd.NA]*len(df)) for c in cols})
    return out

def append_subtile_stats_to_master(
    tiles_root: str,
    tile_ids: list[str],
    master_gpkg: str,
    layer_name: str = "SubTileStats",
    promote_to_multi: bool = True,
) -> None:
    """
    Append all per-tile 'SubTileStats' features into the master GPKG as a layer.
    Does NOT touch your global_stats master layer/table.
    """
    first_schema_cols = None
    target_crs = None
    total_written = 0
    missing = []

    for tile_id in tile_ids:
        tile_dir = os.path.join(tiles_root, tile_id)
        gpkg = find_tile_agg_gpkg(tile_dir, tile_id)
        if not gpkg or not os.path.exists(gpkg):
            logging.warning(f"[{tile_id}] aggregated GPKG not found; skipping SubTileStats")
            missing.append(tile_id)
            continue
        # Does this tile even have SubTileStats?
        try:
            layers = _listlayers_safe(gpkg)
        except Exception:
            # fall back quietly
            logging.warning(f"No layers found")
            layers = []
        if layer_name not in layers:
            logging.warning(f"[{tile_id}] layer '{layer_name}' missing in {Path(gpkg).name}; skipping")
            continue

        try:
            gdf = gpd.read_file(gpkg, layer=layer_name)
        except Exception as e:
            logging.warning(f"[{tile_id}] failed reading {layer_name}: {e}")
            continue
        if gdf.empty:
            continue

        # Establish target CRS on first non-empty read
        if target_crs is None and gdf.crs:
            target_crs = gdf.crs

        # Reproject if needed
        if target_crs and gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Promote to MultiPolygon for stable schema
        if promote_to_multi:
            gdf = _ensure_multi_polygon(gdf)

        # On first write, lock the schema (column order)
        # Keep common, expected columns first if present
        expected = [
            "tile_id","sub_tile_id","id_version","raster_crs_wkt",
            "num_components","total_tcn_length_m","largest_component_size_m","average_component_size_m",
            "num_graph_nodes","num_graph_edges","end_nodes_count","junction_nodes_count",
            "average_node_degree","mean_edge_length_m","mean_edge_orientation_deg_stepweighted",
            "num_edges","norm_input_exp_coverage_weighted","input_coverage","geometry",
        ]
        if first_schema_cols is None:
            # Preserve any additional columns but pin expected order at front if they exist
            cols = [c for c in expected if c in gdf.columns]
            extras = [c for c in gdf.columns if c not in cols + ["geometry"]]  # geometry handled separately
            first_schema_cols = cols + extras + (["geometry"] if "geometry" in gdf.columns else [])
            # Remember CRS
            if target_crs is None:
                target_crs = gdf.crs
        else:
            gdf = _align_columns(gdf, first_schema_cols)  # <-- now a plain DataFrame
            # PATCH: wrap back into a GeoDataFrame and restore CRS
            if "geometry" not in gdf.columns:
                logging.error(f"[{tile_id}] SubTileStats missing 'geometry' after align");
                continue
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=(target_crs or None))

        # Type coercions to avoid boolean/integer warnings
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

        # Set CRS if missing (to the chosen target)
        if target_crs is not None and gdf.crs is None:
            gdf = gdf.set_crs(target_crs, allow_override=True)

        # Append into master (layer will be created if it doesn't exist)
        mode = "a"
        try:
            gdf.to_file(master_gpkg, layer="subtile_stats_poly", driver="GPKG", mode=mode)
            total_written += len(gdf)
        except Exception as e:
            logging.error(f"[{tile_id}] append to master failed: {e}")
            continue

    crs_str = str(target_crs) if target_crs else "None"
    logging.info(f"[SubTileStats] appended {total_written} features to {Path(master_gpkg).name} (CRS={crs_str})")
    if missing:
        logging.warning(f"[SubTileStats] tiles missing aggregated GPKG: {len(missing)} (e.g., {missing[:6]}{'...' if len(missing)>6 else ''})")


#</Adding > subtile layer---------------------------

def main():
    p = argparse.ArgumentParser(description="Harvest per-tile global_stats into master GPKG (serial).")
    p.add_argument("--tiles_root", required=True, help="Root dir with tile subfolders")
    p.add_argument("--imagery_dir", required=True, help="Imagery dir (passed to build_master_gpkg)")
    p.add_argument("--mosaic_gpkg", required=True, help="Output master GPKG path")
    p.add_argument("--tile_rank_st", type=int, default=None, help="1-based start index in sorted tiles")
    p.add_argument("--tile_rank_end", type=int, default=None, help="1-based end index (inclusive)")
    p.add_argument("--one_tile", type=str, default=None, help="Process only this tile_id")
    p.add_argument("--strict", action="store_true", help="Fail on missing GPKG or stats")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    tiles = slice_tile_ids(args.tiles_root, args.tile_rank_st, args.tile_rank_end, args.one_tile)
    if not tiles:
        logging.error("No tiles selected by slice/one_tile; nothing to do.")
        sys.exit(2)

    logging.info(f"Selected tiles: {len(tiles)}  (slice: {args.tile_rank_st}-{args.tile_rank_end}  one_tile: {args.one_tile})")

    t0 = time.time()
    rows = harvest_per_tile_rows(args.tiles_root, tiles, strict=args.strict)
    if not rows:
        logging.error("No per-tile rows harvested; aborting.")
        sys.exit(3)

    # Stable order + assign fresh sequential ids for the master
    rows = sorted(rows, key=lambda r: r["tile_id"])
    for i, r in enumerate(rows, start=1):
        r["id"] = i

    # Build the master using your existing function
    build_master_gpkg(args.mosaic_gpkg, rows, args.imagery_dir)

    append_subtile_stats_to_master(
        tiles_root=args.tiles_root,
        tile_ids=tiles,
        master_gpkg=args.mosaic_gpkg,
        layer_name="SubTileStats",
    )

    dt = time.time() - t0
    logging.info(f"Master built: {args.mosaic_gpkg}  | tiles ingested: {len(rows)}  | time: {dt/60:.1f} min")


if __name__ == "__main__":
    try:
        # keep temp on fast local storage if available (optional)
        for envk in ("TMPDIR","CPL_TMPDIR","SQLITE_TMPDIR"):
            os.environ.setdefault(envk, "/tmp")
        os.environ.setdefault("OGR_SQLITE_SYNCHRONOUS", "OFF")
        os.environ.setdefault("GDAL_NUM_THREADS", "1")

        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)

"""
python gt_gpkg_tile_global_sum_w_sub.py --tiles_root /scratch2/projects/PDG_shared/TCN_gpkgs/  --imagery_dir /scratch2/projects/PDG_shared/AlaskaTundraMosaic/imagery       --mosaic_gpkg /scratch2/projects/PDG_shared/TCN_gpkgs/new_all_mosaic_w_sub_1_10.gpkg       --tile_rank_st 1 --tile_rank_end 10       --loglevel DEBUG

"""



