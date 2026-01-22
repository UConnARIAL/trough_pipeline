#!/usr/bin/env python3
"""
Common utilities shared by:
  1_create_resume_subtile_gt_gpkg.py
  2_create_subtile_sum_gt_gpkg.py
  3_create_global_sum_gt_gpkg.py

Refactored to remove duplicated code while keeping stage-specific logic in each script.

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

from __future__ import annotations
import contextlib
import logging
import os
import re
import sqlite3
import time
from math import sin, cos, atan2, radians, degrees
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from shapely.ops import unary_union

import glob

class StageTimer:
    """Tiny context manager for stage timing that logs on exit if enabled."""
    def __init__(self, enabled: bool, prefix: str, stage: str):
        self.enabled = enabled
        self.prefix = prefix
        self.stage = stage
        self.t0 = None
    def __enter__(self):
        if self.enabled:
            logging.debug(f"{self.prefix} {self.stage}...")
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if self.enabled:
            if exc is None:
                logging.debug(f"{self.prefix} {self.stage} ✓ {dt:.2f}s")
            else:
                logging.debug(f"{self.prefix} {self.stage} ✗ {dt:.2f}s ({exc})")

def logp(verbose: bool, level: int, msg: str):
    logging.log(level, msg) if verbose else None



def axial_mean_deg(deg_list):
    """Axial circular mean on [0,180). Returns -1.0 if empty."""
    if not deg_list:
        return -1.0
    two_theta = np.radians(2.0 * np.asarray(deg_list, dtype=float))
    s, c = np.sum(np.sin(two_theta)), np.sum(np.cos(two_theta))
    if s == 0.0 and c == 0.0:
        return -1.0
    return (0.5 * np.degrees(np.arctan2(s, c))) % 180.0

def combine_axial_deg(weight_angle_pairs):
    """Combine axial angles given (w, theta_deg) pairs. Returns -1.0 if invalid/empty."""
    S = 0.0; C = 0.0; tot_w = 0.0
    for w, theta in weight_angle_pairs:
        if w is None or theta is None:
            continue
        if w <= 0 or theta < 0:
            continue
        two = radians(2.0 * float(theta))
        S += float(w) * sin(two)
        C += float(w) * cos(two)
        tot_w += float(w)
    if tot_w <= 0 or (abs(S) < 1e-15 and abs(C) < 1e-15):
        return -1.0
    return (0.5 * degrees(atan2(S, C))) % 180.0

def pix_to_xy(transform, rr, cc):
    xs, ys = rio_xy(transform, rr, cc, offset="center")
    return np.asarray(xs), np.asarray(ys)


def sqlite_fast_writes(conn):
    """Speed up writes without changing content."""
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=MEMORY;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB page cache
    try:
        yield
    finally:
        cur.execute("PRAGMA synchronous=FULL;")

def write_table(conn, table_name, df, pk_name=None):
    """Create/replace table; optionally set INTEGER PRIMARY KEY 'pk_name'."""
    tmp = f"_{table_name}_tmp"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        if pk_name and col == pk_name:
            cols.append(f"{pk_name} INTEGER PRIMARY KEY")
        else:
            if np.issubdtype(dtype, np.integer):
                cols.append(f"{col} INTEGER")
            elif np.issubdtype(dtype, np.floating):
                cols.append(f"{col} REAL")
            else:
                cols.append(f"{col} TEXT")
    cur.execute(f"CREATE TABLE {table_name} ({', '.join(cols)});")
    cols_list = ", ".join(df.columns)
    cur.execute(f"INSERT INTO {table_name} ({cols_list}) SELECT {cols_list} FROM {tmp};")
    cur.execute(f"DROP TABLE {tmp};")
    conn.commit()



def insert_xml_metadata(conn, xml_text, links):
    """Insert XML into gpkg_metadata and link via gpkg_metadata_reference."""
    cur = conn.cursor()
    cur.executescript("""
CREATE TABLE IF NOT EXISTS gpkg_metadata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  md_scope TEXT NOT NULL DEFAULT 'dataset',
  md_standard_uri TEXT,
  mime_type TEXT NOT NULL DEFAULT 'text/xml',
  metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gpkg_metadata_reference (
  reference_scope TEXT NOT NULL,
  table_name TEXT,
  column_name TEXT,
  row_id_value INTEGER,
  timestamp DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  md_file_id INTEGER NOT NULL,
  srs_id INTEGER,
  FOREIGN KEY (md_file_id) REFERENCES gpkg_metadata(id)
);
""")
    cur.execute(
        "INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata) VALUES (?,?,?,?)",
        ("dataset", "https://www.ogc.org/standards/geopackage", "text/xml", xml_text)
    )
    md_id = cur.lastrowid
    for link in links:
        scope = link.get("reference_scope", "geopackage")
        table_name = link.get("table_name")
        column_name = link.get("column_name")
        row_id_value = link.get("row_id_value")
        cur.execute(
            "INSERT INTO gpkg_metadata_reference (reference_scope, table_name, column_name, row_id_value, md_file_id) VALUES (?,?,?,?,?)",
            (scope, table_name, column_name, row_id_value, md_id)
        )
    conn.commit()
    return md_id

def list_gpkg_layers_sqlite(path):
    """
    Return a list of layer (table) names from a GeoPackage at `path`
    without using Fiona/GDAL.

    Order of attempts:
      1) gpkg_contents (preferred, per spec)
      2) gpkg_geometry_columns (features only)
      3) sqlite_master fallback (excluding SQLite/GPKG internals)
    """
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            # 1) Spec table: gpkg_contents
            try:
                rows = cur.execute(
                    "SELECT table_name, data_type FROM gpkg_contents"
                ).fetchall()
                if rows:
                    # include feature tables; optionally include tiles/attributes if you care
                    layers = [t for t, dt in rows if dt in ("features", "tiles", "attributes")]
                    if layers:
                        return layers
            except sqlite3.Error:
                pass

            # 2) Geometry columns (features)
            try:
                rows = cur.execute(
                    "SELECT table_name FROM gpkg_geometry_columns"
                ).fetchall()
                if rows:
                    return [r[0] for r in rows]
            except sqlite3.Error:
                pass

            # 3) Fallback: list user tables (filter out system/internal tables)
            rows = cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'gpkg_%'"
            ).fetchall()
            return [r[0] for r in rows]
    except sqlite3.Error:
        return []

def has_layer_sqlite(path, layer_name):
    """Case-sensitive check; adjust to lower()==lower() if you want CI."""
    return layer_name in list_gpkg_layers_sqlite(path)

def is_valid_geopackage(path: str) -> bool:
    """Return True if file looks like a real GeoPackage (has gpkg_contents & gpkg_spatial_ref_sys)."""
    if not os.path.exists(path):
        return False
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('gpkg_contents','gpkg_spatial_ref_sys');")
            rows = cur.fetchall()
            return len(rows) == 2
    except Exception:
        return False

def _is_valid_gpkg(path: str) -> bool:
    """Optional sanity check: can OGR open it and see ≥1 layer?"""
    try:
        from osgeo import ogr  # imported here to avoid hard dependency if not verifying
        ds = ogr.Open(path, update=0)
        ok = bool(ds) and ds.GetLayerCount() >= 7
        ds = None
        logging.info(f"[file] {path} Layer count {ds.GetLayerCount()}")
        return ok
    except Exception:
        return False


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    ).fetchone()
    return row is not None



def _table_has_fields(con: sqlite3.Connection, table: str, fields: tuple[str, ...]) -> bool:
    cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}  # r[1] = name
    missing = [f for f in fields if f not in cols]
    if missing:
        logging.debug(f'[{table}] missing fields: {missing}')
        return False
    return True



def _gpkg_delete_layer(gpkg_path: str, layer_name: str) -> None:
    ds = ogr.Open(gpkg_path, update=1)
    if not ds:
        return
    try:
        lyr = ds.GetLayerByName(layer_name)
        if lyr:
            ds.DeleteLayer(layer_name)
    finally:
        ds = None



def get_sub_tile_id(tif_path: str) -> str:
    return os.path.splitext(os.path.basename(tif_path))[0]



def _expected_gpkg_name_for_tif(tif_path: str, gpkg_suffix: str = "_TCN.gpkg") -> str:
    """ArcticMosaic_58_16_1_1_mask.tif -> ArcticMosaic_58_16_1_1_TCN.gpkg"""
    stem = Path(tif_path).stem
    stem = re.sub(r'[_-]mask$', '', stem, flags=re.I)
    return f"{stem}{gpkg_suffix}"



def is_gpkg_complete_by_tail(
    gpkg_path: str,
    layer: str = "GraphTheoraticComponents",
    required_fields: tuple[str, str] = ("norm_input_exp", "input_exp_cnt"),
    require_nonnull: bool = True,
    min_nonnull: int = 1,
) -> bool:
    """
    Minimal 'done' check: the pipeline’s final write succeeded if `layer`
    exists and contains the columns you add last. Optionally ensure some
    rows have both columns filled (non-NULL).
    """
    if not os.path.exists(gpkg_path):
        logging.debug(f"[complete-check] missing file: {gpkg_path}")
        return False

    try:
        with sqlite3.connect(gpkg_path) as con:
            # 1) table exists?
            if not _has_table(con, layer):
                logging.debug(f"[complete-check] table not found: {layer}")
                return False

            # 2) required fields exist?
            if not _table_has_fields(con, layer, required_fields):
                return False

            if require_nonnull:
                f1, f2 = required_fields
                (n_nonnull,) = con.execute(
                    f'SELECT COUNT(*) FROM "{layer}" WHERE "{f1}" IS NOT NULL AND "{f2}" IS NOT NULL'
                ).fetchone()
                if n_nonnull < min_nonnull:
                    logging.debug(f"[complete-check] only {n_nonnull} rows with both {f1},{f2} non-NULL")
                    return False

            return True
    except Exception as e:
        logging.debug(f"[complete-check] error: {e}")
        return False



def filter_done_tiffs(
    tif_paths: Sequence[str],
    tile_out_dir: str,
    *,
    verify: bool = False,     # set True to open GPKG and ensure it has ≥1 layer
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Drop TIFFs that already have their corresponding _TCN.gpkg in tile_out_dir.
    Returns (remaining_tifs, skipped_tifs).
    """
    os.makedirs(tile_out_dir, exist_ok=True)
    existing = {f for f in os.listdir(tile_out_dir) if f.lower().endswith(".gpkg")}
    remaining, skipped = [], []

    for tif in tif_paths:
        gpkg_name = _expected_gpkg_name_for_tif(tif)
        gpkg_path = os.path.join(tile_out_dir, gpkg_name)
        already = gpkg_name in existing and (not verify or is_gpkg_complete_by_tail(gpkg_path))
        if already:
            skipped.append(tif)
        else:
            remaining.append(tif)
    if verbose:
        print(f"[filter] skipped {len(skipped)} already-built; {len(remaining)} to process in {tile_out_dir}")
    logging.info(f"[filter] skipped {len(skipped)} already-built; {len(remaining)} to process in {tile_out_dir}")
    return remaining, skipped



def sanity_check_paths(master_dir, imagery_dir):
    ok = True
    if not os.path.isdir(master_dir):
        logging.error(f"--master_dir not a directory: {master_dir}")
        ok = False
    if not os.path.isdir(imagery_dir):
        logging.error(f"--imagery_dir not a directory: {imagery_dir}")
        ok = False

    tiles = [d for d in sorted(os.listdir(master_dir)) if os.path.isdir(os.path.join(master_dir, d))] if os.path.isdir(master_dir) else []
    tifs_total = 0; sample = []
    for t in tiles[:10]:
        tifs = sorted(glob.glob(os.path.join(master_dir, t, "*.tif")))
        tifs_total += len(tifs)
        if tifs and len(sample) < 5:
            sample.append(tifs[0])

    logging.info(f"[sanity] tiles discovered: {len(tiles)} | total *.tif: {tifs_total}")
    if sample:
        logging.info("[sanity] sample tif(s): " + " | ".join(sample))
    else:
        logging.warning("[sanity] no *.tif found under master_dir/<tile>/*.tif")
    return ok



def _records_to_gdf(records, crs_wkt):
    """Convert list[dict] with 'geometry' into a GeoDataFrame or None if empty."""
    if not records:
        return None
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs_wkt or None)



def build_master_gpkg(mosaic_gpkg_path, per_tile_rows, imagery_dir):
    if not per_tile_rows:
        logging.warning("No per-tile rows to write to master.")
        return

    target_crs_wkt = next((r["raster_crs_wkt"] for r in per_tile_rows if r.get("raster_crs_wkt")), "")

    rows = []
    for i, r in enumerate(sorted(per_tile_rows, key=lambda d: d["tile_id"]), start=1):
        rr = dict(r); rr["id"] = i
        rows.append(rr)

    gs_df = pd.DataFrame(rows, columns=[
        "id","tile_id","id_version","raster_crs_wkt",
        "num_components","total_tcn_length_m",
        "largest_component_size_m","average_component_size_m",
        "num_graph_nodes","num_graph_edges",
        "end_nodes_count","junction_nodes_count",
        "average_node_degree","mean_edge_length_m",
        "mean_edge_orientation_deg_stepweighted",
        "num_edges"
    ])

    os.makedirs(os.path.dirname(mosaic_gpkg_path), exist_ok=True)

    # ---------- 1) build polygons FIRST so file is a valid GPKG ----------
    polys = []
    crs_poly = None
    for r in rows:
        tile_id = r["tile_id"]
        shp_path = os.path.join(imagery_dir, tile_id, f"ArcticMosaic_{tile_id}_tiles.shp")
        if not os.path.exists(shp_path):
            logging.warning(f"[master] missing shapefile for '{tile_id}': {shp_path}")
            continue
        try:
            gdf_tile = gpd.read_file(shp_path)
            if target_crs_wkt:
                gdf_tile = gdf_tile.to_crs(target_crs_wkt)
        except Exception as e:
            logging.warning(f"[master] read/reproject fail {shp_path}: {e}")
            continue
        crs_poly = crs_poly or gdf_tile.crs
        dissolved = unary_union(gdf_tile.geometry)

        # compute requested averages for attributes stored in global_poly
        C = float(r.get("num_components", 0) or 0)
        V = float(r.get("num_graph_nodes", 0) or 0)
        E = float(r.get("num_graph_edges", 0) or 0)
        avg_nodes_per_comp = (V / C) if C > 0 else 0.0
        avg_edges_per_comp = (E / C) if C > 0 else 0.0

        # copy all fields except id, then overwrite node/edge counts with averages as requested
        poly_attrs = {k: v for k, v in r.items() if k != "id"}
        # replace raw counts with averages for the polygon attributes
        poly_attrs.pop("num_graph_nodes", None)
        poly_attrs.pop("num_graph_edges", None)
        poly_attrs["avg_graph_nodes_per_component"] = float(avg_nodes_per_comp)
        poly_attrs["avg_graph_edges_per_component"] = float(avg_edges_per_comp)

        polys.append({
            **poly_attrs,
            "id": r["id"],
            "geometry": dissolved
        })

    if polys:
        gdf_poly = gpd.GeoDataFrame(
            [{k: v for k, v in p.items() if k != "geometry"} for p in polys],
            geometry=[p["geometry"] for p in polys],
            crs=crs_poly
        )
        mode = "w" if not os.path.exists(mosaic_gpkg_path) else "a"
        gdf_poly.to_file(mosaic_gpkg_path, layer="global_poly", driver="GPKG", mode=mode)
        logging.debug("[master] wrote global_poly; GPKG initialized.")
    else:
        if not is_valid_geopackage(mosaic_gpkg_path):
            dummy = gpd.GeoDataFrame({"_id":[1]}, geometry=[sgeom.Point(0.0, 0.0)], crs="EPSG:4326")
            dummy.to_file(mosaic_gpkg_path, layer="__init__", driver="GPKG", mode="w")
            with sqlite3.connect(mosaic_gpkg_path) as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM gpkg_contents WHERE table_name='__init__';")
                cur.execute("DROP TABLE IF EXISTS '__init__';")
                conn.commit()
            logging.debug("[master] initialized empty GPKG with dummy layer (removed).")

    if not is_valid_geopackage(mosaic_gpkg_path):
        raise RuntimeError(f"{mosaic_gpkg_path} is not a valid GeoPackage after initialization")

    # ---------- 2) add per-tile master table ----------
    with sqlite3.connect(mosaic_gpkg_path) as conn:
        with sqlite_fast_writes(conn):
            write_table(conn, "global_stats", gs_df, pk_name="id")

    # ---------- 3) WHOLE-MOSAIC SUMMARY ----------
    # Totals (additive)
    sum_nodes           = int(gs_df["num_graph_nodes"].sum())
    sum_edges_graph     = int(gs_df["num_graph_edges"].sum())
    sum_endnodes        = int(gs_df["end_nodes_count"].sum())
    sum_junctions       = int(gs_df["junction_nodes_count"].sum())
    sum_components      = int(gs_df["num_components"].sum())
    sum_total_tcn_m     = float(gs_df["total_tcn_length_m"].sum())
    sum_num_edges_feat  = int(gs_df["num_edges"].sum())  # branch features
    max_comp_size_m     = float(gs_df["largest_component_size_m"].max()) if not gs_df.empty else 0.0

    # Correct global average node degree
    avg_node_degree_global = (2.0 * sum_edges_graph / sum_nodes) if sum_nodes > 0 else 0.0

    # Average component size (components-weighted)
    avg_comp_size_m = (sum_total_tcn_m / sum_components) if sum_components > 0 else 0.0

    # Approx edge-weighted mean edge length (p95 trimmed within tiles)
    if sum_num_edges_feat > 0:
        avg_edge_len_m = float((gs_df["num_edges"] * gs_df["mean_edge_length_m"]).sum() / sum_num_edges_feat)
    else:
        avg_edge_len_m = 0.0

    # Axial mean orientation (step-weighted only): weights = num_graph_edges
    step_pairs = list(zip(gs_df["num_graph_edges"].tolist(),
                          gs_df["mean_edge_orientation_deg_stepweighted"].tolist()))
    theta_step_global = combine_axial_deg(step_pairs)

    # Optional per-tile averages for counts (convenience)
    n_tiles = int(len(gs_df))
    mean_nodes_per_tile      = (sum_nodes / n_tiles) if n_tiles else 0.0
    mean_edges_graph_per_tile= (sum_edges_graph / n_tiles) if n_tiles else 0.0
    mean_endnodes_per_tile   = (sum_endnodes / n_tiles) if n_tiles else 0.0
    mean_components_per_tile = (sum_components / n_tiles) if n_tiles else 0.0
    mean_total_tcn_per_tile  = (sum_total_tcn_m / n_tiles) if n_tiles else 0.0

    summary_row = {
        "id": 1,
        "id_version": ID_VERSION,
        "raster_crs_wkt": target_crs_wkt or "",
        # headline totals
        "total_num_tiles": n_tiles,
        "total_num_graph_nodes": sum_nodes,
        "total_num_graph_edges": sum_edges_graph,
        "total_end_nodes_count": sum_endnodes,
        "total_junction_nodes_count": sum_junctions,
        "total_num_components": sum_components,
        "total_num_edges": sum_num_edges_feat,  # branch features
        "total_tcn_length_m": sum_total_tcn_m,
        "largest_component_size_m": max_comp_size_m,
        # properly weighted/combined averages
        "average_component_size_m": avg_comp_size_m,
        "average_node_degree": avg_node_degree_global,
        "average_edge_length_m_p95_approx": avg_edge_len_m,
        "mean_edge_orientation_deg_stepweighted": theta_step_global,
        # optional per-tile means
        "mean_num_graph_nodes_per_tile": mean_nodes_per_tile,
        "mean_num_graph_edges_per_tile": mean_edges_graph_per_tile,
        "mean_end_nodes_per_tile": mean_endnodes_per_tile,
        "mean_num_components_per_tile": mean_components_per_tile,
        "mean_total_tcn_length_m_per_tile": mean_total_tcn_per_tile,
    }

    summary_df = pd.DataFrame([summary_row], columns=[
        "id","id_version","raster_crs_wkt",
        "total_num_tiles",
        "total_num_graph_nodes","total_num_graph_edges",
        "total_end_nodes_count","total_junction_nodes_count",
        "total_num_components","total_num_edges",
        "total_tcn_length_m","largest_component_size_m",
        "average_component_size_m","average_node_degree",
        "average_edge_length_m_p95_approx",
        "mean_edge_orientation_deg_stepweighted",
        "mean_num_graph_nodes_per_tile","mean_num_graph_edges_per_tile",
        "mean_end_nodes_per_tile","mean_num_components_per_tile",
        "mean_total_tcn_length_m_per_tile"
    ])

    with sqlite3.connect(mosaic_gpkg_path) as conn:
        with sqlite_fast_writes(conn):
            write_table(conn, "global_stats_summary", summary_df, pk_name="id")

            # attach an XML summary for the whole mosaic
            x = summary_row
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<WholeMosaicSummary>
  <CRS note="Master polygons are reprojected to raster CRS for consistency">
    <RasterCRS_WKT><![CDATA[{x['raster_crs_wkt']}]]></RasterCRS_WKT>
    <Units>meters</Units>
  </CRS>
  <IDs>
    <IDVersion>{x['id_version']}</IDVersion>
  </IDs>
  <Totals>
    <NumTiles>{x['total_num_tiles']}</NumTiles>
    <NumGraphNodes>{x['total_num_graph_nodes']}</NumGraphNodes>
    <NumGraphEdges>{x['total_num_graph_edges']}</NumGraphEdges>
    <EndNodesCount>{x['total_end_nodes_count']}</EndNodesCount>
    <JunctionNodesCount>{x['total_junction_nodes_count']}</JunctionNodesCount>
    <NumComponents>{x['total_num_components']}</NumComponents>
    <NumEdges>{x['total_num_edges']}</NumEdges>
    <TotalTCNLength_m>{x['total_tcn_length_m']:.6f}</TotalTCNLength_m>
    <LargestComponentSize_m>{x['largest_component_size_m']:.6f}</LargestComponentSize_m>
  </Totals>
  <Averages>
    <AverageComponentSize_m>{x['average_component_size_m']:.6f}</AverageComponentSize_m>
    <AverageNodeDegree>{x['average_node_degree']:.6f}</AverageNodeDegree>
    <AverageEdgeLength_p95Approx_m>{x['average_edge_length_m_p95_approx']:.6f}</AverageEdgeLength_p95Approx_m>
    <MeanEdgeOrientation_StepWeighted_deg>{x['mean_edge_orientation_deg_stepweighted']:.6f}</MeanEdgeOrientation_StepWeighted_deg>
  </Averages>
</WholeMosaicSummary>
""".strip()
            insert_xml_metadata(conn, xml, links=[{"reference_scope": "geopackage"}])

    # ---------- per-tile XML linked to stats + polygon ----------
    with sqlite3.connect(mosaic_gpkg_path) as conn:
        with sqlite_fast_writes(conn):
            cur = conn.cursor()
            try:
                cur.execute("PRAGMA table_info('global_poly');")
                schema_rows = cur.fetchall()
                pk_cols = [r[1] for r in schema_rows if r[5] == 1]
                poly_pk = pk_cols[0] if pk_cols else "fid"
            except Exception:
                poly_pk = "fid"

            for r in rows:
                x = r
                xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<TileTCNSummary tile_id="{x['tile_id']}">
  <CRS note="All master polygons are reprojected to raster CRS to match per-tile layers">
    <RasterCRS_WKT><![CDATA[{x['raster_crs_wkt']}]]></RasterCRS_WKT>
    <Units>meters</Units>
  </CRS>
  <IDs>
    <IDVersion>{x['id_version']}</IDVersion>
  </IDs>
  <Totals>
    <NumComponents>{x['num_components']}</NumComponents>
    <TotalTCNLength_m>{x['total_tcn_length_m']:.6f}</TotalTCNLength_m>
    <LargestComponentSize_m>{x['largest_component_size_m']:.6f}</LargestComponentSize_m>
    <AverageComponentSize_m>{x['average_component_size_m']:.6f}</AverageComponentSize_m>
    <NumGraphNodes>{x['num_graph_nodes']}</NumGraphNodes>
    <NumGraphEdges>{x['num_graph_edges']}</NumGraphEdges>
    <EndNodesCount>{x['end_nodes_count']}</EndNodesCount>
    <JunctionNodesCount>{x['junction_nodes_count']}</JunctionNodesCount>
    <AverageNodeDegree>{x['average_node_degree']:.6f}</AverageNodeDegree>
    <MeanEdgeLength_m>{x['mean_edge_length_m']:.6f}</MeanEdgeLength_m>
    <MeanEdgeOrientation_StepWeighted_deg>{x['mean_edge_orientation_deg_stepweighted']:.6f}</MeanEdgeOrientation_StepWeighted_deg>
    <NumEdges>{x['num_edges']}</NumEdges>
  </Totals>
</TileTCNSummary>
""".strip()
                poly_rowid = None
                try:
                    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='global_poly';")
                    if cur.fetchone():
                        cur.execute(f"SELECT {poly_pk} FROM global_poly WHERE id = ?", (x["id"],))
                        got = cur.fetchone()
                        poly_rowid = got[0] if got else None
                except Exception:
                    poly_rowid = None

                links = [{"reference_scope": "row", "table_name": "global_stats", "row_id_value": x["id"]}]
                if poly_rowid is not None:
                    links.append({"reference_scope": "row", "table_name": "global_poly", "row_id_value": poly_rowid})
                insert_xml_metadata(conn, xml, links=links)

import sqlite3
from osgeo import ogr, gdal, osr

def check_gpkg_crs_all_layers(gpkg_path: str, target_srs: str) -> list[str]:
    """
    Validate that all vector layers and raster tables in a GeoPackage match target_srs.
    Returns a list of human-readable problems (empty list => PASS).
    """
    problems: list[str] = []

    # Prepare target SRS
    t = osr.SpatialReference()
    t.SetFromUserInput(target_srs)

    # ---- Vector layers (OGR) ----
    ds = ogr.Open(gpkg_path, update=0)
    if ds is None:
        return [f"Cannot open GPKG with OGR: {gpkg_path}"]
    try:
        for i in range(ds.GetLayerCount()):
            lyr = ds.GetLayer(i)
            if lyr is None:
                continue
            name = lyr.GetName()
            srs = lyr.GetSpatialRef()
            if srs is None:
                problems.append(f"VECTOR CRS MISSING: layer='{name}'")
                continue
            if srs.IsSame(t) != 1:
                problems.append(f"VECTOR CRS MISMATCH: layer='{name}' expected={target_srs}")
    finally:
        ds = None

    # ---- Raster tables (SQLite + GDAL) ----
    try:
        with sqlite3.connect(gpkg_path) as con:
            rows = con.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type='tiles'"
            ).fetchall()
            raster_tables = [r[0] for r in rows]
    except Exception as e:
        problems.append(f"Cannot query gpkg_contents for rasters: {e}")
        raster_tables = []

    for rt in raster_tables:
        rds = gdal.Open(f"GPKG:{gpkg_path}:{rt}", gdal.GA_ReadOnly)
        if rds is None:
            problems.append(f"RASTER OPEN FAIL: table='{rt}'")
            continue
        wkt = rds.GetProjection() or ""
        rds = None

        s = osr.SpatialReference()
        if not wkt:
            problems.append(f"RASTER CRS MISSING: table='{rt}'")
            continue
        try:
            s.ImportFromWkt(wkt)
        except Exception:
            problems.append(f"RASTER CRS PARSE FAIL: table='{rt}'")
            continue

        if s.IsSame(t) != 1:
            problems.append(f"RASTER CRS MISMATCH: table='{rt}' expected={target_srs}")

    return problems

from pathlib import Path
from typing import Optional

from pathlib import Path
from typing import Optional, Sequence

def build_tile_structured_path(
    subtile_path: str | Path,
    root: str | Path,
    *,
    tile_id: Optional[str] = None,
    postfix: Optional[str] = None,
    ext: Optional[str] = None,
    strip_postfixes: Sequence[str] = ("_mask.tif",),  # default: clean current mask postfix
    preserve_rel_under_tile: bool = False,
) -> str:
    """
    Build a path under `root` mirroring the tile/subtile structure of `subtile_path`.

    Enhancements:
      - `strip_postfixes` removes any known postfixes from the source filename
        BEFORE appending the desired `postfix`. This prevents duplicated suffixes
        like *_mask_mask.tif.

    Args:
        subtile_path: Existing subtile path (used as the "key").
        root: Destination root directory.
        tile_id: If provided, use it; else infer from subtile_path.parent.name.
        postfix: Desired postfix to append (e.g., "_mask.tif", "_TCN.gpkg").
        ext: Optional forced extension if postfix is not provided (".tif", ".gpkg", ".json").
        strip_postfixes: Postfixes to remove from the end of the filename before appending `postfix`.
                         Defaults to ("_mask.tif",).
        preserve_rel_under_tile: If True, preserve any subfolders under <tile_id>.
    """
    p = Path(subtile_path)
    r = Path(root)

    tid = tile_id or p.parent.name

    # optional relative subdir preservation
    rel_dir = Path()
    if preserve_rel_under_tile:
        parts = p.parts
        if tid in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index(tid)
            rel_dir = Path(*parts[idx + 1 : -1])

    # ---- derive a "clean base" name from the source filename ----
    name = p.name

    # strip any known postfixes (longest first avoids partial stripping issues)
    for sp in sorted(strip_postfixes or (), key=len, reverse=True):
        if sp and name.endswith(sp):
            name = name[: -len(sp)]
            break

    # If we stripped something like "_mask.tif", `name` now ends with the stem portion.
    # Ensure `name` has no trailing dot leftovers (defensive).
    if name.endswith("."):
        name = name[:-1]

    # If `name` still has an extension (e.g., ".tif"), remove it so postfix/ext logic is consistent.
    base = Path(name).stem if Path(name).suffix else name

    # ---- apply requested postfix/ext ----
    if postfix:
        out_name = f"{base}{postfix}"
    elif ext:
        ext2 = ext if ext.startswith(".") else f".{ext}"
        out_name = f"{base}{ext2}"
    else:
        out_name = p.name  # unchanged

    return str(r / tid / rel_dir / out_name)

