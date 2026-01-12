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

import sys
import gc
import glob
import time
import argparse
import multiprocessing as mp
from contextlib import contextmanager
from math import sin, cos, radians, atan2, degrees

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from shapely.ops import unary_union

import rasterio
from rasterio.transform import xy as rio_xy
from skimage.morphology import skeletonize
from skan.csr import pixel_graph
import networkx as nx
import psutil

# ---- SciPy optional (fast path) ----
try:
    from scipy.sparse.csgraph import connected_components as csgraph_connected_components
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ------------------------------ config ------------------------------

DEFAULT_PIXEL_M  = 0.5
HARD_MAX_WORKERS = 32
ID_VERSION = "v1.0-deterministic-rcsorted-neighsorted-compsorted"
COMPS_HEARTBEAT = 20000  # progress log every N components

# ------------------------------ small logging helpers ------------------------------

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

# ------------------------------ math helpers ------------------------------

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

def fast_pix_to_xy_affine(trf, rr, cc):
    """Vectorized pixel-center to map coords using the affine transform."""
    rr = np.asarray(rr, dtype=float)
    cc = np.asarray(cc, dtype=float)
    x = trf.c + (cc + 0.5) * trf.a + (rr + 0.5) * trf.b
    y = trf.f + (cc + 0.5) * trf.d + (rr + 0.5) * trf.e
    return x, y

# ------------------------------ sqlite helpers ------------------------------

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

@contextmanager
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

def get_sub_tile_id(tif_path: str) -> str:
    return os.path.splitext(os.path.basename(tif_path))[0]

#---------- resume gpkg code helpers -------------------------------
# --- put near your other utilities ---
from typing import Sequence, Tuple, List

def _expected_gpkg_name_for_tif(tif_path: str, gpkg_suffix: str = "_TCN.gpkg") -> str:
    """ArcticMosaic_58_16_1_1_mask.tif -> ArcticMosaic_58_16_1_1_TCN.gpkg"""
    stem = Path(tif_path).stem
    stem = re.sub(r'[_-]mask$', '', stem, flags=re.I)
    return f"{stem}{gpkg_suffix}"

#---------------- Sanity check for gpkg created by pipe line to decide to redo
import sqlite3


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

#</END gpkg sanity check>----Sanity check for gpkg created by pipe line to decide to redo

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

# ------------------------------ per-tile processing updated to include the sub_cell gpkg--
# --- NEW: helpers for per-file GPKG writing ---
def _records_to_gdf(records, crs_wkt):
    """Convert list[dict] with 'geometry' into a GeoDataFrame or None if empty."""
    if not records:
        return None
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs_wkt or None)


from lib import add_from_mask_rastors as AdMsk, add_from_gpkg_layers as AL
import update_in_exp_cover as UpExpCov

EXTRA_GPKG_LAYERS = ["input_image_cutlines"]
NEW_LAYER_NAMES = ["InputMosaicCutlinesVector"]

def _write_per_file_gpkg(tile_dir, tif_path, crs_wkt,
                         nodes_records_f, edges_records_f, components_records_f,
                         verbose=False, tile_id=None):
    """Write per-file layers {nodes, edges, components} into <tile_dir>/<basename>.gpkg."""
    os.makedirs(tile_dir, exist_ok=True)
    #base = os.path.splitext(os.path.basename(tif_path))[0] + ".gpkg"
    stem = Path(tif_path).stem  # e.g., "ArcticMosaic_58_16_1_1_mask"
    stem = re.sub(r'[_-]mask$', '', stem, flags=re.I)  # → "ArcticMosaic_58_16_1_1"
    base = f"{stem}_TCN.gpkg"  # → "ArcticMosaic_58_16_1_1_TCN.gpkg"
    out_gpkg = os.path.join(tile_dir, base)

    # If a previous partial/old run left a GPKG, remove it (and SQLite sidecars) and rebuild fresh
    if os.path.exists(out_gpkg):
        for suf in ("", "-wal", "-shm"):  # SQLite may create these
            p = out_gpkg + suf
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                logging.warning(f"Could not remove {p}: {e}")

    nodes_gdf = _records_to_gdf(nodes_records_f, crs_wkt)
    edges_gdf = _records_to_gdf(edges_records_f, crs_wkt)
    comps_gdf = _records_to_gdf(components_records_f, crs_wkt)

    if all(g is None or g.empty for g in (nodes_gdf, edges_gdf, comps_gdf)):
        # nothing to write for this source file
        return

    written = False
    for layer_name, gdf in (("GraphTheoraticNodes", nodes_gdf), ("GraphTheoraticEdges", edges_gdf), ("GraphTheoraticComponents", comps_gdf)):
        if gdf is None or gdf.empty:
            continue
        gdf.to_file(out_gpkg, layer=layer_name, driver="GPKG", mode=("w" if not written else "a"))
        written = True

    if verbose:
        logging.debug(f"[{tile_id or ''}] created per-file GPKG: {out_gpkg}")

    logp(verbose, logging.DEBUG, f" STARTING PER FILE GPKG ADDING to n,e,c LAYERS{EXTRA_GPKG_LAYERS}")  ###########################################
    # ADDING layers from old gpkg ---------------------
    # --- append extra layers from the existing per-file GPKG with renaming ---

    if EXTRA_GPKG_LAYERS:
        if len(NEW_LAYER_NAMES) not in (0, len(EXTRA_GPKG_LAYERS)):
            logging.warning("NEW_LAYER_NAMES length does not match EXTRA_GPKG_LAYERS; falling back to same names.")
        for i, src_layer in enumerate(EXTRA_GPKG_LAYERS):
            logp(verbose, logging.DEBUG, f" ############# ADDING {src_layer}") ###########################################
            dst_layer = (
                NEW_LAYER_NAMES[i] if (i < len(NEW_LAYER_NAMES)) else src_layer
            )
            # avoid clobbering core layer names
            if dst_layer in {"GraphTheoraticNodes", "GraphTheoraticEdges", "GraphTheoraticComponents"}:
                dst_layer = f"ext_{dst_layer}"
            old_gpkg_path = AL.tiff_to_gpkg_path(tif_path)
            logging.debug(f"OLD GPKG {old_gpkg_path}")
            logging.debug(f"NEW GPKG {out_gpkg}")
            AL.copy_layer(out_gpkg,old_gpkg_path,src_layer,dst_layer)
    if verbose:
        logging.debug(f"[{tile_id or ''}] wrote added layers to per-file GPKG: {out_gpkg}")

    org_mask_path = AL.tiff_to_gpkg_path(tif_path,old_gpkg_root="/scratch2/projects/PDG_shared/AlaskaTundraMosaicMasks/",
                          ensure_parent=False, postfix="_mask.tif")

    AdMsk.add_masks_and_vector(
        output_gpkg=out_gpkg,
        orig_mask_tif=org_mask_path, # or None if not needed
        cleaned_mask_tif=tif_path,
        target_srs="EPSG:3338",  # enforce package CRS
        assume_src_srs="EPSG:3413",  # only if your inputs sometimes lack CRS
        build_overviews=(2, 4, 8, 16),
        eight_connected=True,
        min_area=0.5,
        simplify_tol=0.25
    )
    if verbose:
        logging.debug(f"[{tile_id or ''}] wrote added layers nlcd cleaned to per-file GPKG: {out_gpkg}")
    UpExpCov.add_date_exposure_gaussian(
        gpkg_path=out_gpkg,  # same GPKG you just wrote
        cutline_layer="InputMosaicCutlinesVector",
        date_field="ACQDATE",  # your field
        mean_month=7, mean_day=15,  # the “mean date” (e.g., July 15)
        sigma_days=45, # spread of the Gaussian in days
        nodes_layer="GraphTheoraticNodes",
        edges_layer="GraphTheoraticEdges",
        comps_layer="GraphTheoraticComponents",
        edges_length_field="length_m",
        comps_length_field="total_length_m",
        target_crs="EPSG:3338"  # keep everything in your package CRS
    )
    if verbose:
        logging.debug(f"[{tile_id or ''}] upated exposure cover : {out_gpkg}")

#----- NEW write per file to resume from previous work -------------------

from pathlib import Path
import os, re, logging
from osgeo import ogr

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


#------------------------------ per-tile processing updated..............
def process_one_tile(args):
    """Build <tile_id>.gpkg and return per-tile global_stats dict for the master build."""
    tile_id, tif_paths, output_tiles_dir, verbose = args

    edges_records = []
    nodes_records = []
    components_records = []

    # arrays for efficient per-component metrics (filled later)
    edge_comp_ids = []
    edge_lengths  = []

    total_tcn_length_m_sum = 0.0
    num_graph_nodes_sum = 0
    num_graph_edges_sum = 0
    end_nodes_count_sum = 0
    junction_nodes_count_sum = 0
    branches_lengths_all = []
    component_lengths_all = []
    tile_step_orients = []   # per pixel-step orientations
    edge_orients_all = []    # per-edge axial orientation (kept in edges layer only)

    crs_wkt = None
    running_edge_id = 0
    running_node_id = 0

    tile_t0 = time.perf_counter()
    logp(verbose, logging.DEBUG, f"[{tile_id}] starting, {len(tif_paths)} tif(s)")

    # --- NEW: ensure a per-tile subfolder for per-TIF outputs ---
    tile_out_dir = os.path.join(output_tiles_dir, str(tile_id))
    os.makedirs(tile_out_dir, exist_ok=True)

    #tif_paths, skipped = filter_done_tiffs(tif_paths, tile_out_dir, verify=True, verbose=True)

    #-----</NEW>
    for tif_path in sorted(tif_paths):
        file_id = os.path.basename(tif_path)  # stable per-source file id
        pfx = f"[{tile_id}|{file_id}]"
        # --- NEW: per-file accumulators (separate from per-tile) ---
        edges_records_f = []
        nodes_records_f = []
        components_records_f = []
        #-----</NEW>
        try:
            with StageTimer(verbose, pfx, "OPEN"):
                with rasterio.open(tif_path) as src:
                    img = src.read(1).astype(np.uint8)
                    trf = src.transform
                    crs_wkt = crs_wkt or (src.crs.to_wkt() if src.crs else None)
                    px = abs(trf.a); py = abs(trf.e)
                    step_len_m = px if (px > 0 and py > 0 and abs(px - py) < 1e-9) else DEFAULT_PIXEL_M
        except Exception as e:
            logging.error(f"{pfx} open error: {e}")
            continue

        with StageTimer(verbose, pfx, "THRESH"):
            binary = img > 0
            if not np.any(binary):
                logp(verbose, logging.DEBUG, f"{pfx} empty mask → skip")
                continue

        with StageTimer(verbose, pfx, "SKEL"):
            skel = skeletonize(binary)

        total_tcn_length_m_sum += float(np.count_nonzero(skel) * step_len_m)

        with StageTimer(verbose, pfx, "GRAPH"):
            graph_csr, coords_or_ids = pixel_graph(skel.astype(bool), connectivity=2)
            if isinstance(coords_or_ids, np.ndarray) and coords_or_ids.ndim == 2 and coords_or_ids.shape[1] == 2:
                coords = coords_or_ids.astype(int)
            else:
                rr, cc = np.unravel_index(coords_or_ids, skel.shape)
                coords = np.column_stack([rr.astype(int), cc.astype(int)])

            idx_rc = [(idx, (int(r), int(c))) for idx, (r, c) in enumerate(coords)]
            idx_rc.sort(key=lambda t: (t[1][0], t[1][1]))
            id_remap = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(idx_rc)}
            rc_by_new = {new_idx: rc for new_idx, (_, rc) in enumerate(idx_rc)}

            # Reverse map: new_idx -> old_idx (for CSR-based components)
            old_by_new = np.empty(len(id_remap), dtype=np.int64)
            for old_idx, new_idx in id_remap.items():
                old_by_new[new_idx] = old_idx

            G = nx.Graph()
            for new_idx, rc in rc_by_new.items():
                G.add_node(new_idx, rc=rc)
            rows, cols = graph_csr.nonzero()
            for i, j in zip(rows, cols):
                if i == j:
                    continue
                ni, nj = id_remap[i], id_remap[j]
                if ni == nj:
                    continue
                if ni < nj:
                    G.add_edge(ni, nj)

        V = G.number_of_nodes()
        E = G.number_of_edges()
        num_graph_nodes_sum += V
        num_graph_edges_sum += E
        logp(verbose, logging.DEBUG, f"{pfx} graph nodes={V} edges={E}")

        degs = dict(G.degree())
        end_nodes_count_sum      += sum(1 for d in degs.values() if d == 1)
        junction_nodes_count_sum += sum(1 for d in degs.values() if d >= 2)

        with StageTimer(verbose, pfx, "STEP-ORIENTS"):
            for u, v in G.edges():
                (r0, c0) = G.nodes[u]["rc"]; (r1, c1) = G.nodes[v]["rc"]
                dy, dx = (r1 - r0), (c1 - c0)
                tile_step_orients.append((np.degrees(np.arctan2(dy, dx)) % 180.0))

        with StageTimer(verbose, pfx, "KEY NODES"):
            key_nodes = [n for n, d in degs.items() if d != 2]
            # cycles: ensure at least one key node
            comps_for_cycles = list(nx.connected_components(G))
            for comp_nodes in comps_for_cycles:
                if not comp_nodes:
                    continue
                if all(degs[n] == 2 for n in comp_nodes):
                    seed = min(comp_nodes)
                    if seed not in key_nodes:
                        key_nodes.append(seed)
            key_nodes.sort(key=lambda n: (G.nodes[n]["rc"][0], G.nodes[n]["rc"][1]))

            node_id_map = {}
            for n in key_nodes:
                node_id_map[n] = running_node_id
                r, c = G.nodes[n]["rc"]
                x, y = fast_pix_to_xy_affine(trf, [r], [c])  # faster
                degree = degs[n]
                nodes_records.append({
                    "tile_id": tile_id,
                    "file_id": file_id,
                    "component_id": -1,  # filled later
                    "global_component_id": None,  # filled later
                    "node_id": running_node_id,
                    "degree": int(degree),
                    "is_endpoint": int(degree == 1),
                    "is_junction": int(degree >= 2),
                    "geometry": sgeom.Point(float(x[0]), float(y[0]))
                })
                running_node_id += 1
            logp(verbose, logging.DEBUG, f"{pfx} key_nodes={len(key_nodes)}")

        visited_edges = set()
        branches = []
        with StageTimer(verbose, pfx, "TRACE"):
            for kn in key_nodes:
                neighs = sorted(G.neighbors(kn))
                for nbr in neighs:
                    e0 = (min(kn, nbr), max(kn, nbr))
                    if e0 in visited_edges:
                        continue
                    rc_chain = []
                    prev, curr = kn, nbr
                    pr, pc = G.nodes[prev]["rc"]; cr, cc = G.nodes[curr]["rc"]
                    rc_chain.append((pr, pc)); rc_chain.append((cr, cc))
                    visited_edges.add(e0)
                    while G.degree(curr) == 2:
                        n0, n1 = sorted(G.neighbors(curr))
                        nxt = n0 if n0 != prev else n1
                        e2 = (min(curr, nxt), max(curr, nxt))
                        if e2 in visited_edges:
                            break
                        visited_edges.add(e2)
                        prev, curr = curr, nxt
                        r, c = G.nodes[curr]["rc"]
                        rc_chain.append((r, c))

                    nx_a = kn
                    nx_b_is_key = (curr in key_nodes)
                    nx_b = curr if nx_b_is_key else None
                    node_a = node_id_map.get(nx_a, -1)
                    node_b = node_id_map.get(nx_b, -1)
                    is_open_edge = int(node_b == -1)

                    chain_pixels = len(rc_chain)
                    length_m = float(chain_pixels * step_len_m)

                    seg_orients = []
                    for i in range(chain_pixels - 1):
                        (r0, c0), (r1, c1) = rc_chain[i], rc_chain[i + 1]
                        seg_orients.append((np.degrees(np.arctan2(r1 - r0, c1 - c0)) % 180.0))
                    orient = axial_mean_deg(seg_orients)

                    branches.append({
                        "rc_chain": rc_chain,
                        "node_a": node_a,
                        "node_b": node_b,
                        "is_open_edge": is_open_edge,
                        "length_m": length_m,
                        "orientation": float(orient)
                    })
            logp(verbose, logging.DEBUG, f"{pfx} branches={len(branches)}")

        # ---------- Component labeling ----------
        if HAVE_SCIPY:
            n_comps, labels_old = csgraph_connected_components(graph_csr, directed=False)
            comp_id_by_new = labels_old[old_by_new]  # map to our node order
            num_comps = int(n_comps)
        else:
            comps = list(nx.connected_components(G))
            comps.sort(key=lambda s: min(s))
            comp_id_by_new = np.empty(V, dtype=np.int64)
            for cidx, comp_nodes in enumerate(comps):
                for n in comp_nodes:
                    comp_id_by_new[n] = cidx
            num_comps = len(comps)

        with StageTimer(verbose, pfx, "COMPS"):
            # lookups
            rc_to_nx = {G.nodes[n]["rc"]: n for n in G.nodes()}
            comp_to_lines = {}

            # per-component node stats (vectorized)
            deg_arr = np.fromiter((degs[i] for i in range(V)), dtype=np.int64)
            nodes_per_comp = np.bincount(comp_id_by_new, minlength=num_comps)
            end_per_comp   = np.bincount(comp_id_by_new[deg_arr == 1], minlength=num_comps)
            junc_per_comp  = np.bincount(comp_id_by_new[deg_arr >= 2], minlength=num_comps)
            sumdeg_per_comp = np.bincount(comp_id_by_new, weights=deg_arr, minlength=num_comps).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                avgdeg_per_comp = np.divide(sumdeg_per_comp, nodes_per_comp,
                                            out=np.zeros_like(sumdeg_per_comp), where=nodes_per_comp > 0)
            comp_len_m = nodes_per_comp.astype(np.float64) * float(step_len_m)

            # edges/lines
            for b in branches:
                nx_first = rc_to_nx[b["rc_chain"][0]]
                comp_id  = int(comp_id_by_new[nx_first])
                global_component_id = f"{tile_id}|{file_id}|{comp_id}"

                rr = np.fromiter((r for r, _ in b["rc_chain"]), dtype=float)
                cc = np.fromiter((c for _, c in b["rc_chain"]), dtype=float)
                xs, ys = fast_pix_to_xy_affine(trf, rr, cc)
                line = sgeom.LineString(list(zip(xs.tolist(), ys.tolist())))

                rec_edge = {
                    "tile_id": tile_id,
                    "file_id": file_id,
                    "component_id": comp_id,
                    "global_component_id": global_component_id,
                    "edge_id": running_edge_id,
                    "node_id_a": int(b["node_a"]),
                    "node_id_b": int(b["node_b"]),
                    "is_open_edge": int(b["is_open_edge"]),
                    "length_m": float(b["length_m"]),
                    "orientation_deg_axial": float(b["orientation"]),
                    "geometry": line
                }
                edges_records.append(rec_edge)        # existing per-tile
                edges_records_f.append(rec_edge)      # NEW per-file
                running_edge_id += 1

                edge_comp_ids.append(comp_id)
                edge_lengths.append(float(b["length_m"]))
                branches_lengths_all.append(b["length_m"])
                edge_orients_all.append(b["orientation"])
                comp_to_lines.setdefault(comp_id, []).append(line)

            # replace placeholder key-node block
            last_block_size = len({v for v in node_id_map.values()})
            if last_block_size:
                nodes_records = nodes_records[:-last_block_size]
            for n, nid in sorted(node_id_map.items(), key=lambda kv: kv[1]):
                r, c = G.nodes[n]["rc"]
                x, y = fast_pix_to_xy_affine(trf, [r], [c])
                degree = degs[n]
                comp_id = int(comp_id_by_new[n])
                global_component_id = f"{tile_id}|{file_id}|{comp_id}"
                rec_node = {
                    "tile_id": tile_id,
                    "file_id": file_id,
                    "component_id": comp_id,
                    "global_component_id": global_component_id,
                    "node_id": nid,
                    "degree": int(degree),
                    "is_endpoint": int(degree == 1),
                    "is_junction": int(degree >= 2),
                    "geometry": sgeom.Point(float(x[0]), float(y[0]))
                }
                nodes_records.append(rec_node)       # existing per-tile
                nodes_records_f.append(rec_node)     # NEW per-file

            # p95-trimmed mean branch length per component (used only in components layer)
            _edge_df = pd.DataFrame({"comp": edge_comp_ids, "len": edge_lengths})
            if not _edge_df.empty:
                p95 = _edge_df.groupby("comp")["len"].quantile(0.95)
                _edge_df = _edge_df.join(p95, on="comp", rsuffix="_p95")
                _edge_df = _edge_df[_edge_df["len"] <= _edge_df["len_p95"]]
                mean_b95_by_comp = _edge_df.groupby("comp")["len"].mean()
            else:
                mean_b95_by_comp = pd.Series(dtype=float)

            # components layer
            # find single node components via a single compute
            isolates = (nodes_per_comp == 1) & (end_per_comp == 0) & (junc_per_comp == 0)

            for cidx in range(num_comps):
                if (cidx % COMPS_HEARTBEAT) == 0 and cidx:
                    logp(verbose, logging.DEBUG, f"{pfx} COMPS progress {cidx}/{num_comps}")

                lines = comp_to_lines.get(cidx, [])
                # find single node components
                if isolates[cidx] and not lines:
                    continue
                if not lines:
                    ml = None
                else:
                    # Force MULTILINESTRING even when there's only a single branch
                    # (prevents a separate "LineString" sublayer in QGIS)
                    ml = sgeom.MultiLineString([lines[0]] if len(lines) == 1 else lines)

                #ml = sgeom.MultiLineString(lines) if lines else sgeom.GeometryCollection()

                rec_comp = {
                    "tile_id": tile_id,
                    "file_id": file_id,
                    "component_local": int(cidx),
                    "component_id": int(cidx),
                    "global_component_id": f"{tile_id}|{file_id}|{int(cidx)}",
                    "total_length_m": float(comp_len_m[cidx]),
                    "num_edges": int(len(lines)),
                    "num_nodes": int(nodes_per_comp[cidx]),
                    "num_endnodes": int(end_per_comp[cidx]),
                    "num_junctions": int(junc_per_comp[cidx]),
                    "mean_branch_len_95_m": float(mean_b95_by_comp.get(cidx, 0.0)),
                    "avg_degree": float(avgdeg_per_comp[cidx]),
                    "geometry": ml
                }
                components_records.append(rec_comp)  # existing per-tile
                components_records_f.append(rec_comp)  # NEW per-file

            component_lengths_all.extend(comp_len_m.tolist())
        # --- NEW: write per-file GPKG for this source TIF ---
        logp(verbose, logging.DEBUG, f" STARTING PER FILE GPKG ") ###########################################
        try:
            _write_per_file_gpkg(
                tile_out_dir, tif_path, crs_wkt,
                nodes_records_f, edges_records_f, components_records_f,
                verbose=verbose, tile_id=tile_id
            )
        except Exception as e:
            logp(verbose, logging.DEBUG, f" PER FILE GPKG write ERROR ")  ###########################################
            logging.error(f"{pfx} per-file GPKG write error: {e}")

        del img, binary, skel, graph_csr, coords_or_ids, coords, G
        gc.collect()
        logp(verbose, logging.DEBUG, f"{pfx} done")

    if (len(edges_records) + len(nodes_records) + len(components_records)) == 0:
        logging.error(f"[{tile_id}] no records created; returning None")
        return None
    # (Make sure this line executes OUTSIDE the for-each-TIF loop.)
    if DEFER_TILE_AGG:
        tile_dt = time.perf_counter() - tile_t0
        logp(verbose, logging.DEBUG,
             f"[{tile_id}] DEFER per-tile aggregation & GPKG writes (per-file GPKGs done) in {tile_dt:.2f}s")
        # Return a minimal row so the caller can record progress; adjust fields as you like.
        return {
            "id": None,
            "tile_id": tile_id,
            "id_version": ID_VERSION,
            "raster_crs_wkt": crs_wkt or "",
            "deferred": True
        }
    # ===== END GUARD deffer AGG=====

    if branches_lengths_all:
        p95 = np.percentile(branches_lengths_all, 95)
        trimmed = [L for L in branches_lengths_all if L <= p95]
        mean_edge_length_m = float(np.mean(trimmed)) if trimmed else 0.0
    else:
        mean_edge_length_m = 0.0

    largest_component_size_m = float(np.max(component_lengths_all)) if component_lengths_all else 0.0
    average_component_size_m = float(np.mean(component_lengths_all)) if component_lengths_all else 0.0
    average_node_degree = (2.0 * num_graph_edges_sum / num_graph_nodes_sum) if num_graph_nodes_sum > 0 else 0.0

    mean_edge_orientation_deg_stepweighted = float(axial_mean_deg(tile_step_orients)) if tile_step_orients else -1.0
    # NOTE: we DROP edge-weighted orientation entirely by decision

    tile_row = {
        "id": 1,
        "tile_id": tile_id,
        "id_version": ID_VERSION,
        "raster_crs_wkt": crs_wkt or "",
        "num_components": int(len(component_lengths_all)),
        "total_tcn_length_m": float(total_tcn_length_m_sum),
        "largest_component_size_m": largest_component_size_m,
        "average_component_size_m": average_component_size_m,
        "num_graph_nodes": int(num_graph_nodes_sum),
        "num_graph_edges": int(num_graph_edges_sum),
        "end_nodes_count": int(end_nodes_count_sum),
        "junction_nodes_count": int(junction_nodes_count_sum),
        "average_node_degree": float(average_node_degree),
        "mean_edge_length_m": float(mean_edge_length_m),
        "mean_edge_orientation_deg_stepweighted": float(mean_edge_orientation_deg_stepweighted),
        "num_edges": int(len(edges_records))
    }

    gpkg_path = os.path.join(output_tiles_dir, f"{tile_id}.gpkg")
    os.makedirs(output_tiles_dir, exist_ok=True)

    def append_layer(layer_name, records, crs_wkt):
        if not records:
            return
        gdf = gpd.GeoDataFrame(
            [{k: v for k, v in r.items() if k != "geometry"} for r in records],
            geometry=[r["geometry"] for r in records],
            crs=crs_wkt
        )
        mode = "a" if os.path.exists(gpkg_path) else "w"
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode)
        logging.info(f"Created {len(records):,} records in layer '{layer_name}'")

    with StageTimer(verbose, f"[{tile_id}]", "WRITE layers"):
        append_layer("GraphTheoraticEdges", edges_records, crs_wkt)
        append_layer("GraphTheoraticNodes", nodes_records, crs_wkt)
        append_layer("GraphTheoraticComponents", components_records, crs_wkt)

    with sqlite3.connect(gpkg_path) as conn:
        with sqlite_fast_writes(conn):
            row_df = pd.DataFrame([tile_row])
            with StageTimer(verbose, f"[{tile_id}]", "WRITE global_stats + XML"):
                write_table(conn, "global_stats", row_df[[
                    "id","tile_id","id_version","raster_crs_wkt",
                    "num_components","total_tcn_length_m",
                    "largest_component_size_m","average_component_size_m",
                    "num_graph_nodes","num_graph_edges",
                    "end_nodes_count","junction_nodes_count",
                    "average_node_degree","mean_edge_length_m",
                    "mean_edge_orientation_deg_stepweighted",
                    "num_edges"
                ]], pk_name="id")

                x = tile_row
                xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<TileTCNSummary tile_id="{x['tile_id']}">
  <CRS note="All per-tile layers and stats use the raster CRS">
    <RasterCRS_WKT><![CDATA[{x['raster_crs_wkt']}]]></RasterCRS_WKT>
    <Units>meters</Units>
  </CRS>
  <IDs>
    <IDVersion>{x['id_version']}</IDVersion>
    <Keys>
      <Components>composite: (tile_id, file_id, component_id); also global_component_id = tile_id|file_id|component_id</Components>
      <Edges>composite: (tile_id, file_id, edge_id)</Edges>
      <Nodes>composite: (tile_id, file_id, node_id)</Nodes>
    </Keys>
  </IDs>
  <Semantics>
    <NodeClasses>endpoint: degree==1; junction: degree>=2; isolates (degree==0) are neither</NodeClasses>
    <Lengths>
      <ComponentAndTile>pixel-accurate: (# skeleton pixels) * step_len_m; no double-counting at junctions</ComponentAndTile>
      <Edges>per-edge length (analysis only); sum of edge lengths may exceed component/tile totals</Edges>
    </Lengths>
    <Orientation>
      <EdgeField>orientation_deg_axial (per-edge axial mean)</EdgeField>
      <Aggregate>
        <StepWeighted>mean_edge_orientation_deg_stepweighted (each pixel-step contributes equally)</StepWeighted>
      </Aggregate>
    </Orientation>
    <OpenEdges>is_open_edge=1 means node_id_b==-1 (edge terminates without a key-node)</OpenEdges>
  </Semantics>
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
                insert_xml_metadata(conn, xml, links=[
                    {"reference_scope": "row", "table_name": "global_stats", "row_id_value": 1},
                    {"reference_scope": "geopackage"}
                ])

    tile_dt = time.perf_counter() - tile_t0
    logp(verbose, logging.DEBUG, f"[{tile_id}] finished in {tile_dt:.2f}s")

    out_row = dict(tile_row)
    out_row["id"] = None
    return out_row

# ------------------------------ master build ------------------------------

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

# ------------------------------ startup sanity ------------------------------

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

DEFER_TILE_AGG = True # Deffer tile aggregation
# ------------------------------ main ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master_dir", required=True, help="Folder of tiles; each subfolder is a tile_id with *.tif masks")
    p.add_argument("--output_tiles_dir", required=True, help="Output dir for per-tile GeoPackages (<tile_id>.gpkg)")
    p.add_argument("--mosaic_gpkg", required=True, help="Output path for master GeoPackage (whole_mosaic.gpkg)")
    p.add_argument("--imagery_dir", required=True, help="Folder with per-tile shapefiles ArcticMosaic_<tile_id>_tiles.shp")
    p.add_argument("--workers", type=int, default=0, help="Processes; 0=auto (all-1, mem-capped)")
    p.add_argument("--max_gb_per_worker", type=float, default=64, help="Estimated GB RAM per worker")
    # Selecting the tile sequence in rank order
    p.add_argument("--tile_rank_st", type=int, default=0, help="Processes tiles starting from ")
    p.add_argument("--tile_rank_end", type=int, default=0, help="Processes tiles end at ")

    p.add_argument("--defer-tile-agg", action="store_true",
                    help="Skip per-tile stats and GPKG writes; only write per-file GPKGs")
    p.add_argument("--tifs_per_run", type=int, default=35, help="Max number of tiffs to process")

    # testing controls
    p.add_argument("--one_tile", type=str, default=None, help="Only process this tile_id (subfolder name)")
    p.add_argument("--one_tif", action="store_true", help="Within each selected tile, only process the first .tif")
    args = p.parse_args()

    global DEFER_TILE_AGG
    DEFER_TILE_AGG = args.defer_tile_agg

    #python gt_gpkg.py --master_dir ../masked --output_tiles_dir ./test_tile_dir --mosaic_gpkg ./test_mosaic_dir, --imagery_dir /scratch2/projects/PDG_shared/AlaskaTundraMosaic/imagery --one_tile 43_17
    #python gt_gpkg.py --master_dir ../masked --output_tiles_dir ./test_tile_dir --mosaic_gpkg ./test_mosaic.gpkg --imagery_dir /scratch2/projects/PDG_shared/AlaskaTundraMosaic/imagery --one_tile 43_17 --one_tif

    # logging early, force=True so it shows even if already configured by env
    smart_verbose = bool(args.one_tile and args.one_tif)
    logging.basicConfig(
        level=logging.DEBUG if smart_verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )

    logging.info("=== gt.py startup ===")
    logging.info(f"Python: {sys.version.split()[0]} | SciPy fast-path: {'ON' if HAVE_SCIPY else 'OFF'}")

    if not sanity_check_paths(args.master_dir, args.imagery_dir):
        return

    # ---------- discover all tiles ----------
    all_tiles = [d for d in sorted(os.listdir(args.master_dir))
                 if os.path.isdir(os.path.join(args.master_dir, d))]

    tiles = all_tiles[:]  # start with all

    # ---------- rank/world partition (round-robin) ----------
    rank = int(getattr(args, "rank", 1))
    world = int(getattr(args, "world_size", 1))
    if world <= 0:
        world = 1

    rank0 = rank - 1  # 0-based
    tiles = [t for i, t in enumerate(tiles) if (i % world) == rank0]
    logging.info(f"[rank {rank}/{world}] assigned {len(tiles)} tiles")

    # ---------- optional manual slice override takes precedence ----------
    st = getattr(args, "tile_rank_st", None)
    en = getattr(args, "tile_rank_end", None)
    if (st is not None) or (en is not None):
        n = len(tiles)
        st = 1 if st is None else int(st)
        en = n if en is None else int(en)
        st_i, en_i = max(0, st - 1), min(n, en)
        tiles = tiles[st_i:en_i] if st_i < en_i else []
        logging.info(f"[rank {rank}] slice override {st}-{en} → {len(tiles)} tiles")

    # ---------- determine cap BEFORE building tasks ----------
    cap_per_run = int(getattr(args, "tifs_per_run", 35))  # 0 or <0 means "no cap"
    remaining = cap_per_run if cap_per_run > 0 else float("inf")
    logging.info(f"[capacity : {remaining}]")

    tile_tasks = []

    for tile_id in tiles:
        logging.info(f"[chcking {tile_id}] for sub tiles")
        if remaining <= 0:
            break  # stop scanning once we've queued cap_per_run TIFFs

        # gather & filter TIFFs for this tile
        tif_paths = glob.glob(os.path.join(args.master_dir, tile_id, "*.tif"))  # no need to sort for counts
        tile_out_dir = os.path.join(args.output_tiles_dir, str(tile_id))

        # Only rank 1 prints verbose filter lines to avoid spam
        verbose_here = (rank == 1)

        tif_paths, _skipped = filter_done_tiffs(
            tif_paths, tile_out_dir, verify=True, verbose=verbose_here
        )
        if not tif_paths:
            continue

        take = min(remaining, len(tif_paths))
        for tif in tif_paths[:take]:
            # add any per-task flags you pass to the worker (e.g., defer flag) here
            logging.info(f"[tile_id {tile_id}] with {tif} sub tile added remaining{remaining}")
            tile_tasks.append((tile_id, [tif], args.output_tiles_dir, smart_verbose))
        remaining -= take

    if not tile_tasks:
        logging.error(f"[rank {rank}] No pending TIFFs after filtering; check paths/partition.")
        return

    cap_label = "∞" if cap_per_run <= 0 else str(cap_per_run)
    logging.info(f"[rank {rank}] queued {len(tile_tasks)} TIF(s) (cap={cap_label})")

    total_mem_gb = psutil.virtual_memory().total / 1e9
    cpu_cnt      = max(1, (os.cpu_count() or 1) - 1)
    mem_based    = max(1, int(total_mem_gb // args.max_gb_per_worker))
    auto_workers = min(cpu_cnt, mem_based, HARD_MAX_WORKERS)

    if smart_verbose:
        n_workers = 1
        logging.debug(f"[smart-verbose] forcing Workers=1 for ordered logs")
    else:
        n_workers = args.workers or auto_workers

    logging.info(f"Tiles: {len(tile_tasks)} | Workers: {n_workers} | RAM {total_mem_gb:.1f} GB cap/worker {args.max_gb_per_worker} GB")

    start = time.time()
    per_tile_rows = []

    if n_workers == 1:
        for task in tile_tasks:
            res = process_one_tile(task)
            if res:
                per_tile_rows.append(res)
    else:
        with mp.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(process_one_tile, tile_tasks, chunksize=1):
                if res:
                    per_tile_rows.append(res)

    build_master_gpkg(args.mosaic_gpkg, per_tile_rows, args.imagery_dir)

    elapsed = time.time() - start
    logging.info(f"Done in {elapsed/60:.1f} min | {elapsed/len(tile_tasks):.1f} s/tile")

if __name__ == "__main__":
    try:
        os.environ.setdefault("OGR_SQLITE_SYNCHRONOUS", "OFF")
        os.environ.setdefault("OGR_SQLITE_CACHE", "200000")
        main()
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)

