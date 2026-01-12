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
import argparse
import multiprocessing as mp
from contextlib import contextmanager
from math import sin, cos, radians, atan2, degrees

import shapely.geometry as sgeom

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


from gt_gpkg_common import StageTimer, _expected_gpkg_name_for_tif, _gpkg_delete_layer, _has_table, _is_valid_gpkg, _records_to_gdf, _table_has_fields, axial_mean_deg, build_master_gpkg, combine_axial_deg, filter_done_tiffs, get_sub_tile_id, has_layer_sqlite, insert_xml_metadata, is_gpkg_complete_by_tail, is_valid_geopackage, list_gpkg_layers_sqlite, logp, pix_to_xy, sanity_check_paths, sqlite_fast_writes, write_table

# ------------------------------ config ------------------------------
SUBTILE_AREA_M2 = 375000000.0 # 400000000. USED if area cannot be found from the tile_id_tiles.shp
DEFAULT_PIXEL_M  = 0.5
HARD_MAX_WORKERS = 32
ID_VERSION = "v1.0-deterministic-rcsorted-neighsorted-compsorted"
COMPS_HEARTBEAT = 20000  # progress log every N components

# ------------------------------ small logging helpers ------------------------------



# ------------------------------ math helpers ------------------------------




def fast_pix_to_xy_affine(trf, rr, cc):
    """Vectorized pixel-center to map coords using the affine transform."""
    rr = np.asarray(rr, dtype=float)
    cc = np.asarray(cc, dtype=float)
    x = trf.c + (cc + 0.5) * trf.a + (rr + 0.5) * trf.b
    y = trf.f + (cc + 0.5) * trf.d + (rr + 0.5) * trf.e
    return x, y

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

from osgeo import ogr



#------------------------------ per-tile processing updated..............
################# AGGREGATION #################################################
import os, glob, time, logging, sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd

# reuse your helpers:
# - StageTimer, sqlite_fast_writes, write_table, insert_xml_metadata, ID_VERSION

def _axial_mean_deg_weighted(angles_deg, weights):
    if len(angles_deg) == 0:
        return -1.0
    a = (np.asarray(angles_deg, dtype=float) % 180.0)
    w = np.asarray(weights, dtype=float)
    if len(a) != len(w) or w.sum() <= 0:
        return float(a[0])
    theta = np.deg2rad(2.0 * a)
    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    if C == 0 and S == 0:
        return -1.0
    mean_doubled = np.arctan2(S, C)
    return float((np.rad2deg(mean_doubled) / 2.0) % 180.0)


def _iter_perfile(tile_dir, tile_id):
    """
    Scan ONLY: <tile_dir>/*.gpkg Skip the aggregate <tile_id>.gpkg.
    """
    tile_dir = os.path.join(tile_dir,str(tile_id))
    print(f"tile dir: {tile_dir}")
    if not os.path.isdir(tile_dir):
        logging.error(f"[{tile_id}] tile directory not found: {tile_dir}")
        return
    agg_name = f"{tile_id}_TCN_summary.gpkg"
    for p in glob.glob(os.path.join(tile_dir, "*_TCN.gpkg")):
        print(f"p: {p}")
        if os.path.basename(p) != agg_name:
            yield p



################# </AGGREGATION>#################################################
# ------------------------------ master build ------------------------------


# ------------------------------ startup sanity ------------------------------


#-------------CODE ADDED for sub_tile aggregations------------------------------------------------------
# Add at top of module:
import re
from shapely.ops import unary_union

def _subtile_id_from_basename(gpkg_path: str) -> str:
    """Derive a subtile id from per-file GPKG basename by stripping trailing '_TCN[...].gpkg'."""
    stem = os.path.splitext(os.path.basename(gpkg_path))[0]
    # Strip suffix patterns like: _TCN.gpkg, _TCN_*.gpkg (case-insensitive)
    subtile = re.sub(r'(?i)_TCN(?:_.*)?$', '', stem)
    return subtile or stem


    # --- helpers (near your other helpers) ---

from pathlib import Path

def _subtile_id_from_basename(gpkg_path: str) -> str:
    stem = Path(gpkg_path).stem
    # strip trailing _TCN or _TCN_* suffix
    return re.sub(r'(?i)_TCN(?:_.*)?$', '', stem)

def _make_tilename(sub_tile_id: str) -> list[str]:
    """Return possible TILENAME keys (case/extension tolerant)."""
    base = f"{sub_tile_id}"
    return [f"{base}.tif", f"{base}.TIF", base]  # some tilesets omit extension in attrs

def _norm_key(s: str) -> str:
    return s.strip().lower()

# --- worker now receives imagery_dir too ---
def _agg_worker(task):
    """task = (tile_id, in_dir, out_dir, imagery_dir, smart_verbose)"""
    tile_id, tile_dir, output_tiles_dir, imagery_dir, smart_verbose = task
    return aggregate_tile_from_perfile_gpkgs(
        tile_id, tile_dir, output_tiles_dir, imagery_dir, verbose=smart_verbose
    )

def _ensure_crs(gdf, target_crs, label=""):
    # Nothing to do if no target CRS or empty
    if target_crs is None or gdf.empty:
        return gdf

    if gdf.crs is None:
        logging.warning("CRS missing for %s; assuming %s", label, target_crs)
        return gdf.set_crs(target_crs, allow_override=True)

    # Normalize both to WKT for comparison
    gdf_wkt = gdf.crs.to_wkt() if hasattr(gdf.crs, "to_wkt") else str(gdf.crs)
    target_wkt = target_crs.to_wkt() if hasattr(target_crs, "to_wkt") else str(target_crs)

    if gdf_wkt != target_wkt:
        logging.warning(
            "Reprojecting %s from %s to %s", label, gdf.crs, target_crs
        )
        return gdf.to_crs(target_crs)
    return gdf

def aggregate_tile_from_perfile_gpkgs(tile_id, tile_dir, output_tiles_dir, imagery_dir, verbose=False):
    """Build <tile_id>_TCN_aggregated.gpkg from per-file GPKGs; also write SubTileStats using
       polygons from ArcticMosaic_<tile_id>_tiles.shp and exposure from InputMosaicCutlinesVector."""
    t0 = time.perf_counter()

    if not os.path.isdir(tile_dir):
        logging.error(f"[{tile_id}] missing tile dir: {tile_dir}")
        return None

    # ---- load tile polygons once (by TILENAME) ----
    tiles_shp = os.path.join(imagery_dir, tile_id, f"ArcticMosaic_{tile_id}_tiles.shp")
    tile_geom_by_name = {}
    tiles_gdf = None
    tiles_crs = None
    if os.path.exists(tiles_shp):
        try:
            tiles_gdf = gpd.read_file(tiles_shp)
            if tiles_gdf.crs is None:
                logging.warning(f"[{tile_id}] tiles.shp has no CRS; will set later from data CRS if available")
                tiles_gdf=tiles_gdf.set_crs(3338)
            elif tiles_gdf.crs != 3338:
                tiles_gdf=tiles_gdf.to_crs(3338)

            tiles_crs = tiles_gdf.crs
            if "TILENAME" not in tiles_gdf.columns:
                logging.warning(f"[{tile_id}] TILENAME missing in tiles.shp; cannot map sub-tile polygons")
            else:
                # keep only polygonal features
                poly_mask = tiles_gdf.geometry.apply(
                    lambda g: getattr(g, "geom_type", "") in ("Polygon", "MultiPolygon"))
                tiles_gdf = tiles_gdf.loc[poly_mask].copy()
                tile_geom_by_name = {
                    _norm_key(str(v)): geom
                    for v, geom in zip(tiles_gdf["TILENAME"], tiles_gdf.geometry)
                    if geom is not None
                }
        except Exception as e:
            logging.warning(f"[{tile_id}] failed to read tiles.shp: {e}")
    else:
        logging.warning(f"[{tile_id}] tiles.shp not found at {tiles_shp}")

    edges_gdfs, nodes_gdfs, comps_gdfs = [], [], []
    sub_rows = []
    crs_wkt = None

    with StageTimer(verbose, f"[{tile_id}]", "READ per-file GPKGs"):
        gpkg_files = list(_iter_perfile(tile_dir, tile_id))
        if not gpkg_files:
            logging.error(f"[{tile_id}] no *.gpkg files found in {tile_dir}")
            return None

        for gpkg in gpkg_files:
            bn = os.path.basename(gpkg).lower()
            # skip tile-level aggregates / non-subtile files
            if bn.endswith("_tcn_aggregated.gpkg") or "_tcn_agg" in bn or bn == f"{tile_id.lower()}.gpkg":
                continue

            layers = _listlayers_safe(gpkg)

            e_gdf = n_gdf = c_gdf = None

            if "GraphTheoraticEdges" in layers:
                try:
                    e_gdf = gpd.read_file(gpkg, layer="GraphTheoraticEdges")

                    # Set canonical CRS once
                    if crs_wkt is None and e_gdf.crs:
                        crs_wkt = e_gdf.crs.to_wkt()

                    # Normalize to canonical CRS
                    e_gdf = _ensure_crs(
                        e_gdf, crs_wkt, f"edges {os.path.basename(gpkg)}"
                    )

                    edges_gdfs.append(e_gdf)
                except Exception as e:
                    logging.warning(f"[{os.path.basename(gpkg)}] edges read fail: {e}")

            if "GraphTheoraticNodes" in layers:
                try:
                    n_gdf = gpd.read_file(gpkg, layer="GraphTheoraticNodes")

                    if crs_wkt is None and n_gdf.crs:
                        crs_wkt = n_gdf.crs.to_wkt()

                    n_gdf = _ensure_crs(
                        n_gdf, crs_wkt, f"nodes {os.path.basename(gpkg)}"
                    )

                    nodes_gdfs.append(n_gdf)
                except Exception as e:
                    logging.warning(f"[{os.path.basename(gpkg)}] nodes read fail: {e}")

            if "GraphTheoraticComponents" in layers:
                try:
                    c_gdf = gpd.read_file(gpkg, layer="GraphTheoraticComponents")

                    if crs_wkt is None and c_gdf.crs:
                        crs_wkt = c_gdf.crs.to_wkt()

                    c_gdf = _ensure_crs(
                        c_gdf, crs_wkt, f"comps {os.path.basename(gpkg)}"
                    )

                    comps_gdfs.append(c_gdf)
                except Exception as e:
                    logging.warning(f"[{tile_id}] comps read fail: {e}")

            # ---- per-subtile metrics (from this per-file GPKG) ----
            # Components-derived totals
            if c_gdf is not None and not c_gdf.empty:
                num_nodes_arr = c_gdf.get("num_nodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
                avg_degree_arr = c_gdf.get("avg_degree", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
                total_len_arr = c_gdf.get("total_length_m", pd.Series(dtype=float)).fillna(0).astype(
                    float).to_numpy()
                endnodes_arr = c_gdf.get("num_endnodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
                junctions_arr = c_gdf.get("num_junctions", pd.Series(dtype=float)).fillna(0).astype(
                    float).to_numpy()

                sub_num_graph_nodes_sum = int(np.round(num_nodes_arr.sum()))
                sub_num_graph_edges_sum = int(np.round(0.5 * (avg_degree_arr * num_nodes_arr).sum()))
                sub_end_nodes_count_sum = int(np.round(endnodes_arr.sum()))
                sub_junction_nodes_count_sum = int(np.round(junctions_arr.sum()))

                sub_total_tcn_length_m_sum = float(total_len_arr.sum())
                sub_num_components = int(len(c_gdf))
                sub_average_component_size_m = float(np.mean(total_len_arr)) if sub_num_components > 0 else 0.0
                sub_largest_component_size_m = float(np.max(total_len_arr)) if sub_num_components > 0 else 0.0
            else:
                sub_num_graph_nodes_sum = sub_num_graph_edges_sum = sub_end_nodes_count_sum = sub_junction_nodes_count_sum = 0
                sub_total_tcn_length_m_sum = 0.0
                sub_num_components = 0
                sub_average_component_size_m = sub_largest_component_size_m = 0.0

            # Edges-derived metrics
            if e_gdf is not None and not e_gdf.empty:
                edge_lengths = e_gdf.get("length_m", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
                edge_orients = e_gdf.get("orientation_deg_axial", pd.Series(dtype=float)).fillna(0).astype(
                    float).to_numpy()
                if len(edge_lengths) > 0:
                    p95 = np.percentile(edge_lengths, 95)
                    trimmed = edge_lengths[edge_lengths <= p95]
                    sub_mean_edge_length_m = float(trimmed.mean()) if len(trimmed) else 0.0
                else:
                    sub_mean_edge_length_m = 0.0
                sub_mean_edge_orientation_deg_stepweighted = (
                    _axial_mean_deg_weighted(edge_orients, edge_lengths) if edge_lengths.sum() > 0 else -1.0
                )
                sub_num_edges_branchlevel = int(len(e_gdf))
            else:
                sub_mean_edge_length_m = 0.0
                sub_mean_edge_orientation_deg_stepweighted = -1.0
                sub_num_edges_branchlevel = 0

            sub_average_node_degree = (
                        2.0 * sub_num_graph_edges_sum / sub_num_graph_nodes_sum) if sub_num_graph_nodes_sum > 0 else 0.0

            # ---- geometry from tiles.shp by TILENAME ----
            subtile_id = _subtile_id_from_basename(gpkg)
            subtile_geom = None
            if tile_geom_by_name:
                keys = _make_tilename(subtile_id)
                for k in keys:
                    geom = tile_geom_by_name.get(_norm_key(k))
                    if geom is not None:
                        subtile_geom = geom
                        break
            if subtile_geom is None:
                logging.warning(f"[{tile_id}/{subtile_id}/{_make_tilename(subtile_id)}]"
                                " no polygon in tiles.shp; using  falling back to default SUBTILE_AREA_M2")

            # ---- exposure from cutlines (area-weighted norm_input_exp) ----
            subtile_area_m2 = float(subtile_geom.area) if subtile_geom is not None else SUBTILE_AREA_M2
            # ---- exposure from cutlines (area-weighted norm_input_exp) ----
            norm_input_exp_coverage_weighted = None
            input_coverage = None
            if "InputMosaicCutlinesVector" in layers:
                try:
                    cut = gpd.read_file(gpkg, layer="InputMosaicCutlinesVector")
                    # EPSG:3338 (equal-area) is expected for areas; if CRS is missing, set from crs_wkt if available
                    if cut.crs is None and crs_wkt:
                        cut = cut.set_crs(crs_wkt)
                    if cut.crs != tiles_crs:
                        cut = cut.to_crs(tiles_crs)

                    # polygonal only
                    poly_mask = cut.geometry.apply(
                        lambda g: getattr(g, "geom_type", "") in ("Polygon", "MultiPolygon")
                    )
                    cut = cut.loc[poly_mask].copy()
                    if not cut.empty:
                        areas = pd.to_numeric(cut.geometry.area, errors="coerce").fillna(0.0).astype(float)
                        total_area = float(areas.sum())

                        if "norm_input_exp" in cut.columns and total_area > 0:
                            exp = pd.to_numeric(cut["norm_input_exp"], errors="coerce").fillna(0.0).astype(float)
                            val = float((exp * areas).sum() / total_area)
                            norm_input_exp_coverage_weighted = max(0.0, min(1.0, val))

                        # CHANGED: use subtile_area_m2 instead of SUBTILE_AREA_M2
                        if total_area > 0 and subtile_area_m2 and subtile_area_m2 > 0:
                            cov = total_area / float(subtile_area_m2)
                            cov = max(0.0, min(1.0, cov))

                            # optional: snap near-1 to exactly 1.0
                            EPS = 1e-4
                            if abs(cov - 1.0) < EPS:
                                cov = 1.0

                            input_coverage = cov
                            logging.info(
                                f"input coverage: {input_coverage} "
                                f"total CUTLINEs area: {total_area} "
                                f"subtile area: {subtile_area_m2}"
                            )
                except Exception as e:
                    logging.warning(f"[{os.path.basename(gpkg)}] cutlines exposure failed: {e}")

                if input_coverage is None:
                    input_coverage = 0.0
                if norm_input_exp_coverage_weighted is None:
                    norm_input_exp_coverage_weighted = 0.0
            # If still no geometry, last resort: bbox from edges (not ideal)
            if subtile_geom is None and e_gdf is not None and not e_gdf.empty:
                try:
                    minx, miny, maxx, maxy = e_gdf.total_bounds
                    subtile_geom = shapely.geometry.box(minx, miny, maxx, maxy)
                except Exception:
                    pass

            sub_rows.append({
                "tile_id": tile_id,
                "sub_tile_id": subtile_id,
                "id_version": ID_VERSION,
                "raster_crs_wkt": crs_wkt or "",
                "num_components": int(sub_num_components),
                "total_tcn_length_m": float(sub_total_tcn_length_m_sum),
                "largest_component_size_m": float(sub_largest_component_size_m),
                "average_component_size_m": float(sub_average_component_size_m),
                "num_graph_nodes": int(sub_num_graph_nodes_sum),
                "num_graph_edges": int(sub_num_graph_edges_sum),
                "end_nodes_count": int(sub_end_nodes_count_sum),
                "junction_nodes_count": int(sub_junction_nodes_count_sum),
                "average_node_degree": float(sub_average_node_degree),
                "mean_edge_length_m": float(sub_mean_edge_length_m),
                "mean_edge_orientation_deg_stepweighted": float(sub_mean_edge_orientation_deg_stepweighted),
                "num_edges": int(sub_num_edges_branchlevel),
                "norm_input_exp_coverage_weighted": None if norm_input_exp_coverage_weighted is None else float(
                    norm_input_exp_coverage_weighted),
                "input_coverage": None if input_coverage is None else float(
                    input_coverage),
                "geometry": subtile_geom,
            })

    if not edges_gdfs and not nodes_gdfs and not comps_gdfs:
        logging.error(f"[{tile_id}] no per-file layers to aggregate (dir={tile_dir})")
        return None

    # ---- aggregate (tile-level) unchanged ----

    edges_all = gpd.GeoDataFrame(pd.concat(edges_gdfs, ignore_index=True)) if edges_gdfs else gpd.GeoDataFrame(geometry=[])
    nodes_all = gpd.GeoDataFrame(pd.concat(nodes_gdfs, ignore_index=True)) if nodes_gdfs else gpd.GeoDataFrame(geometry=[])
    comps_all = gpd.GeoDataFrame(pd.concat(comps_gdfs, ignore_index=True)) if comps_gdfs else gpd.GeoDataFrame(geometry=[])

    if not comps_all.empty:
        num_nodes_arr = comps_all.get("num_nodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        avg_degree_arr = comps_all.get("avg_degree", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        total_len_arr = comps_all.get("total_length_m", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        endnodes_arr = comps_all.get("num_endnodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        junctions_arr = comps_all.get("num_junctions", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()

        num_graph_nodes_sum = int(np.round(num_nodes_arr.sum()))
        num_graph_edges_sum = int(np.round(0.5 * (avg_degree_arr * num_nodes_arr).sum()))
        end_nodes_count_sum = int(np.round(endnodes_arr.sum()))
        junction_nodes_count_sum = int(np.round(junctions_arr.sum()))

        total_tcn_length_m_sum = float(total_len_arr.sum())
        num_components = int(len(comps_all))
        average_component_size_m = float(np.mean(total_len_arr)) if num_components > 0 else 0.0
        largest_component_size_m = float(np.max(total_len_arr)) if num_components > 0 else 0.0
    else:
        num_graph_nodes_sum = num_graph_edges_sum = end_nodes_count_sum = junction_nodes_count_sum = 0
        total_tcn_length_m_sum = 0.0
        num_components = 0
        average_component_size_m = largest_component_size_m = 0.0

    # ---- edge-derived stats (tile scope) ----
    if not edges_all.empty:
        edge_lengths = edges_all.get("length_m", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        edge_orients = edges_all.get("orientation_deg_axial", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        if len(edge_lengths) > 0:
            p95 = np.percentile(edge_lengths, 95)
            trimmed = edge_lengths[edge_lengths <= p95]
            mean_edge_length_m = float(trimmed.mean()) if len(trimmed) else 0.0
        else:
            mean_edge_length_m = 0.0
        mean_edge_orientation_deg_stepweighted = _axial_mean_deg_weighted(edge_orients,
                                                                          edge_lengths) if edge_lengths.sum() > 0 else -1.0
        num_edges_branchlevel = int(len(edges_all))
    else:
        mean_edge_length_m = 0.0
        mean_edge_orientation_deg_stepweighted = -1.0
        num_edges_branchlevel = 0

    average_node_degree = (2.0 * num_graph_edges_sum / num_graph_nodes_sum) if num_graph_nodes_sum > 0 else 0.0

    # ---- write outputs (existing layers + new SubTileStats) ----
    gpkg_path = os.path.join(output_tiles_dir, f"{tile_id}", f"{tile_id}_TCN_summary.gpkg")
    os.makedirs(os.path.dirname(gpkg_path), exist_ok=True)
    logging.info(f"################################################################GPKG path {gpkg_path}")
    # Remove if existing
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
        logging.info(f"#################################################for new summary removing {gpkg_path}")

    def _write_layer(gdf, layer):
        if gdf is None or gdf.empty:
            return
        if crs_wkt and (gdf.crs is None):
            gdf = gdf.set_crs(crs_wkt)
        mode = "a" if os.path.exists(gpkg_path) else "w"
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG", mode=mode)
        logging.info(f"[{tile_id}] wrote {len(gdf):,} to '{layer}'")

    if crs_wkt is None and e_gdf.crs:
        crs_wkt = e_gdf.crs.to_wkt()

    with StageTimer(verbose, f"[{tile_id}]", "WRITE aggregate layers"):
        _write_layer(edges_all, "GraphTheoraticEdges")
        _write_layer(nodes_all, "GraphTheoraticNodes")
        _write_layer(comps_all, "GraphTheoraticComponents")

        # New: per-subtile polygon layer from tiles.shp + cutline exposure
        if sub_rows:
            st_df = pd.DataFrame(sub_rows)
            # cast numerics to avoid pyogrio warnings
            int_cols = ["num_components", "num_graph_nodes", "num_graph_edges", "end_nodes_count",
                        "junction_nodes_count", "num_edges"]
            float_cols = [
                "total_tcn_length_m", "largest_component_size_m", "average_component_size_m",
                "average_node_degree", "mean_edge_length_m", "mean_edge_orientation_deg_stepweighted",
                "norm_input_exp_coverage_weighted", "input_coverage",
            ]
            for c in int_cols:
                if c in st_df.columns:
                    st_df[c] = pd.to_numeric(st_df[c], errors="coerce").fillna(0).astype(np.int64)
            for c in float_cols:
                if c in st_df.columns:
                    st_df[c] = pd.to_numeric(st_df[c], errors="coerce").astype(float)

            # New: write SubTileStats with correct CRS handling (tiles.shp -> target data CRS)
            if "geometry" in st_df.columns:
                st_gdf = gpd.GeoDataFrame(st_df, geometry="geometry")

                # 1) Label geometries with the source CRS from tiles.shp (e.g., EPSG:3413)
                if tiles_crs:
                    st_gdf = st_gdf.set_crs(tiles_crs, allow_override=True)

                # 2) Reproject to target data CRS (from per-file layers; typically EPSG:3338)
                target_crs_wkt = crs_wkt or (tiles_crs.to_wkt() if tiles_crs else None)
                if target_crs_wkt and (not tiles_crs or tiles_crs.to_wkt() != target_crs_wkt):
                    st_gdf = st_gdf.to_crs(target_crs_wkt)

                _write_layer(st_gdf, "SubTileStats")

    # -------- write non-spatial global_stats + XML (unchanged) --------
    tile_row = {
        "id": 1,
        "tile_id": tile_id,
        "id_version": ID_VERSION,
        "raster_crs_wkt": crs_wkt or "",
        "num_components": int(num_components),
        "total_tcn_length_m": float(total_tcn_length_m_sum),
        "largest_component_size_m": float(largest_component_size_m),
        "average_component_size_m": float(average_component_size_m),
        "num_graph_nodes": int(num_graph_nodes_sum),
        "num_graph_edges": int(num_graph_edges_sum),
        "end_nodes_count": int(end_nodes_count_sum),
        "junction_nodes_count": int(junction_nodes_count_sum),
        "average_node_degree": float(average_node_degree),
        "mean_edge_length_m": float(mean_edge_length_m),
        "mean_edge_orientation_deg_stepweighted": float(mean_edge_orientation_deg_stepweighted),
        "num_edges": int(num_edges_branchlevel),
    }

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

    dt = time.perf_counter() - t0
    logging.debug(f"[{tile_id}] aggregate-from-perfile done in {dt:.2f}s")
    out_row = dict(tile_row); out_row["id"] = None
    return out_row

#-------------</CODE ADDED for sub_tile aggregations>------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master_dir", required=True, help="Folder of tiles; each subfolder is a tile_id with *.tif masks")
    p.add_argument("--output_tiles_dir", required=True, help="Output dir containing per-file GPKGs under <tile_id>/")
    p.add_argument("--mosaic_gpkg", required=True, help="Output path for master GeoPackage (whole_mosaic.gpkg)")
    p.add_argument("--imagery_dir", required=True, help="Folder with per-tile shapefiles ArcticMosaic_<tile_id>_tiles.shp")
    p.add_argument("--workers", type=int, default=0, help="Processes; 0=auto (all-1, mem-capped)")
    p.add_argument("--max_gb_per_worker", type=float, default=64, help="Estimated GB RAM per worker")

    # Selecting the tile sequence in rank order
    p.add_argument("--tile_rank_st", type=int, default=None, help="1-based start index of tiles to process (optional)")
    p.add_argument("--tile_rank_end", type=int, default=None, help="1-based end index of tiles to process (optional)")

    p.add_argument("--defer-tile-agg", action="store_true",
                   help="Skip per-tile stats and GPKG writes in pass-1; only write per-file GPKGs")
    p.add_argument("--tifs_per_run", type=int, default=35, help="Max number of items to process this run (tiles in pass-2)")

    # testing controls
    p.add_argument("--one_tile", type=str, default=None, help="Only process this tile_id (subfolder name)")
    p.add_argument("--one_tif", action="store_true", help="Within each selected tile, only process the first .tif")
    args = p.parse_args()

    # make available for any code that still checks this flag
    global DEFER_TILE_AGG
    DEFER_TILE_AGG = args.defer_tile_agg

    # logging
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

    tiles = all_tiles[:]

    # ---------- rank/world partition (round-robin) ----------
    rank = int(getattr(args, "rank", 1))
    world = int(getattr(args, "world_size", 1))
    world = max(world, 1)
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

    # If user specified a single tile, honor it (kept as before)
    if args.one_tile:
        tiles = [t for t in tiles if t == args.one_tile]

    # ---------- determine cap BEFORE building tasks ----------
    cap_per_run = int(getattr(args, "tifs_per_run", 35))  # here: cap = max tiles to aggregate
    remaining = cap_per_run if cap_per_run > 0 else float("inf")
    logging.info(f"[capacity: {remaining}]")

    tile_tasks = []
    for tile_id in tiles:
        logging.info(f"[checking {tile_id}] for per-file GPKGs")
        if remaining <= 0:
            break
        # Only rank 1 prints verbose filter lines to avoid spam (kept; currently unused here)
        verbose_here = (rank == 1)
        # Each task is ONE tile to aggregate from its per-file GPKGs
        tile_tasks.append((tile_id, args.output_tiles_dir, args.output_tiles_dir,args.imagery_dir, smart_verbose))
        remaining -= 1

    if not tile_tasks:
        logging.error(f"[rank {rank}] No tiles queued; check paths/partition.")
        return

    cap_label = "∞" if cap_per_run <= 0 else str(cap_per_run)
    logging.info(f"[rank {rank}] queued {len(tile_tasks)} tile(s) (cap={cap_label})")

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
            res = _agg_worker(task)  # unpack + call aggregator
            if res:
                per_tile_rows.append(res)
    else:
        with mp.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_agg_worker, tile_tasks, chunksize=1):
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

"""
python gt_gpkg_sub_tile_sum.py \
            --master_dir ../../masked \
            --output_tiles_dir /scratch2/projects/PDG_shared/TCN_gpkgs \
            --mosaic_gpkg /scratch2/projects/PDG_shared/TCN_gpkgs/all_mosaic.gpkg \
            --imagery_dir /scratch2/projects/PDG_shared/AlaskaTundraMosaic/imagery \
            --tifs_per_run 1 \
            --tile_rank_st 101 \
            --tile_rank_end 101 \
            --workers 1
"""