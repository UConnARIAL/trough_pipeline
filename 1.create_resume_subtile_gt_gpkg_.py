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

from rasterio.transform import xy as rio_xy
from skimage.morphology import skeletonize
from skan.csr import pixel_graph
import networkx as nx
import psutil

import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

# ---- SciPy optional (fast path) ----
try:
    from scipy.sparse.csgraph import connected_components as csgraph_connected_components
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


from gt_gpkg_common import StageTimer, _expected_gpkg_name_for_tif, _gpkg_delete_layer, _has_table, _is_valid_gpkg, _records_to_gdf, _table_has_fields, axial_mean_deg, build_master_gpkg, combine_axial_deg, filter_done_tiffs, get_sub_tile_id, insert_xml_metadata, is_gpkg_complete_by_tail, is_valid_geopackage, logp, pix_to_xy, sanity_check_paths, sqlite_fast_writes, write_table

# ------------------------------ config ------------------------------

DEFAULT_PIXEL_M  = 0.5
HARD_MAX_WORKERS = 32
ID_VERSION = "v1.0-deterministic-rcsorted-neighsorted-compsorted"
COMPS_HEARTBEAT = 20000  # progress log every N components
TARGET_CRS = CRS.from_epsg(3338)  # Alaska Albers (equal-area)
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
    ################## REMOVE HARD CODING #########################################################################33
    org_mask_path = AL.tiff_to_gpkg_path(tif_path,old_gpkg_root="/scratch2/projects/PDG_shared/AK_TCN_Masks_only/",
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
        # ------------------------------ CRS checking and reporjections added later
        try:
            with (StageTimer(verbose, pfx, "OPEN")):
                with rasterio.open(tif_path) as src:
                    # Decide whether to read directly or via a warped (reprojected) virtual raster
                    src_crs = src.crs
                    if src_crs is None:
                        # safest default: stop early so you don't silently misplace pixels
                        raise ValueError(f"No CRS found in raster: {tif_path}")

                    if src_crs != TARGET_CRS:
                        # Masks/labels => nearest neighbor to avoid class mixing
                        with WarpedVRT(src, crs=TARGET_CRS, resampling=Resampling.nearest) as vrt:
                            img = vrt.read(1).astype(np.uint8)
                            trf = vrt.transform
                            crs_wkt = crs_wkt or vrt.crs.to_wkt()
                    else:
                        img = src.read(1).astype(np.uint8)
                        trf = src.transform
                        crs_wkt = crs_wkt or src_crs.to_wkt()
                    # Pixel size in meters in the *target CRS grid*
                    px = abs(trf.a)
                    py = abs(trf.e)
                    step_len_m = 0.5 * (px + py) if (px > 0 and py > 0 and abs(px - py) / max(px, py) < 1e-6) else DEFAULT_PIXEL_M
                    #step_len_m = 0.5 * (px+py) if (px > 0 and py > 0 and abs(px - py) < 1e-6) else DEFAULT_PIXEL_M
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


# ------------------------------ startup sanity ------------------------------


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

    #build_master_gpkg(args.mosaic_gpkg, per_tile_rows, args.imagery_dir)

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

