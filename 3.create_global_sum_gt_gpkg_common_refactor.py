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

import re
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
################# AGGREGATION #################################################

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
    agg_name = f"{tile_id}.gpkg"
    for p in glob.glob(os.path.join(tile_dir, "*.gpkg")):
        print(f"p: {p}")
        if os.path.basename(p) != agg_name:
            yield p


def aggregate_tile_from_perfile_gpkgs(tile_id, tile_dir, output_tiles_dir, verbose=False):
    """Build <tile_id>.gpkg from per-file GPKGs under <tile_dir>; return global_stats dict."""
    t0 = time.perf_counter()

    if not os.path.isdir(tile_dir):
        logging.error(f"[{tile_id}] missing tile dir: {tile_dir}")
        return None

    edges_gdfs, nodes_gdfs, comps_gdfs = [], [], []
    crs_wkt = None

    with StageTimer(verbose, f"[{tile_id}]", "READ per-file GPKGs"):
        gpkg_files = list(_iter_perfile(tile_dir, tile_id))
        logging.debug(f"[{tile_id}] scanning {tile_dir} | {len(gpkg_files)} *.gpkg files")
        if not gpkg_files:
            logging.error(f"[{tile_id}] no *.gpkg files found in {tile_dir}")
            return None

        for gpkg in gpkg_files[:3]:
            logging.info(f"[{tile_id}] probe {os.path.basename(gpkg)} -> {list_gpkg_layers_sqlite(gpkg)}")

        for gpkg in gpkg_files:
            print(f"reading Layers")
            layers = _listlayers_safe(gpkg)  # uses fiona.listlayers under the hood
            logging.debug(f"[{tile_id}] {os.path.basename(gpkg)} layers={sorted(layers)}")

            if "GraphTheoraticEdges" in layers:
                try:
                    #print(f"reading Edges")
                    logging.info(f"[{os.path.basename(gpkg)}] Reading Edges..")
                    g = gpd.read_file(gpkg, layer="GraphTheoraticEdges")
                    edges_gdfs.append(g)
                    if crs_wkt is None and g.crs: crs_wkt = g.crs.to_wkt()
                except Exception as e:
                    logging.warning(f"[{os.path.basename(gpkg)}] edges read fail {gpkg}: {e}")

            if "GraphTheoraticNodes" in layers:
                try:
                    logging.info(f"[{os.path.basename(gpkg)}] Reading Nodes....")
                    g = gpd.read_file(gpkg, layer="GraphTheoraticNodes")
                    nodes_gdfs.append(g)
                    if crs_wkt is None and g.crs: crs_wkt = g.crs.to_wkt()
                except Exception as e:
                    logging.warning(f"[{os.path.basename(gpkg)}] nodes read fail {gpkg}: {e}")

            if "GraphTheoraticComponents" in layers:
                try:
                    #print(f"reading Comps")
                    logging.info(f"[{os.path.basename(gpkg)}] Reading Comps........")
                    g = gpd.read_file(gpkg, layer="GraphTheoraticComponents")
                    comps_gdfs.append(g)
                    if crs_wkt is None and g.crs: crs_wkt = g.crs.to_wkt()
                except Exception as e:
                    logging.warning(f"[{tile_id}] comps read fail {gpkg}: {e}")

    if not edges_gdfs and not nodes_gdfs and not comps_gdfs:
        logging.error(f"[{tile_id}] no per-file layers to aggregate (dir={tile_dir})")
        return None
    print(f"reading DONE")
    # ... (aggregation + writing unchanged) ...
    # Write aggregate <parent>/<tile_id>.gpkg:
    parent_dir = os.path.dirname(os.path.normpath(tile_dir))
    gpkg_path  = os.path.join(parent_dir, f"{tile_id}.gpkg")
    os.makedirs(parent_dir, exist_ok=True)

    edges_all = gpd.GeoDataFrame(pd.concat(edges_gdfs, ignore_index=True)) if edges_gdfs else gpd.GeoDataFrame(geometry=[])
    nodes_all = gpd.GeoDataFrame(pd.concat(nodes_gdfs, ignore_index=True)) if nodes_gdfs else gpd.GeoDataFrame(geometry=[])
    comps_all = gpd.GeoDataFrame(pd.concat(comps_gdfs, ignore_index=True)) if comps_gdfs else gpd.GeoDataFrame(geometry=[])

    print(f"GeoDF creation done")
    # -------- parity calculations (mirror first pass) --------
    # totals from components
    if not comps_all.empty:
        num_nodes_arr   = comps_all.get("num_nodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        avg_degree_arr  = comps_all.get("avg_degree", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        total_len_arr   = comps_all.get("total_length_m", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        endnodes_arr    = comps_all.get("num_endnodes", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        junctions_arr   = comps_all.get("num_junctions", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()

        num_graph_nodes_sum = int(np.round(num_nodes_arr.sum()))
        num_graph_edges_sum = int(np.round(0.5 * (avg_degree_arr * num_nodes_arr).sum()))
        end_nodes_count_sum = int(np.round(endnodes_arr.sum()))
        junction_nodes_count_sum = int(np.round(junctions_arr.sum()))

        total_tcn_length_m_sum   = float(total_len_arr.sum())
        component_lengths_all    = total_len_arr.tolist()
        num_components           = int(len(comps_all))
        average_component_size_m = float(np.mean(total_len_arr)) if num_components > 0 else 0.0
        largest_component_size_m = float(np.max(total_len_arr)) if num_components > 0 else 0.0
    else:
        num_graph_nodes_sum = num_graph_edges_sum = end_nodes_count_sum = junction_nodes_count_sum = 0
        total_tcn_length_m_sum = 0.0
        component_lengths_all, num_components = [], 0
        average_component_size_m = largest_component_size_m = 0.0

    # branches_lengths_all from edges.length_m (same source as first pass)
    if not edges_all.empty:
        edge_lengths = edges_all.get("length_m", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        edge_orients = edges_all.get("orientation_deg_axial", pd.Series(dtype=float)).fillna(0).astype(float).to_numpy()
        # 95% trim mean (same as before)
        if len(edge_lengths) > 0:
            p95 = np.percentile(edge_lengths, 95)
            trimmed = edge_lengths[edge_lengths <= p95]
            mean_edge_length_m = float(trimmed.mean()) if len(trimmed) else 0.0
        else:
            mean_edge_length_m = 0.0
        # step-weighted orient ≈ length-weighted axial mean (parity when px size is uniform)
        mean_edge_orientation_deg_stepweighted = _axial_mean_deg_weighted(edge_orients, edge_lengths) if edge_lengths.sum() > 0 else -1.0
        num_edges_branchlevel = int(len(edges_all))
    else:
        mean_edge_length_m = 0.0
        mean_edge_orientation_deg_stepweighted = -1.0
        num_edges_branchlevel = 0

    average_node_degree = (2.0 * num_graph_edges_sum / num_graph_nodes_sum) if num_graph_nodes_sum > 0 else 0.0

    # -------- write aggregate per-tile GPKG (layers identical) --------
    gpkg_path = os.path.join(output_tiles_dir, f"{tile_id}", f"{tile_id}_TCN_summary.gpkg")
    os.makedirs(output_tiles_dir, exist_ok=True)

    def _write_layer(gdf, layer):
        if gdf is None or gdf.empty:
            return
        if crs_wkt and (gdf.crs is None):
            gdf = gdf.set_crs(crs_wkt)
        mode = "a" if os.path.exists(gpkg_path) else "w"
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG", mode=mode)
        logging.info(f"[{tile_id}] wrote {len(gdf):,} to '{layer}'")

    with StageTimer(verbose, f"[{tile_id}]", "WRITE aggregate layers"):
        _write_layer(edges_all, "GraphTheoraticEdges")
        _write_layer(nodes_all, "GraphTheoraticNodes")
        _write_layer(comps_all, "GraphTheoraticComponents")

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

################# </AGGREGATION>#################################################
# ------------------------------ master build ------------------------------


# ------------------------------ startup sanity ------------------------------


def _agg_worker_OLD(task):
    """Unpack (tile_id, output_tiles_dir, smart_verbose) and run the aggregator."""
    tile_id, output_tiles_dir, output_tiles_dir, smart_verbose = task
    return aggregate_tile_from_perfile_gpkgs(tile_id, output_tiles_dir, output_tiles_dir, verbose=smart_verbose)

# --- imports at top ---
import multiprocessing as mp
import psutil

def _agg_worker(task):
    """task = (tile_id, in_dir, out_dir, smart_verbose)"""
    tile_id, in_dir, out_dir, smart_verbose = task
    try:
        # Must open/close all handles inside; write atomically.
        return aggregate_tile_from_perfile_gpkgs(
            tile_id,
            in_dir,
            out_dir,
            verbose=smart_verbose,
        )
    except Exception:
        logging.exception(f"[{tile_id}] aggregation failed")
        return None

def _set_start_method():
    # safer with GDAL/SQLite in many HPC stacks
    m = mp.get_start_method(allow_none=True)
    if m is None:
        try:
            mp.set_start_method("forkserver")
        except RuntimeError:
            pass

def _effective_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def _effective_mem_gb():
    # fall back to psutil; you can ignore SLURM/cgroups if you’re not using them
    return psutil.virtual_memory().total / 1e9

def _setup_fast_tmp(tmp_root=None):
    # Push temp to NVDIMM or /dev/shm for fast SQLite/tmp
    tmp_root = tmp_root or os.environ.get("TMPDIR") or "/dev/shm"
    os.makedirs(tmp_root, exist_ok=True)
    os.environ.setdefault("TMPDIR", tmp_root)
    os.environ.setdefault("CPL_TMPDIR", tmp_root)
    os.environ.setdefault("SQLITE_TMPDIR", tmp_root)
    # conservative, fast-ish SQLite/OGR knobs (single-writer per tile)
    os.environ.setdefault("OGR_SQLITE_SYNCHRONOUS", "OFF")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")

def _worker_init():
    # per-worker private tmp to reduce contention
    wtmp = os.path.join(os.environ.get("TMPDIR", "/dev/shm"), f"w{os.getpid()}")
    try: os.makedirs(wtmp, exist_ok=True)
    except Exception: pass
    os.environ["TMPDIR"] = wtmp
    os.environ["CPL_TMPDIR"] = wtmp
    os.environ["SQLITE_TMPDIR"] = wtmp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [PID %(process)d] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


#!/usr/bin/env python3
import os, sys, glob, argparse, sqlite3, time
from typing import List, Dict, Optional


# --- Assume your existing build_master_gpkg is importable in the same module or path ---
# from your_module import build_master_gpkg
def build_master_gpkg_stub(mosaic_gpkg_path, per_tile_rows, imagery_dir):
    """PLACEHOLDER: Replace with your existing implementation."""
    raise NotImplementedError("Import your existing build_master_gpkg() here.")


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




