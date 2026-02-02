#!/usr/bin/env python3
"""
Written for ocean filter from NSIDC water polygons. But can be used with any filter layer
Note it will only put a symlink to files that are the same after the filtering.

- Reads water polygons via GDAL/OGR (osgeo.ogr) OR from GeoJSON (stdlib json).
- Reprojects to EPSG:3338 (Alaska Albers), applies a -500 m buffer "toward sea".
- For each input mask tile:
  * Vector quick test in EPSG:3338 to drop pure-ocean tiles.
  * Otherwise rasterize water (all_touched=True) on the tile grid and write a
    cleaned "new mask" into --out-masks-dir using the SAME filename.
- Writes:
  * CSV (metrics + decision path)
  * drop_tiles.txt (pure ocean)
  * boundary_tiles.txt (tiles rewritten & to be reprocessed)

Requires: numpy, rasterio, shapely, pyproj, GDAL (for OGR path).

Usage
-----
bash -lc '
python 0_filter_and_make_new_masks_tree.py --config config.toml --verbose
'

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera

Ocean filter + boundary mask rewrite

Behavior:
- DROP  : tiles whose footprint is (near) entirely ocean (vector test in EPSG:3338)
- KEEP  : tiles that are all land (vector test ~0)
- BOUNDARY_REWRITE: mixed tiles -> write a new mask with ocean zeroed (same basename) in --out-masks-dir

"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import glob
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, transform as shp_transform
from shapely import wkb, wkt
from shapely.validation import make_valid
from pyproj import CRS, Transformer

# Optional OGR for reading vector water input
try:
    from osgeo import ogr
    HAS_OGR = True
except Exception:
    HAS_OGR = False

ALASKA_ALBERS = CRS.from_epsg(3338)

SAFE_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_TIFF_INTERNAL_MASK="NO",
)

# ----------------------------- TOML helpers (local fallback) -----------------------------
from gt_gpkg_common import load_toml, cfg_get
# ----------------------------- CLI ---------------------------------
def parse_args():
    p = argparse.ArgumentParser("Step 0: Filter ocean tiles + rewrite boundary masks")

    p.add_argument("--config", default=None, help="Path to config.toml (optional but recommended).")

    # Ocean source: either GPKG (OGR) or GeoJSON
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--ocean-gpkg", dest="ocean_gpkg", default=None,
                     help="Path to ocean/water polygons (GPKG/Shapefile/etc via OGR).")
    src.add_argument("--water-geojson", dest="water_geojson", default=None,
                     help="Path to GeoJSON water polygons.")

    p.add_argument("--ocean-layer", default=None,
                   help="Layer name in the GPKG (default: config [io].ocean_mask_layer or first layer).")

    # Inputs: either explicit globs/list OR inferred from config io.model_mask_dir / --in-root
    p.add_argument("--tifs", action="append", default=None,
                   help="Glob(s) for input mask GeoTIFFs (can repeat).")
    p.add_argument("--tif-list", dest="tif_list", default=None,
                   help="Text file with one TIFF path per line.")

    p.add_argument("--in-root", default=None,
                   help="Root to preserve folder structure. If omitted, uses config io.model_mask_dir or commonpath.")
    p.add_argument("--out-masks-dir", required=False, default=None,
                   help="Output root for cleaned+mirrored masks. If omitted, uses config io.master_dir.")

    p.add_argument("--ocean-threshold", type=float, default=None,
                   help="Drop tile if vector ocean fraction >= this. (default from config or 0.99)")
    p.add_argument("--buffer-m", type=float, default=None,
                   help="Negative buffer on water polygon, meters. (default from config or 500)")
    p.add_argument("--simplify-m", type=float, default=None,
                   help="Simplify tolerance in meters. (default from config or 0)")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Processes (default: min(4, max(1, cpu//4))).")
    p.add_argument("--gdal-cache-mb", type=int, default=128,
                   help="GDAL cache per process (MB).")

    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink",
                   help="How to mirror KEEP tiles into output tree.")
    p.add_argument("--overwrite-existing", action="store_true",
                   help="Overwrite existing outputs/links in output tree.")
    p.add_argument("--include-dropped", action="store_true",
                   help="Also mirror DROP tiles into output tree (normally skipped).")
    p.add_argument("--absolute-symlinks", action="store_true",
                   help="Create absolute symlinks (only for link-mode=symlink).")

    p.add_argument("--out-csv", default="ocean_filter.csv")
    p.add_argument("--out-drop", default="drop_tiles.txt")
    p.add_argument("--out-boundary", default="boundary_tiles.txt")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# --------------------------- helpers -------------------------------
def _polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None

def collect_tifs(args, cfg: Optional[dict]) -> List[str]:
    paths = set()

    # 1) explicit globs
    if args.tifs:
        for pat in args.tifs:
            for p in glob.glob(pat, recursive=True):
                if p.lower().endswith((".tif", ".tiff")):
                    paths.add(os.path.abspath(p))

    # 2) explicit list file
    if args.tif_list and os.path.exists(args.tif_list):
        with open(args.tif_list, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.add(os.path.abspath(p))

    # 3) if nothing provided, use config io.model_mask_dir (recursive)
    if not paths and cfg is not None:
        model_root = cfg_get(cfg, "io", "model_mask_dir", default=None)
        if model_root:
            pat = os.path.join(model_root, "**", "*.tif")
            for p in glob.glob(pat, recursive=True):
                paths.add(os.path.abspath(p))

    return sorted(paths)

def infer_in_root(tif_paths, explicit_in_root=None):
    if explicit_in_root:
        return os.path.abspath(explicit_in_root)
    try:
        return os.path.commonpath([os.path.abspath(p) for p in tif_paths])
    except Exception:
        return os.path.dirname(os.path.abspath(tif_paths[0])) if tif_paths else os.getcwd()

def compute_out_path(src_path, in_root, out_root):
    src_abs = os.path.abspath(src_path)
    in_root_abs = os.path.abspath(in_root)
    try:
        rel = os.path.relpath(src_abs, start=in_root_abs)
        if rel.startswith(".." + os.sep) or rel == "..":
            raise ValueError("src not under in_root")
        return os.path.join(out_root, rel)
    except Exception:
        return os.path.join(out_root, os.path.basename(src_abs))

def safe_remove_path(p):
    try:
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
    except FileNotFoundError:
        return
    except IsADirectoryError:
        return

def mirror_file(src_path, dst_path, mode="symlink", overwrite=False, absolute_symlinks=False):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    if overwrite and os.path.exists(dst_path):
        safe_remove_path(dst_path)
    if os.path.exists(dst_path):
        return

    if mode == "copy":
        shutil.copy2(src_path, dst_path); return

    if mode == "hardlink":
        try:
            os.link(src_path, dst_path); return
        except OSError:
            shutil.copy2(src_path, dst_path); return

    # symlink
    try:
        if absolute_symlinks:
            os.symlink(os.path.abspath(src_path), dst_path)
        else:
            rel = os.path.relpath(os.path.abspath(src_path), start=os.path.dirname(os.path.abspath(dst_path)))
            os.symlink(rel, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

def list_gpkg_layers(path: str) -> List[str]:
    if not HAS_OGR:
        return []
    ds = ogr.Open(path)
    if ds is None:
        return []
    return [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]

def _ogr_geom_to_shapely(g):
    try:
        buf = g.ExportToWkb()
        if isinstance(buf, (bytes, bytearray)):
            return wkb.loads(buf)
    except Exception:
        pass
    return wkt.loads(g.ExportToWkt())

def _read_water_geom_ogr(path: str, layer_name: Optional[str]) -> Tuple[object, Optional[CRS]]:
    """
    Read polygons from a vector datasource (GPKG, Shapefile, etc) via OGR.
    If layer_name is provided, only that layer is read; otherwise all layers are unioned.
    """
    if not HAS_OGR:
        raise RuntimeError("GDAL/OGR not available; use --water-geojson or install GDAL.")

    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"OGR failed to open: {path}")

    geoms = []
    layer_crs = None

    if layer_name:
        lyr = ds.GetLayerByName(layer_name)
        if lyr is None:
            layers = list_gpkg_layers(path)
            raise RuntimeError(f"Layer '{layer_name}' not found. Available: {layers}")
        layers_iter = [lyr]
    else:
        layers_iter = [ds.GetLayerByIndex(i) for i in range(ds.GetLayerCount())]

    for lyr in layers_iter:
        srs = lyr.GetSpatialRef()
        if srs and layer_crs is None:
            try:
                layer_crs = CRS.from_wkt(srs.ExportToWkt())
            except Exception:
                layer_crs = None

        lyr.ResetReading()
        for feat in lyr:
            g = feat.GetGeometryRef()
            if g:
                try:
                    g = g.Clone(); g.FlattenTo2D()
                except Exception:
                    pass
                geoms.append(_ogr_geom_to_shapely(g))

    if not geoms:
        raise RuntimeError("No geometries found in vector file/layer.")
    return unary_union(geoms), layer_crs

def _read_water_geom_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    crs = None
    if isinstance(gj, dict) and "crs" in gj and gj["crs"]:
        try:
            props = gj["crs"].get("properties", {})
            name = props.get("name") or props.get("code")
            if name:
                crs = CRS.from_user_input(name)
        except Exception:
            crs = None
    if crs is None:
        crs = CRS.from_epsg(4326)

    geoms = []
    if gj.get("type") == "FeatureCollection":
        for feat in gj.get("features", []):
            if "geometry" in feat and feat["geometry"]:
                geoms.append(shape(feat["geometry"]))
    elif gj.get("type") in ("Feature", "Polygon", "MultiPolygon"):
        geom = shape(gj["geometry"] if gj.get("type") == "Feature" else gj)
        geoms.append(geom)
    else:
        raise RuntimeError("Unsupported GeoJSON structure.")
    if not geoms:
        raise RuntimeError("No geometries found in GeoJSON.")
    return unary_union(geoms), crs

def load_water_geometry_albers(ocean_gpkg: Optional[str],
                              ocean_layer: Optional[str],
                              water_geojson: Optional[str],
                              buffer_m: float,
                              simplify_m: float) -> str:
    if ocean_gpkg:
        water, src_crs = _read_water_geom_ogr(ocean_gpkg, ocean_layer)
    elif water_geojson:
        water, src_crs = _read_water_geom_geojson(water_geojson)
    else:
        raise RuntimeError("No ocean source specified (use --ocean-gpkg/--water-geojson or config [io].ocean_mask_shp).")

    if src_crs is None:
        src_crs = ALASKA_ALBERS

    if src_crs != ALASKA_ALBERS:
        to_alb = Transformer.from_crs(src_crs, ALASKA_ALBERS, always_xy=True).transform
        water = shp_transform(to_alb, water)

    # Topology clean
    try:
        water = make_valid(water)
    except Exception:
        water = water.buffer(0)

    # Negative buffer (seawards)
    if buffer_m and buffer_m > 0:
        water = water.buffer(-buffer_m)
        if water.is_empty:
            raise RuntimeError("Water geometry vanished after -buffer. Reduce buffer_m.")

    if simplify_m and simplify_m > 0:
        try:
            water = water.simplify(simplify_m, preserve_topology=True)
        except Exception:
            pass

    try:
        water = make_valid(water)
    except Exception:
        water = water.buffer(0)

    return wkb.dumps(water, hex=True)

def vector_ocean_fraction_albers(tile_bounds_src_crs, tile_crs, water_wkb_hex) -> float:
    minx, miny, maxx, maxy = transform_bounds(tile_crs, ALASKA_ALBERS, *tile_bounds_src_crs, densify_pts=21)
    tile_poly_alb = box(minx, miny, maxx, maxy)
    water_alb = wkb.loads(water_wkb_hex, hex=True)
    inter = water_alb.intersection(tile_poly_alb)
    area_tile = tile_poly_alb.area
    return float(inter.area / area_tile) if area_tile > 0 else 0.0

def rasterize_water_full(ds, water_tile):
    return rasterize(
        [(water_tile, 1)],
        out_shape=(ds.height, ds.width),
        transform=ds.transform,
        all_touched=True,
        fill=0,
        dtype="uint8",
    )

def write_clean_mask_fullframe(ds, water_tile, out_mask_path: str, nodata):
    os.makedirs(os.path.dirname(os.path.abspath(out_mask_path)) or ".", exist_ok=True)
    if os.path.exists(out_mask_path) or os.path.islink(out_mask_path):
        safe_remove_path(out_mask_path)

    height, width = ds.height, ds.width
    profile = ds.profile.copy()
    if nodata is not None:
        profile.update(nodata=nodata)

    def _round16(n): return max(16, (n // 16) * 16)
    bx = _round16(min(512, width))
    by = _round16(min(512, height))

    profile.update(
        driver="GTiff",
        tiled=True,
        blockxsize=bx,
        blockysize=by,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    water_mask = rasterize_water_full(ds, water_tile)
    ocean = water_mask > 0

    if nodata is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        with rasterio.Env(**SAFE_ENV):
            b1 = ds.read(1, masked=False)
        if isinstance(nodata, float) and np.isnan(nodata):
            valid = ~np.isnan(b1)
        else:
            valid = (b1 != nodata)

    land = valid & ~ocean
    ocean_valid = ocean & valid

    with rasterio.Env(**SAFE_ENV):
        with rasterio.open(out_mask_path, "w", **profile) as dst:
            for b in range(1, ds.count + 1):
                band = ds.read(b, masked=False)
                if nodata is None:
                    cleaned = np.where(land, band, 0)
                else:
                    fill = np.nan if (isinstance(nodata, float) and np.isnan(nodata)) else nodata
                    cleaned = np.where(valid, np.where(land, band, 0), fill)

                if cleaned.dtype != band.dtype:
                    cleaned = cleaned.astype(band.dtype, copy=False)
                dst.write(cleaned, b)

    return int(land.sum()), int(ocean_valid.sum()), int(valid.sum())

def process_one_tile(tile_path: str,
                     out_mask_path: str,
                     water_wkb_hex: str,
                     ocean_threshold: float,
                     gdal_cache_mb: int,
                     simplify_m: float):
    """
    Returns:
      (src_path, out_path, ocean_vec, ocean_ras, land_cnt, decision, written_mask_path)
    """
    try:
        with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb, GDAL_NUM_THREADS="1", **SAFE_ENV):
            with rasterio.open(tile_path) as ds:
                if ds.crs is None:
                    return (tile_path, out_mask_path, None, None, None, "ERROR: tile has no CRS", None)

                ocean_frac_vec = vector_ocean_fraction_albers(ds.bounds, ds.crs, water_wkb_hex)

                # DROP
                if ocean_frac_vec >= ocean_threshold:
                    return (tile_path, out_mask_path, ocean_frac_vec, None, None, "DROP", None)

                # KEEP (pure land-ish)
                if ocean_frac_vec < 0.01:
                    return (tile_path, out_mask_path, ocean_frac_vec, 0.0, 0, "KEEP", None)

                # boundary: reproject water to tile CRS and clip
                to_tile = Transformer.from_crs(ALASKA_ALBERS, ds.crs, always_xy=True).transform
                water_tile = shp_transform(to_tile, wkb.loads(water_wkb_hex, hex=True))

                px = max(abs(ds.transform.a), abs(ds.transform.e))
                tile_box = box(*ds.bounds).buffer(2 * px)

                try:
                    water_tile = make_valid(water_tile)
                except Exception:
                    water_tile = water_tile.buffer(0)

                water_tile = _polygonal_only(water_tile.intersection(tile_box))
                if not water_tile or water_tile.is_empty or water_tile.area == 0:
                    return (tile_path, out_mask_path, ocean_frac_vec, 0.0, 0, "KEEP", None)

                if simplify_m and simplify_m > 0:
                    try:
                        water_tile = water_tile.simplify(simplify_m, preserve_topology=True)
                    except Exception:
                        pass

                nodata = ds.nodata
                land_total, ocean_total, valid_total = write_clean_mask_fullframe(ds, water_tile, out_mask_path, nodata)
                raster_ocean_frac = (ocean_total / valid_total) if valid_total else 0.0

                decision = "BOUNDARY_REWRITE" if (ocean_total > 0 and land_total > 0) else "KEEP"
                written_path = out_mask_path if decision == "BOUNDARY_REWRITE" else None
                return (tile_path, out_mask_path, ocean_frac_vec, raster_ocean_frac, land_total, decision, written_path)

    except Exception as e:
        return (tile_path, out_mask_path, None, None, None, f"ERROR: {e}", None)

# ------------------------------ main ------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    cfg = None
    if args.config:
        cfg = load_toml(args.config)

    # Resolve ocean source defaults from config
    ocean_gpkg = args.ocean_gpkg or (cfg_get(cfg, "io", "ocean_mask_shp", default=None) if cfg else None)
    ocean_layer = args.ocean_layer or (cfg_get(cfg, "io", "ocean_mask_layer", default=None) if cfg else None)

    # Resolve output dir defaults from config
    out_root = args.out_masks_dir or (cfg_get(cfg, "io", "master_dir", default=None) if cfg else None)
    if not out_root:
        raise ValueError("Need --out-masks-dir or config [io].master_dir")

    paths = collect_tifs(args, cfg)
    if not paths:
        logging.error("No TIFFs found. Use --tifs/--tif-list or set config [io].model_mask_dir.")
        return

    in_root = args.in_root or (cfg_get(cfg, "io", "model_mask_dir", default=None) if cfg else None)
    in_root = infer_in_root(paths, in_root)
    out_root = os.path.abspath(out_root)

    # Defaults (CLI overrides config)
    ocean_threshold = args.ocean_threshold if args.ocean_threshold is not None else float(cfg_get(cfg, "steps", "filter", "ocean_threshold", default=0.99) if cfg else 0.99)
    buffer_m = args.buffer_m if args.buffer_m is not None else float(cfg_get(cfg, "steps", "filter", "buffer_m", default=500.0) if cfg else 500.0)
    simplify_m = args.simplify_m if args.simplify_m is not None else float(cfg_get(cfg, "steps", "filter", "simplify_m", default=0.0) if cfg else 0.0)

    if args.max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(4, max(1, cpu // 4))
    else:
        max_workers = max(1, args.max_workers)

    # Load water once and broadcast as WKB hex
    if ocean_gpkg:
        if ocean_layer is None:
            # nice behavior: auto-pick first layer if user didn’t specify
            layers = list_gpkg_layers(ocean_gpkg)
            if layers:
                ocean_layer = layers[0]
                logging.info("No --ocean-layer provided; using first layer: %s", ocean_layer)
        logging.info("Ocean source: %s (layer=%s)", ocean_gpkg, ocean_layer)
    else:
        logging.info("Ocean source: geojson=%s", args.water_geojson)

    water_wkb_hex = load_water_geometry_albers(
        ocean_gpkg=ocean_gpkg,
        ocean_layer=ocean_layer,
        water_geojson=args.water_geojson,
        buffer_m=buffer_m,
        simplify_m=simplify_m,
    )

    jobs = [(p, compute_out_path(p, in_root, out_root)) for p in paths]

    logging.info("Input root : %s", in_root)
    logging.info("Output root: %s", out_root)
    logging.info("Tiles=%d | workers=%d | ocean_threshold=%.3f | buffer=%.1f m | simplify=%.1f m",
                 len(paths), max_workers, ocean_threshold, buffer_m, simplify_m)

    rows, dropped, boundary = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(process_one_tile, src, outp, water_wkb_hex, ocean_threshold, args.gdal_cache_mb, simplify_m)
            for (src, outp) in jobs
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)

            src_path, out_path, ocean_vec, ocean_ras, land_cnt, decision, written_path = row
            if decision == "DROP":
                dropped.append(out_path)
            elif decision == "BOUNDARY_REWRITE":
                boundary.append(written_path or out_path)

            if args.log_every and (i % args.log_every == 0 or i == len(paths)):
                logging.info("...processed %d/%d", i, len(paths))

    # Mirror KEEP tiles (and optionally DROP) into output tree so step 1 can consume --master_dir=out_root.
    mirrored = 0
    for (src_path, out_path, ocean_vec, ocean_ras, land_cnt, decision, written_path) in rows:
        if decision == "DROP" and not args.include_dropped:
            continue

        # If not rewritten, mirror original into output tree
        if decision in ("KEEP", "DROP") or (decision == "BOUNDARY_REWRITE" and not written_path):
            try:
                mirror_file(src_path, out_path, mode=args.link_mode,
                            overwrite=args.overwrite_existing,
                            absolute_symlinks=args.absolute_symlinks)
                mirrored += 1
            except Exception as e:
                logging.warning("Could not mirror %s -> %s: %s", src_path, out_path, e)

    if mirrored:
        logging.info("Mirrored %d tile(s) into %s", mirrored, out_root)

    # CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_path", "out_path", "ocean_fraction_vector", "ocean_fraction_raster_valid",
                    "land_pixel_count", "decision", "written_mask_path"])
        for r in rows:
            w.writerow(r)

    # Lists
    os.makedirs(os.path.dirname(os.path.abspath(args.out_drop)) or ".", exist_ok=True)
    with open(args.out_drop, "w") as f:
        for p in dropped:
            f.write(p + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_boundary)) or ".", exist_ok=True)
    with open(args.out_boundary, "w") as f:
        for p in boundary:
            f.write(p + "\n")

    kept_n = sum(1 for r in rows if r[5] == "KEEP")
    dropped_n = sum(1 for r in rows if r[5] == "DROP")
    rewritten_n = sum(1 for r in rows if r[5] == "BOUNDARY_REWRITE")
    errors_n = sum(1 for r in rows if isinstance(r[5], str) and r[5].startswith("ERROR"))

    logging.info(
        "Summary: %d KEEP (mirrored=%d), %d BOUNDARY_REWRITE (rewritten=%d), %d DROP, %d ERROR (total proc=%d)",
        kept_n, mirrored, rewritten_n, rewritten_n, dropped_n, errors_n, len(rows)
    )


if __name__ == "__main__":
    os.environ.setdefault("GDAL_CACHEMAX", "128")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    for k, v in SAFE_ENV.items():
        os.environ.setdefault(k, v)
    main()
