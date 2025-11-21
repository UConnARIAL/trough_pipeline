Usage
-----
python ocean_filter_make_new_masks.py \
  --water-shp /path/arctic_water.shp \
  --tifs "/data/masks/**/*.tif" \
  --ocean-threshold 0.99 \
  --buffer-m 500 \
  --out-masks-dir /data/new_masks \
  --out-csv ocean_filter.csv \
  --out-drop drop_tiles.txt \
  --out-boundary boundary_tiles.txt \
  --max-workers 5

# OR if you pre-convert to GeoJSON:
python ocean_filter_make_new_masks_nofiona.py \
  --water-geojson /path/arctic_water_3338.json \
  ... (same args)
"""

#!/usr/bin/env python3
"""
Ocean filter + boundary mask rewrite

Behavior:
- DROP  : tiles whose footprint is (near) entirely ocean (vector test in EPSG:3338)
- KEEP  : tiles that are all land (vector test ~0)
- BOUNDARY_REWRITE: mixed tiles -> write a new mask with ocean zeroed (same basename) in --out-masks-dir

Key robustness:
- Never calls read_masks(); validity is derived from nodata or assumed all-valid
- Reads for boundary tiles are full-frame (like your pipeline) to avoid mask/overview sub-IFDs
- Wraps all opens/reads in a SAFE GDAL env to avoid scanning .ovr/.msk and internal mask IFDs
"""

import argparse
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import shape, box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, transform as shp_transform
from shapely import wkb, wkt
from shapely.validation import make_valid
from pyproj import CRS, Transformer

# Optional OGR for reading vector water input
try:
    from osgeo import ogr  # optional
    HAS_OGR = True
except Exception:
    HAS_OGR = False

ALASKA_ALBERS = CRS.from_epsg(3338)

# Keep GDAL on the base image only (don’t touch .ovr/.msk or internal mask IFDs)
SAFE_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    GDAL_TIFF_INTERNAL_MASK='NO',
)

# ----------------------------- CLI ---------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--water-shp", help="Path to water polygons (GPKG/ESRI Shapefile/etc., via GDAL/OGR).")
    src.add_argument("--water-geojson", help="Path to GeoJSON water polygons (read via stdlib json).")

    p.add_argument("--tifs", action="append", help="Glob(s) for input mask GeoTIFFs. Can be used multiple times.")
    p.add_argument("--tif-list", dest="tif_list", help="Optional text file with one TIFF path per line.")
    p.add_argument("--ocean-threshold", type=float, default=0.99, help="Drop tile if vector ocean fraction >= this.")
    p.add_argument("--buffer-m", type=float, default=500.0, help="Negative buffer on water polygon (toward sea), meters.")
    p.add_argument("--simplify-m", type=float, default=0.0, help="Simplify tolerance (m) pre-rasterization.")
    p.add_argument("--window-size", type=int, default=2048, help="(Unused in full-frame mode; kept for parity/logging).")
    p.add_argument("--max-workers", type=int, default=None, help="Processes (default: ~CPU/4 capped at 4).")
    p.add_argument("--gdal-cache-mb", type=int, default=128, help="GDAL cache per process (MB).")
    p.add_argument("--out-masks-dir", required=True, help="Directory to write cleaned masks for boundary tiles.")
    p.add_argument("--out-csv", default="ocean_filter.csv", help="CSV with per-tile metrics.")
    p.add_argument("--out-drop", default="drop_tiles.txt", help="List of pure-ocean tiles to drop.")
    p.add_argument("--out-boundary", default="boundary_tiles.txt", help="List of boundary tiles (cleaned + reprocess).")
    p.add_argument("--log-every", type=int, default=1, help="Print progress every N tiles.")
    return p.parse_args()

# --------------------------- helpers -------------------------------
def _polygonal_only(geom):
    """Return only polygonal parts; None if empty/non-polygonal."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None

def iter_windows(height, width, block):
    rows = int(np.ceil(height / block))
    cols = int(np.ceil(width / block))
    for r in range(rows):
        for c in range(cols):
            h = block if (r+1)*block <= height else height - r*block
            w = block if (c+1)*block <= width else width - c*block
            yield Window(c*block, r*block, w, h)

def rasterize_water_full(ds, water_tile):
    """Full-frame rasterization of water polygon to ds grid."""
    return rasterize(
        [(water_tile, 1)],
        out_shape=(ds.height, ds.width),
        transform=ds.transform,
        all_touched=True,
        fill=0,
        dtype="uint8",
    )

def collect_tifs(args):
    paths = set()
    if args.tifs:
        for pat in args.tifs:
            for p in glob.glob(pat, recursive=True):
                if p.lower().endswith((".tif", ".tiff")):
                    paths.add(os.path.abspath(p))
    if args.tif_list and os.path.exists(args.tif_list):
        with open(args.tif_list, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.add(os.path.abspath(p))
    return sorted(paths)

def _ogr_geom_to_shapely(g):
    try:
        buf = g.ExportToWkb()
        if isinstance(buf, (bytes, bytearray)):
            return wkb.loads(buf)
    except Exception:
        pass
    return wkt.loads(g.ExportToWkt())

def _read_water_geom_ogr(path):
    if not HAS_OGR:
        raise RuntimeError("GDAL/OGR not available; use --water-geojson or install GDAL.")
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"OGR failed to open: {path}")
    geoms = []
    layer_crs = None
    for i in range(ds.GetLayerCount()):
        lyr = ds.GetLayerByIndex(i)
        srs = lyr.GetSpatialRef()
        if srs and layer_crs is None:
            try:
                layer_crs = CRS.from_wkt(srs.ExportToWkt())
            except Exception:
                layer_crs = None
        for feat in lyr:
            g = feat.GetGeometryRef()
            if g:
                try:
                    g = g.Clone(); g.FlattenTo2D()
                except Exception:
                    pass
                geoms.append(_ogr_geom_to_shapely(g))
    if not geoms:
        raise RuntimeError("No geometries found in vector file.")
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

def load_water_geometry_albers(args):
    if args.water_shp:
        water, src_crs = _read_water_geom_ogr(args.water_shp)
    else:
        water, src_crs = _read_water_geom_geojson(args.water_geojson)

    # CRS -> Alaska Albers
    if src_crs is None:
        # Assume already in 3338 if missing (common after manual clips); user confirmed 3413 earlier,
        # but here we operate in 3338 footprint space only; water itself is reprojected to tile CRS later.
        src_crs = ALASKA_ALBERS
    if src_crs != ALASKA_ALBERS:
        to_alb = Transformer.from_crs(src_crs, ALASKA_ALBERS, always_xy=True).transform
        water = shp_transform(to_alb, water)

    # Topology clean
    try:
        water = make_valid(water)
    except Exception:
        water = water.buffer(0)

    # Negative buffer seawards
    if args.buffer_m > 0:
        water = water.buffer(-args.buffer_m)
        if water.is_empty:
            raise RuntimeError("Water geometry vanished after -buffer. Reduce --buffer-m.")

    # Optional simplify
    if args.simplify_m and args.simplify_m > 0:
        try:
            water = water.simplify(args.simplify_m, preserve_topology=True)
        except Exception:
            pass

    # Final clean
    try:
        water = make_valid(water)
    except Exception:
        water = water.buffer(0)

    # Return hex WKB (so it's cheap to pass between processes)
    return wkb.dumps(water, hex=True)

def vector_ocean_fraction_albers(tile_bounds_src_crs, tile_crs, water_wkb_hex):
    # Compute ocean fraction of the tile's footprint in Alaska Albers
    minx, miny, maxx, maxy = transform_bounds(tile_crs, ALASKA_ALBERS, *tile_bounds_src_crs, densify_pts=21)
    tile_poly_alb = box(minx, miny, maxx, maxy)
    water_alb = wkb.loads(water_wkb_hex, hex=True)
    inter = water_alb.intersection(tile_poly_alb)
    area_tile = tile_poly_alb.area
    frac = float(inter.area / area_tile) if area_tile > 0 else 0.0
    print(f"tile_ocean_frac {frac}")
    return frac

# ------------------------ writers (full-frame) ---------------------
def write_clean_mask_fullframe(ds, water_tile, out_dir, nodata):
    """
    Boundary path: read the base image full-frame (pipeline style), zero ocean pixels, write new GTiff.
    - Never calls read_masks()
    - No windowed reads
    - Disables .ovr/.msk scanning to avoid bad IFDs
    Returns (out_path, land_total, ocean_total, valid_total)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(ds.name)
    out_mask_path = os.path.join(out_dir, base)

    height, width = ds.height, ds.width

    # Build destination profile (legal tile sizes, stable compression)
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
        compress="deflate",   # or 'zstd' if available
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    # 1) Full-frame water mask (all_touched=True)
    water_mask = rasterize_water_full(ds, water_tile)
    ocean = water_mask > 0

    # 2) Validity from nodata only (never touch TIFF mask IFD)
    if nodata is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        with rasterio.Env(**SAFE_ENV):
            b1 = ds.read(1, masked=False)  # base image full read (like your pipeline)
        if isinstance(nodata, float) and np.isnan(nodata):
            valid = ~np.isnan(b1)
        else:
            valid = (b1 != nodata)

    land = valid & ~ocean
    ocean_valid = ocean & valid

    # 3) Read–clean–write per band (full-frame) with mask/overview scanning disabled
    with rasterio.Env(**SAFE_ENV):
        with rasterio.open(out_mask_path, "w", **profile) as dst:
            for b in range(1, ds.count + 1):
                band = ds.read(b, masked=False)  # full-frame, base image only
                if nodata is None:
                    cleaned = np.where(land, band, 0)
                else:
                    fill = np.nan if (isinstance(nodata, float) and np.isnan(nodata)) else nodata
                    cleaned = np.where(valid, np.where(land, band, 0), fill)

                if cleaned.dtype != band.dtype:
                    cleaned = cleaned.astype(band.dtype, copy=False)
                dst.write(cleaned, b)

    land_total  = int(land.sum())
    ocean_total = int(ocean_valid.sum())
    valid_total = int(valid.sum())
    return out_mask_path, land_total, ocean_total, valid_total

# ------------------------- main worker -----------------------------
def process_one_tile(tile_path, water_wkb_hex, ocean_threshold, gdal_cache_mb,
                     out_masks_dir, window_size, simplify_m):
    out_mask_path = None
    try:
        with rasterio.Env(GDAL_CACHEMAX=gdal_cache_mb, GDAL_NUM_THREADS='1', **SAFE_ENV):
            with rasterio.open(tile_path) as ds:
                if ds.crs is None:
                    return (tile_path, None, None, None, "ERROR: tile has no CRS", out_mask_path)

                # Phase A: fast vector fraction in EPSG:3338 to drop/keep
                ocean_frac_vec = vector_ocean_fraction_albers(ds.bounds, ds.crs, water_wkb_hex)

                # 1) Pure ocean → DROP
                if ocean_frac_vec >= ocean_threshold:
                    return (tile_path, ocean_frac_vec, None, None, "DROP", out_mask_path)

                # 2) Pure land (tolerate tiny slivers) → KEEP
                if ocean_frac_vec < 0.01:
                    return (tile_path, ocean_frac_vec, 0.0, 0, "KEEP", out_mask_path)

                # 3) Boundary: reproject water to tile, clip to bbox (+2 px), polygon-only, optional simplify
                to_tile = Transformer.from_crs(ALASKA_ALBERS, ds.crs, always_xy=True).transform
                water_tile = shp_transform(to_tile, wkb.loads(water_wkb_hex, hex=True))

                # Clip to tile bounds with a small pixel buffer
                px = max(abs(ds.transform.a), abs(ds.transform.e))
                tile_box = box(*ds.bounds).buffer(2 * px)
                try:
                    water_tile = make_valid(water_tile)
                except Exception:
                    water_tile = water_tile.buffer(0)
                water_tile = _polygonal_only(water_tile.intersection(tile_box))

                if not water_tile or water_tile.is_empty or water_tile.area == 0:
                    # Nothing polygonal to remove in this tile
                    return (tile_path, ocean_frac_vec, 0.0, 0, "KEEP", out_mask_path)

                if simplify_m and simplify_m > 0:
                    try:
                        water_tile = water_tile.simplify(simplify_m, preserve_topology=True)
                    except Exception:
                        pass

                # 4) Full-frame rewrite (pipeline-style reads; SAFE_ENV)
                nodata = ds.nodata
                out_mask_path, land_total, ocean_total, valid_total = write_clean_mask_fullframe(
                    ds, water_tile, out_masks_dir, nodata
                )
                raster_ocean_frac = (ocean_total / valid_total) if valid_total else 0.0
                decision = "BOUNDARY_REWRITE" if (ocean_total > 0 and land_total > 0) else "KEEP"
                return (tile_path, ocean_frac_vec, raster_ocean_frac, land_total,
                        decision, out_mask_path)

    except Exception as e:
        return (tile_path, None, None, None, f"ERROR: {e}", out_mask_path)

# ------------------------------ main ------------------------------
def main():
    args = parse_args()
    paths = collect_tifs(args)
    if not paths:
        print("No TIFFs found. Check --tifs or --tif-list.")
        return

    water_wkb_hex = load_water_geometry_albers(args)

    if args.max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(4, max(1, cpu // 4))
    else:
        max_workers = max(1, args.max_workers)

    rows, dropped, boundary = [], [], []
    print(f"Processing {len(paths)} tiles with {max_workers} workers | "
          f"ocean-threshold={args.ocean_threshold} | buffer={args.buffer_m} m | simplify={args.simplify_m} m | "
          f"window={args.window_size} px | all_touched=True")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(process_one_tile, p, water_wkb_hex, args.ocean_threshold, args.gdal_cache_mb,
                      args.out_masks_dir, args.window_size, args.simplify_m)
            for p in paths
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if row[4] == "DROP":
                dropped.append(row[0])
            elif row[4] == "BOUNDARY_REWRITE":
                boundary.append(row[0])
            if args.log_every and (i % len(paths) == 0 or i % args.log_every == 0):
                print(f"...processed {i}/{len(paths)} tiles")

    # CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_path", "ocean_fraction_vector", "ocean_fraction_raster_valid",
                    "land_pixel_count", "decision", "output_mask_path"])
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

    kept_n = sum(1 for r in rows if r[4] == "KEEP")
    dropped_n = len(dropped)
    boundary_n = len(boundary)
    errors_n = sum(1 for r in rows if isinstance(r[4], str) and r[4].startswith("ERROR"))
    print(f"Summary: {kept_n} KEEP, {boundary_n} BOUNDARY_REWRITE, {dropped_n} DROP, {errors_n} ERROR out of {len(rows)}")

if __name__ == "__main__":
    os.environ.setdefault("GDAL_CACHEMAX", "128")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    # also set SAFE_ENV defaults at process start (workers inherit)
    for k, v in SAFE_ENV.items():
        os.environ.setdefault(k, v)
    main()


  
