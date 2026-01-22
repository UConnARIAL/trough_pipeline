#!/usr/bin/env python3
"""
lib/get_cutline_vector_...., Refactored to keep stage-specific logic
This script does job of extracting the image cutlines used for the subtile from the input mosaic

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera
"""
import argparse
import rasterio
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

DEFAULT_CUTLINE_ROOT = "../PDG_shared/AlaskaTundraMosaic/imagery"

def find_cutline_vector_path(*, cutline_root: str | Path | None, tif_path: str, tile_id: str | None) -> str | None:
    """
        Locate the cutlines vector file for a given subtile raster.

        Args:
            cutline_root: Root directory containing per-tile cutlines (e.g., .../imagery).
                May be a Path, a string, or None (use module default).
            tif_path: Path to the subtile raster (.tif). Used to infer tile_id if tile_id is None.
            tile_id: Mosaic tile identifier like "54_12". If None, the function attempts
                to infer it from tif_path (parent folder or filename).

        Returns:
            Full path (string) to the cutlines vector file if found (e.g., .shp or .gpkg),
            otherwise None.
        """
    root = Path(cutline_root) if cutline_root is not None else DEFAULT_CUTLINE_ROOT

    if tile_id is None:
        p = Path(tif_path)
        # try parent folder name first
        if p.parent.name.count("_") == 1:
            tile_id = p.parent.name
        else:
            # fallback: parse from filename: ArcticMosaic_46_19_1_1_mask.tif -> 46_19
            stem = p.stem
            parts = stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                tile_id = f"{parts[1]}_{parts[2]}"
            else:
                return None
    candidate = root / tile_id / f"ArcticMosaic_{tile_id}_cutlines.shp"
    return str(candidate) if candidate.exists() else None

def clip_vector_to_raster_extent(raster_path: str, vector_path: str, *, fix_invalid: bool = False) -> gpd.GeoDataFrame:
    """
        Clip a vector dataset to the bounding box (extent) of a raster.

        The vector is reprojected to the raster CRS (if needed), then features are
        filtered by bbox intersection and geometries are intersected with the bbox.

        Args:
            raster_path: Path to the raster (.tif/.tiff) whose bounds define the clip extent.
            vector_path: Path to the vector dataset (e.g., .shp/.gpkg) to be clipped.
            fix_invalid: If True, attempt to repair invalid geometries prior to clipping
                (e.g., via make_valid/buffer(0)). Default False to preserve production behavior.

        Returns:
            A GeoDataFrame in the raster CRS containing geometries clipped to the raster bbox.
            May be empty if no features intersect the raster extent.
        """
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
        if raster_crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=raster_crs)

    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {vector_path}")

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    if fix_invalid:
        try:
            gdf["geometry"] = gdf.geometry.make_valid()
        except Exception:
            gdf["geometry"] = gdf.geometry.buffer(0)

    raster_bbox = box(*raster_bounds)
    gdf = gdf[gdf.geometry.notna()].copy()

    clipped_gdf = gdf[gdf.geometry.intersects(raster_bbox)].copy()
    if clipped_gdf.empty:
        return gpd.GeoDataFrame(clipped_gdf, geometry="geometry", crs=raster_crs)

    clipped_gdf["geometry"] = clipped_gdf.geometry.intersection(raster_bbox)
    clipped_gdf = clipped_gdf[clipped_gdf.geometry.notna() & (~clipped_gdf.geometry.is_empty)].copy()
    return gpd.GeoDataFrame(clipped_gdf, geometry="geometry", crs=raster_crs)

def main():
    # For testing
    ap = argparse.ArgumentParser("Integration test: clip cutlines to raster extent (read-only inputs)")
    ap.add_argument("--raster", required=True, help="Input raster .tif (read-only OK)")
    ap.add_argument("--vector", required=True, help="Input cutlines .shp/.gpkg (read-only OK)")
    ap.add_argument("--fix-invalid", action="store_true", help="Attempt to fix invalid geometries")
    ap.add_argument("--write-out-gpkg", default=None, help="Optional: write result to this GPKG")
    ap.add_argument("--layer", default="cutlines_clipped", help="Layer name if writing output")
    args = ap.parse_args()

    out = clip_vector_to_raster_extent(args.raster, args.vector, fix_invalid=args.fix_invalid)

    with rasterio.open(args.raster) as src:
        bbox = box(*src.bounds)
        rcrs = src.crs

    # Assertions / checks
    if rcrs is None:
        raise SystemExit("FAIL: raster has no CRS")
    if out.crs != rcrs:
        raise SystemExit(f"FAIL: CRS mismatch (out={out.crs}, raster={rcrs})")

    if not out.empty:
        bad = ~out.geometry.within(bbox.buffer(0))
        if bad.sum() != 0:
            raise SystemExit(f"FAIL: {bad.sum()} geometries fall outside raster bbox after clipping")

    print(f"PASS: produced {len(out)} feature(s)")

    if args.write_out_gpkg:
        out.to_file(args.write_out_gpkg, layer=args.layer, driver="GPKG")
        print(f"WROTE: {args.write_out_gpkg} layer={args.layer}")

if __name__ == "__main__":
    main()

"""
USAGE for testing
python get_cutline_vector_for_rastor.py \
  --raster ./PDG_shared/AlaskaTundraMosaic/imagery/54_12/ArcticMosaic_54_12_cutlines.shp \
  --vector ../TCN_refac_test_imgs/54_12/ArcticMosaic_54_12_5_4_mask.tif
"""
