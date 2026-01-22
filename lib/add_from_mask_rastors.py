# -*- coding: utf-8 -*-
"""
GeoPackage helpers: embed single-band rasters and polygonize to vector.

Key capabilities
----------------
1) add_raster_to_gpkg(...)
   - Accepts either a raster file path (GeoTIFF) OR a NumPy array (H,W) / single-band 3D.
   - Writes a tiled PNG8 raster table into a GeoPackage.
   - Optionally reprojects to a target CRS (or matches existing GPKG CRS if configured).
   - Optionally builds overviews.

2) polygonize_raster_table_to_vector(...)
   - Reads a raster table from the GeoPackage and polygonizes it into a MULTIPOLYGON vector layer.
   - Supports 4- or 8-connected polygonization, area filtering, and simplification.

3) add_masks_and_vector(...)
   - Convenience wrapper to add optional "original" mask + required "cleaned" mask + vector outline.

Notes
-----
- Uses GDAL/OGR Python bindings for writing rasters and polygonization.
- Does in-place writes to the GeoPackage.

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera

"""

from __future__ import annotations

# ----------------------- standard library -----------------------
import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Iterable, Dict
from collections import Counter
from os import PathLike

# ----------------------- third-party -----------------------
import numpy as np
from osgeo import gdal, ogr, osr

# Enable GDAL/OGR/OSR exceptions once at import time.
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

# ----------------------- utils -----------------------

def _ensure_gpkg_path(p: str | Path) -> str:
    """Ensure output path ends with .gpkg and parent directory exists."""
    p = Path(p)
    if p.suffix.lower() != ".gpkg":
        p = p.with_suffix(".gpkg")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _gpkg_has_layer(gpkg: str, layer: str) -> bool:
    """True if `layer` exists in `gpkg`."""
    ds = ogr.Open(gpkg, update=0)
    if not ds:
        return False
    try:
        return ds.GetLayerByName(layer) is not None
    finally:
        ds = None


def _srs_from_any(srs_like: str | osr.SpatialReference) -> osr.SpatialReference:
    """Coerce an EPSG/auth string or SpatialReference into a SpatialReference."""
    srs = osr.SpatialReference()
    if isinstance(srs_like, osr.SpatialReference):
        srs.ImportFromWkt(srs_like.ExportToWkt())
    else:
        srs.SetFromUserInput(str(srs_like))
    return srs


def _srs_auth_string(con: sqlite3.Connection, srs_id: int) -> Optional[str]:
    """Return 'ORG:CODE' (preferred) or WKT for a given srs_id from gpkg_spatial_ref_sys."""
    row = con.execute(
        "SELECT organization, organization_coordsys_id, definition "
        "FROM gpkg_spatial_ref_sys WHERE srs_id=?",
        (srs_id,),
    ).fetchone()
    if not row:
        return None
    org, code, wkt = row
    if org and code is not None:
        return f"{org.upper()}:{code}"
    return wkt


def _gpkg_preferred_srs(gpkg_path: str) -> Optional[str]:
    """
    Prefer an authority code like 'EPSG:3338' from existing GPKG contents.
    Uses the most common srs_id seen in geometry/raster metadata tables.
    """
    if not os.path.exists(gpkg_path):
        return None

    with sqlite3.connect(gpkg_path) as con:
        srs_ids = []
        srs_ids += [sid for (sid,) in con.execute(
            "SELECT DISTINCT srs_id FROM gpkg_geometry_columns WHERE srs_id IS NOT NULL"
        )]
        srs_ids += [sid for (sid,) in con.execute(
            "SELECT DISTINCT srs_id FROM gpkg_tile_matrix_set WHERE srs_id IS NOT NULL"
        )]
        if not srs_ids:
            return None

        sid = Counter(srs_ids).most_common(1)[0][0]
        return _srs_auth_string(con, sid)

def _wkt_from_dataset(ds: gdal.Dataset) -> str:
    """Get dataset projection WKT, or '' if unavailable."""
    try:
        return ds.GetProjection() or ""
    except Exception:
        return ""


def _same_crs(ds: gdal.Dataset, target_auth: str | None) -> bool:
    """True if dataset CRS is the same as target_auth (e.g., 'EPSG:3338')."""
    if not target_auth:
        return False
    s = osr.SpatialReference()
    wkt = ds.GetProjection() or ""
    if wkt:
        s.ImportFromWkt(wkt)
    t = osr.SpatialReference()
    t.SetFromUserInput(target_auth)
    return s.IsSame(t) == 1


def _units_name(wkt: str) -> str:
    """Return unit name (lowercase) for a WKT string, or ''."""
    s = osr.SpatialReference()
    if not wkt:
        return ""
    try:
        s.ImportFromWkt(wkt)
    except Exception:
        return ""
    return (s.GetLinearUnitsName() or s.GetAngularUnitsName() or "").lower()


def _both_metre_units(src_wkt: str, target_auth: str) -> bool:
    """True if both source and target CRS report linear/angular units as 'metre'."""
    if not src_wkt or not target_auth:
        return False
    t = osr.SpatialReference()
    t.SetFromUserInput(target_auth)
    dst_units = (t.GetLinearUnitsName() or t.GetAngularUnitsName() or "").lower()
    return _units_name(src_wkt) == "metre" and dst_units == "metre"


def _source_pixel_size(ds: gdal.Dataset) -> tuple[Optional[float], Optional[float]]:
    """Return (pixel_width, pixel_height) from GeoTransform, or (None,None)."""
    gt = ds.GetGeoTransform()
    if not gt:
        return (None, None)
    return (abs(gt[1]) if gt[1] else None, abs(gt[5]) if gt[5] else None)


def _to_multipolygon(geom: ogr.Geometry) -> ogr.Geometry:
    """Coerce geometry into MULTIPOLYGON (best-effort) and return a clone."""
    gt = geom.GetGeometryType()
    if gt in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        return geom.Clone()
    if gt in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        mp = ogr.Geometry(ogr.wkbMultiPolygon)
        mp.AddGeometry(geom.Clone())
        return mp
    try:
        return ogr.ForceToMultiPolygon(geom).Clone()
    except Exception:
        return geom.Clone()


# ----------------------- single-band raster writer -----------------------

def add_raster_to_gpkg(
    raster_data: str | Path | PathLike | np.ndarray,
    output_gpkg: str | Path,
    *,
    layer_name: str,
    crs: str | osr.SpatialReference | None = None,   # for NumPy input
    transform: Optional[tuple] = None,               # for NumPy input
    reproject_to: Optional[str] = None,              # e.g., "EPSG:3338"
    match_existing_crs: bool = True,
    assume_src_srs: Optional[str] = None,            # if input TIFF may lack CRS
    tile_blocksize: int = 512,
    build_overviews: Iterable[int] = (2, 4, 8, 16),
) -> None:
    """
    Add a single-band raster table to a GeoPackage.

    Args:
        raster_data: Path-like raster (e.g., GeoTIFF) OR NumPy array.
        output_gpkg: Destination .gpkg (created if missing).
        layer_name: New raster table name in the GPKG.
        crs/transform: Required when raster_data is a NumPy array.
        reproject_to: If set, warp raster into this CRS (auth string).
        match_existing_crs: If True and output exists, attempt to use existing GPKG CRS.
        assume_src_srs: If source raster has no CRS, assign this CRS for processing.
        tile_blocksize: Tile size for GPKG raster table.
        build_overviews: Overview factors to build after writing.
    """
    out_gpkg = _ensure_gpkg_path(output_gpkg)
    if _gpkg_has_layer(out_gpkg, layer_name):
        raise ValueError(f"Layer '{layer_name}' already exists in {out_gpkg}")

    # Decide target CRS
    target_srs = reproject_to
    if target_srs is None and match_existing_crs and Path(out_gpkg).exists():
        target_srs = _gpkg_preferred_srs(out_gpkg)

    logging.info(f"[add_raster_to_gpkg] target_srs={target_srs!r}")

    # ---------- Build source dataset ----------
    if isinstance(raster_data, (str, Path, PathLike)):
        src_path = str(raster_data)
        src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if src_ds is None:
            raise RuntimeError(f"Could not open raster: {src_path}")

        src_wkt = _wkt_from_dataset(src_ds)
        if not src_wkt:
            if not assume_src_srs:
                raise ValueError(
                    "Source raster has no CRS. Pass assume_src_srs='EPSG:xxxx' "
                    "or assign one with gdal_edit.py -a_srs."
                )
            logging.info(f"[add_raster_to_gpkg] assigning missing CRS -> {assume_src_srs}")
            gdal.Translate(
                "/vsimem/_assign_srs.tif",
                src_ds,
                options=gdal.TranslateOptions(outputSRS=assume_src_srs),
            )
            src_ds = gdal.Open("/vsimem/_assign_srs.tif", gdal.GA_ReadOnly)

    elif isinstance(raster_data, np.ndarray):
        arr = np.asarray(raster_data)

        # Accept (H,W), (H,W,1), (1,H,W), (H,W,3) → use first band
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] == 1 and arr.shape[1] > 1:
                arr = arr[0, :, :]
            else:
                logging.info(f"[add_raster_to_gpkg] WARNING: 3D array shape {arr.shape}; using first channel")
                arr = arr[..., 0]

        if arr.ndim != 2:
            raise ValueError("NumPy input must be 2-D (H, W) or 3-D with a single band.")

        mem = gdal.GetDriverByName("MEM").Create("", arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte)
        b = mem.GetRasterBand(1)
        b.WriteArray(arr.astype(np.uint8))
        b.SetNoDataValue(0)

        if transform is None:
            raise ValueError("transform is required with NumPy input.")
        gt = (
            (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
            if hasattr(transform, "a")
            else tuple(transform)
        )
        mem.SetGeoTransform(gt)

        if crs is None:
            raise ValueError("crs is required with NumPy input.")
        srs = _srs_from_any(crs)
        mem.SetProjection(srs.ExportToWkt())

        src_ds = mem

    else:
        raise TypeError(f"raster_data must be a path or numpy array, not {type(raster_data)}")

    src_wkt = _wkt_from_dataset(src_ds)
    need_warp = bool(target_srs) and not _same_crs(src_ds, target_srs)
    logging.info(f"[add_raster_to_gpkg] need_warp={need_warp}")

    creation_opts = [
        f"RASTER_TABLE={layer_name}",
        "TILE_FORMAT=PNG8",
        "DITHER=NO",
        f"BLOCKXSIZE={int(tile_blocksize)}",
        f"BLOCKYSIZE={int(tile_blocksize)}",
    ]
    if Path(out_gpkg).exists():
        creation_opts.append("APPEND_SUBDATASET=YES")

    try:
        if need_warp:
            xres, yres = _source_pixel_size(src_ds)
            same_units = _both_metre_units(src_wkt, target_srs)
            logging.info(f"[add_raster_to_gpkg] same_units={same_units} xres={xres} yres={yres}")

            wkw = dict(
                format="MEM",
                dstSRS=target_srs,
                resampleAlg="near",
                dstNodata=0,
                outputType=gdal.GDT_Byte,
                srcBands=[1],
                multithread=True,
            )
            if assume_src_srs:
                wkw["srcSRS"] = assume_src_srs
            if same_units and xres and yres:
                wkw.update(dict(xRes=xres, yRes=yres))

            mem_warp = gdal.Warp("", src_ds, options=gdal.WarpOptions(**wkw))
            topt = gdal.TranslateOptions(
                format="GPKG",
                noData=0,
                outputType=gdal.GDT_Byte,
                bandList=[1],
                creationOptions=creation_opts,
            )
            gdal.Translate(out_gpkg, mem_warp, options=topt)
            mem_warp = None
        else:
            topt = gdal.TranslateOptions(
                format="GPKG",
                noData=0,
                outputType=gdal.GDT_Byte,
                bandList=[1],
                creationOptions=creation_opts,
            )
            gdal.Translate(out_gpkg, src_ds, options=topt)

    except Exception as e:
        raise RuntimeError(f"[add_raster_to_gpkg] GDAL failed: {e}")

    if build_overviews:
        sub = gdal.Open(f"GPKG:{out_gpkg}:{layer_name}", gdal.GA_Update)
        if sub:
            sub.BuildOverviews("NEAREST", list(build_overviews))
            sub = None


# ----------------------- polygonize (cleaned raster -> vector) -----------------------

def polygonize_raster_table_to_vector(
    output_gpkg: str | Path,
    raster_table: str,
    vector_layer: str,
    *,
    eight_connected: bool = True,
    min_area: float = 0.0,
    simplify_tol: float = 0.0,
    overwrite: bool = True,
) -> None:
    """
    Polygonize a GPKG raster table band into a MULTIPOLYGON vector layer.

    - Uses a temporary in-memory POLYGON layer to avoid POLYGON->MULTIPOLYGON warnings.
    - Writes mask_value (integer) = 1 for all output polygons (as currently implemented).

    Args:
        output_gpkg: GeoPackage path.
        raster_table: Raster table name within the GPKG.
        vector_layer: Output vector layer name.
        eight_connected: If True, use 8-connected polygonization.
        min_area: Minimum polygon area filter (units of CRS).
        simplify_tol: Simplification tolerance (units of CRS).
        overwrite: If True and layer exists, delete and recreate.
    """
    out_gpkg = _ensure_gpkg_path(output_gpkg)

    sub = gdal.Open(f"GPKG:{out_gpkg}:{raster_table}", gdal.GA_ReadOnly)
    if sub is None:
        raise RuntimeError(f"Raster table not found: {raster_table}")

    band = sub.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(sub.GetProjection())

    drv_gpkg = ogr.GetDriverByName("GPKG")
    ds = drv_gpkg.Open(out_gpkg, update=1)
    if ds is None:
        sub = None
        raise RuntimeError(f"Cannot open {out_gpkg} for update.")

    if _gpkg_has_layer(out_gpkg, vector_layer):
        if not overwrite:
            ds = None
            sub = None
            raise ValueError(f"Vector layer '{vector_layer}' already exists.")
        ds.DeleteLayer(vector_layer)

    # Final MULTIPOLYGON layer
    out_lyr = ds.CreateLayer(vector_layer, srs=srs, geom_type=ogr.wkbMultiPolygon)
    out_lyr.CreateField(ogr.FieldDefn("mask_value", ogr.OFTInteger))
    out_defn = out_lyr.GetLayerDefn()
    out_idx = out_defn.GetFieldIndex("mask_value")

    # Temp POLYGON layer (avoid POLYGON→MULTIPOLYGON insert warning)
    mem_drv = ogr.GetDriverByName("Memory")
    mem_ds = mem_drv.CreateDataSource("mem_poly")
    tmp_lyr = mem_ds.CreateLayer("tmp", srs=srs, geom_type=ogr.wkbPolygon)
    tmp_lyr.CreateField(ogr.FieldDefn("mask_value", ogr.OFTInteger))
    tmp_defn = tmp_lyr.GetLayerDefn()
    tmp_idx = tmp_defn.GetFieldIndex("mask_value")
    if tmp_idx < 0:
        raise RuntimeError("Failed to create 'mask_value' field in temp layer.")

    opts = ["8CONNECTED=8"] if eight_connected else None
    gdal.Polygonize(band, band, tmp_lyr, tmp_idx, options=opts)

    # Copy → final, coerce to MULTIPOLYGON, apply filters
    tmp_lyr.ResetReading()
    created = 0
    f = tmp_lyr.GetNextFeature()
    while f:
        try:
            val = f.GetField("mask_value") or 0
            if val <= 0:
                f = tmp_lyr.GetNextFeature()
                continue

            geom = f.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                f = tmp_lyr.GetNextFeature()
                continue

            g = geom.Clone()
            if simplify_tol > 0:
                g = g.SimplifyPreserveTopology(float(simplify_tol))
            g = _to_multipolygon(g)

            if min_area > 0 and g.Area() <= float(min_area):
                f = tmp_lyr.GetNextFeature()
                continue

            nf = ogr.Feature(out_defn)
            nf.SetField(out_idx, 1)
            nf.SetGeometry(g)
            out_lyr.CreateFeature(nf)
            created += 1
            nf = None
        finally:
            f = tmp_lyr.GetNextFeature()

    try:
        out_lyr.SyncToDisk()
    except Exception:
        pass

    # Cleanup
    tmp_lyr = None
    mem_ds = None
    out_lyr = None
    ds = None
    sub = None

    logging.info(f"[polygonize] wrote {created} features to '{vector_layer}' in {out_gpkg}")


# ----------------------- wrapper: add original + cleaned + vector -----------------------

def add_masks_and_vector(
    output_gpkg: str | Path,
    *,
    orig_mask_tif: Optional[str] = None,         # original mask GeoTIFF (if available)
    cleaned_mask_tif: str,                       # cleaned mask GeoTIFF
    orig_raster_table: str = "ModelDetectedMask",
    cleaned_raster_table: str = "NLCD-cleanedDetectedMask",
    cleaned_vector_layer: str = "NLCD-cleanedMaskOutlineVector",
    target_srs: Optional[str] = None,            # e.g., "EPSG:3338"
    assume_src_srs: Optional[str] = None,        # if input TIFF has no CRS tag
    tile_blocksize: int = 512,
    build_overviews: Iterable[int] = (2, 4, 8, 16),
    eight_connected: bool = True,
    min_area: float = 0.0,
    simplify_tol: float = 0.0,
    overwrite_vector: bool = True,
) -> None:
    """
    Add mask rasters (original optional, cleaned required) and polygonize cleaned mask.

    This writes:
      - orig_raster_table (optional)
      - cleaned_raster_table (required)
      - cleaned_vector_layer (polygonized cleaned mask)

    CRS handling:
      - If target_srs is provided, data is reprojected to it.
      - Otherwise, it attempts to match existing CRS in the GPKG if present.
    """
    out_gpkg = _ensure_gpkg_path(output_gpkg)

    # 1) Original mask raster (optional)
    if orig_mask_tif:
        if _gpkg_has_layer(out_gpkg, orig_raster_table):
            raise ValueError(f"Layer '{orig_raster_table}' already exists in {out_gpkg}")
        add_raster_to_gpkg(
            orig_mask_tif, out_gpkg,
            layer_name=orig_raster_table,
            reproject_to=target_srs,
            match_existing_crs=True,
            assume_src_srs=assume_src_srs,
            tile_blocksize=tile_blocksize,
            build_overviews=build_overviews,
        )
    logging.info(" ADDED orginal mask ")

    # 2) Cleaned mask raster (required)
    if _gpkg_has_layer(out_gpkg, cleaned_raster_table):
        raise ValueError(f"Layer '{cleaned_raster_table}' already exists in {out_gpkg}")
    add_raster_to_gpkg(
        cleaned_mask_tif, out_gpkg,
        layer_name=cleaned_raster_table,
        reproject_to=target_srs,
        match_existing_crs=True,
        assume_src_srs=assume_src_srs,
        tile_blocksize=tile_blocksize,
        build_overviews=build_overviews,
    )
    logging.info(" ADDED filtered mask ")

    # 3) Vectorize cleaned mask (>0) to MULTIPOLYGON
    polygonize_raster_table_to_vector(
        output_gpkg=out_gpkg,
        raster_table=cleaned_raster_table,
        vector_layer=cleaned_vector_layer,
        eight_connected=eight_connected,
        min_area=min_area,
        simplify_tol=simplify_tol,
        overwrite=overwrite_vector,
    )

    logging.info(
        f"✅ Added: {orig_raster_table if orig_mask_tif else '(no orig)'} , "
        f"{cleaned_raster_table} , {cleaned_vector_layer} to {out_gpkg}"
    )


# ----------------------- small unit test tied to main -----------------------

def _unit_test() -> int:
    """
    Minimal local unit test (no external data required):
    - Create a small binary NumPy mask.
    - Write it into a temp GeoPackage as a raster table via add_raster_to_gpkg.
    - Polygonize the raster table into a vector layer.
    - Assert both layers exist and vector has >0 features.

    Returns:
        0 on PASS, non-zero on FAIL.
    """
    try:
        tmp_dir = Path("./_tmp_gpkg_unit_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_gpkg = tmp_dir / "unit_test.gpkg"

        # Clean start
        if out_gpkg.exists():
            out_gpkg.unlink()

        # Simple mask with a filled block
        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[8:24, 10:22] = 1

        # GDAL geotransform tuple: (originX, pixelW, rotX, originY, rotY, pixelHneg)
        gt = (0.0, 1.0, 0.0, 32.0, 0.0, -1.0)

        add_raster_to_gpkg(
            raster_data=arr,
            output_gpkg=out_gpkg,
            layer_name="TestRaster",
            crs="EPSG:3338",
            transform=gt,
            reproject_to=None,
            match_existing_crs=True,
            assume_src_srs=None,
            tile_blocksize=16,
            build_overviews=(),
        )

        polygonize_raster_table_to_vector(
            output_gpkg=out_gpkg,
            raster_table="TestRaster",
            vector_layer="TestVector",
            eight_connected=True,
            min_area=0.0,
            simplify_tol=0.0,
            overwrite=True,
        )

        # Validate: both layers exist, and vector has features
        if not _gpkg_has_layer(str(out_gpkg), "TestRaster"):
            raise RuntimeError("FAIL: raster table 'TestRaster' not found in output gpkg")
        if not _gpkg_has_layer(str(out_gpkg), "TestVector"):
            raise RuntimeError("FAIL: vector layer 'TestVector' not found in output gpkg")

        ds = ogr.Open(str(out_gpkg), update=0)
        lyr = ds.GetLayerByName("TestVector")
        n = lyr.GetFeatureCount() if lyr else 0
        ds = None

        if n <= 0:
            raise RuntimeError("FAIL: polygonize produced 0 features")

        print(f"UNIT TEST PASS: wrote TestRaster + TestVector ({n} features) -> {out_gpkg}")
        return 0

    except Exception as e:
        print(f"UNIT TEST FAIL: {e}")
        return 2


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    raise SystemExit(_unit_test())

if __name__ == "__main__":
    main()
