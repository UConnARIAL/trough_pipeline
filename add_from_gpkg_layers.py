# --- imports & small helpers -----------------------------------------------------
import os, re, sqlite3, subprocess
from pathlib import Path
from typing import Iterable, Optional, Dict
from collections import Counter
from osgeo import gdal, osr  # used only for CRS checks in raster branch


from pathlib import Path
import re

import logging
def tiff_to_gpkg_path(tif_path, old_gpkg_root="/scratch2/projects/PDG_shared/AlaskaTundraMosaicTroughGeoPkg",
                      ensure_parent=False, postfix="_trough_layers.gpkg"):
    """
    Build the GPKG path for a given TIFF path.

    Example:
      tif:  ../masked/58_16/ArcticMosaic_58_16_1_1_mask.tif
      root: /old/gpkg/root
      -->   /old/gpkg/root/58_16/ArcticMosaic_58_16_1_1_trough_layers.gpkg

    Args:
        tif_path (str|Path): Path to the source .tif
        old_gpkg_root (str|Path): Root directory for GPKGs
        ensure_parent (bool): If True, create parent directory for the gpkg path
        postfix (str): Filename suffix to append after removing trailing '_mask'
    Returns:
        Path: Target GPKG path
    """
    tif_path = Path(tif_path)
    root = Path(old_gpkg_root)
    # Last parent (e.g., "58_16")
    subdir = tif_path.parent.name
    # Base name without .tif (or .tiff)
    base = tif_path.stem  # e.g., "ArcticMosaic_58_16_1_1_mask"
    # Remove a trailing "_mask" (only if it’s at the end)
    base_no_mask = re.sub(r"_mask$", "", base)
    # Build new filename
    new_name = f"{base_no_mask}{postfix}"  # e.g., "..._trough_layers.gpkg"
    gpkg_path = root / subdir / new_name
    if ensure_parent:
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    return gpkg_path



def _run(args: list[str]) -> None:
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(args)}") from e

def _ensure_gpkg_path(p: str | Path) -> str:
    p = Path(p)
    if p.suffix.lower() != ".gpkg":
        p = p.with_suffix(".gpkg")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def _layer_types(gpkg_path: str) -> Dict[str, str]:
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(gpkg_path)
    with sqlite3.connect(gpkg_path) as con:
        rows = con.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall()
    return {name: dtype for name, dtype in rows}

def _has_layer(gpkg_path: str, layer: str) -> bool:
    if not os.path.exists(gpkg_path):
        return False
    with sqlite3.connect(gpkg_path) as con:
        row = con.execute("SELECT 1 FROM gpkg_contents WHERE table_name=?", (layer,)).fetchone()
    return row is not None

def _srs_auth_string(con: sqlite3.Connection, srs_id: int) -> Optional[str]:
    row = con.execute(
        "SELECT organization, organization_coordsys_id, definition "
        "FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs_id,)
    ).fetchone()
    if not row:
        return None
    org, code, wkt = row
    if org and code is not None:
        return f"{org.upper()}:{code}"  # e.g., EPSG:3338
    return wkt  # fallback

def _gpkg_preferred_srs(gpkg_path: str) -> Optional[str]:
    """Return 'EPSG:xxxx' if found (preferred), else WKT, else None."""
    if not os.path.exists(gpkg_path):
        return None
    with sqlite3.connect(gpkg_path) as con:
        srs_ids = []
        srs_ids += [sid for (sid,) in con.execute("SELECT DISTINCT srs_id FROM gpkg_geometry_columns WHERE srs_id IS NOT NULL")]
        srs_ids += [sid for (sid,) in con.execute("SELECT DISTINCT srs_id FROM gpkg_tile_matrix_set WHERE srs_id IS NOT NULL")]
        if not srs_ids:
            return None
        srs_id = Counter(srs_ids).most_common(1)[0][0]
        return _srs_auth_string(con, srs_id)

# --- CRS / raster utilities (used in raster branch) -----------------------------

def _wkt_from_dataset(ds: gdal.Dataset) -> str:
    try:
        return ds.GetProjection() or ""
    except Exception:
        return ""

def _same_crs_wkt_vs_auth(src_wkt: str, target_auth: str) -> bool:
    """Compare dataset WKT with a target 'EPSG:xxxx' using OSR IsSame()."""
    if not target_auth:
        return False
    s = osr.SpatialReference()
    t = osr.SpatialReference()
    if src_wkt:
        s.ImportFromWkt(src_wkt)
    t.SetFromUserInput(target_auth)
    try:
        return s.IsSame(t) == 1
    except Exception:
        return False

def _linear_or_angular_units_name(wkt: str) -> str:
    s = osr.SpatialReference()
    try:
        s.ImportFromWkt(wkt)
    except Exception:
        return ""
    # prefer linear; fall back to angular for geographic CRS
    name = s.GetLinearUnitsName() or s.GetAngularUnitsName() or ""
    return name.lower()

def _both_meter_units(src_wkt: str, target_auth: str) -> bool:
    if not target_auth or not src_wkt:
        return False
    t = osr.SpatialReference(); t.SetFromUserInput(target_auth)
    dst_units = (t.GetLinearUnitsName() or t.GetAngularUnitsName() or "").lower()
    src_units = _linear_or_angular_units_name(src_wkt)
    return src_units == "metre" and dst_units == "metre"

def _source_pixel_size(ds: gdal.Dataset) -> tuple[Optional[float], Optional[float]]:
    gt = ds.GetGeoTransform()
    if not gt:
        return (None, None)
    xres = abs(gt[1]) if gt[1] else None
    yres = abs(gt[5]) if gt[5] else None
    return (xres, yres)

# --- main: copy_layer with CRS-safe raster handling -----------------------------

def copy_layer(
    out_gpkg: str | Path,
    old_gpkg_path: str | Path,
    src_layer: str,
    dst_layer: str,
    *,
    # Raster options
    raster_tile_format: str = "PNG",       # use "PNG8" for masks to keep single-band
    raster_blocksize: int = 512,
    build_overviews: bool = True,
    overview_levels: Optional[Iterable[int]] = (2, 4, 8, 16),
    overview_resampling: str = "nearest",  # "nearest" for masks, "average" for imagery
    enforce_single_band: bool = False,     # True for masks
    nodata: Optional[float] = None,        # e.g., 0 for masks
    preserve_resolution_if_units_match: bool = True,
    # Vector options
    vector_sql_where: Optional[str] = None,
    # CRS policy
    reproject_to: Optional[str] = None,    # e.g., "EPSG:3338" (recommended)
    match_existing_crs: bool = True        # if True, match existing out_gpkg CRS when reproject_to is None
) -> None:
    """
    Copy a layer (vector or raster tile table) from an existing GPKG into another GPKG.

    - If reproject_to is set, output is reprojected to that CRS.
    - If reproject_to is None and out_gpkg exists + match_existing_crs=True, will align to that CRS.
    - Raster copy can be kept single-band (PNG8) for masks by setting enforce_single_band=True and nodata=0.
    - Preserves pixel size on warp only when both CRSs are meter-based (e.g., 3413 <-> 3338).
    """
    old_gpkg_path = str(old_gpkg_path)
    out_gpkg = _ensure_gpkg_path(out_gpkg)

    if not os.path.exists(old_gpkg_path):
        raise FileNotFoundError(f"Source GPKG not found: {old_gpkg_path}")

    src_types = _layer_types(old_gpkg_path)
    if src_layer not in src_types:
        raise ValueError(f"Layer '{src_layer}' not found in {old_gpkg_path}")

    if _has_layer(out_gpkg, dst_layer):
        raise ValueError(f"Destination layer '{dst_layer}' already exists in {out_gpkg}")

    dtype = src_types[src_layer]

    # Decide target SRS to enforce
    target_srs = reproject_to
    if target_srs is None and match_existing_crs and os.path.exists(out_gpkg):
        target_srs = _gpkg_preferred_srs(out_gpkg)  # may be 'EPSG:xxxx' or WKT

    # ---------------- VECTORS ----------------
    if dtype == "features":
        args = ["ogr2ogr", "-f", "GPKG"]
        if os.path.exists(out_gpkg):
            args += ["-update"]
        args += [out_gpkg, old_gpkg_path, "-nln", dst_layer]

        # optional where
        sql = f'SELECT * FROM "{src_layer}"' + (f" WHERE {vector_sql_where}" if vector_sql_where else "")
        args += ["-sql", sql]

        # keep geometry homogenous
        args += ["-nlt", "PROMOTE_TO_MULTI"]

        # reproject if requested/available
        if target_srs:
            args += ["-t_srs", target_srs]

        _run(args)

    # ---------------- RASTERS (tile tables) ----------------
    elif dtype == "tiles":
        # open source subdataset to interrogate CRS, bands, pixel size
        src_subds = f"GPKG:{old_gpkg_path}:{src_layer}"
        src_ds = gdal.Open(src_subds, gdal.GA_ReadOnly)
        if src_ds is None:
            raise RuntimeError(f"Cannot open source raster table: {src_subds}")

        src_wkt = _wkt_from_dataset(src_ds)
        need_warp = False
        if target_srs:
            # Prefer OSR IsSame() for correctness
            if isinstance(target_srs, str) and target_srs.upper().startswith("EPSG:"):
                need_warp = not _same_crs_wkt_vs_auth(src_wkt, target_srs)
            else:
                # if target_srs is a WKT string; compare strictly
                s = osr.SpatialReference(); s.ImportFromWkt(src_wkt or "")
                t = osr.SpatialReference(); t.ImportFromWkt(str(target_srs))
                need_warp = (s.IsSame(t) != 1)

        # Common creation options
        co = [
            f"RASTER_TABLE={dst_layer}",
            f"TILE_FORMAT={raster_tile_format}",
            f"BLOCKXSIZE={int(raster_blocksize)}",
            f"BLOCKYSIZE={int(raster_blocksize)}",
        ]
        if os.path.exists(out_gpkg):
            co.append("APPEND_SUBDATASET=YES")

        # Build CLI args

        if need_warp:
            args = [
                "gdalwarp",
                src_subds,
                out_gpkg,
                "-of", "GPKG",
                "-t_srs", target_srs,
                "-r", overview_resampling,
                "-multi",
            ]
            # Preserve resolution only when both CRSs are metre-based
            if preserve_resolution_if_units_match and _both_meter_units(src_wkt, str(target_srs)):
                xres, yres = _source_pixel_size(src_ds)
                if xres and yres:
                    args += ["-tr", str(xres), str(yres)]

            # Mask-friendly options
            if enforce_single_band and src_ds.RasterCount >= 1:
                args += ["-b", "1", "-ot", "Byte"]
            if nodata is not None:
                args += ["-dstnodata", str(nodata)]

            # creation options
            for opt in co:
                args += ["-co", opt]

            _run(args)

        else:
            # Same CRS → fast translate
            args = [
                "gdal_translate",
                src_subds,
                out_gpkg,
                "-of", "GPKG",
            ]
            if enforce_single_band and src_ds.RasterCount >= 1:
                args += ["-b", "1", "-ot", "Byte"]
            if nodata is not None:
                args += ["-a_nodata", str(nodata)]
            for opt in co:
                args += ["-co", opt]

            _run(args)

        # Build overviews on that raster table
        if build_overviews and overview_levels:
            dst_subds = f"GPKG:{out_gpkg}:{dst_layer}"
            _run(["gdaladdo", "-r", overview_resampling, dst_subds] + [str(l) for l in overview_levels])

    else:
        raise ValueError(f"Unsupported data_type '{dtype}' for layer '{src_layer}'")
    logging.info(f"✅ Copied {dtype} layer '{src_layer}' → '{dst_layer}' into {out_gpkg} "f"{'(CRS enforced: ' + str(target_srs) + ')' if target_srs else '(source CRS)'}")
