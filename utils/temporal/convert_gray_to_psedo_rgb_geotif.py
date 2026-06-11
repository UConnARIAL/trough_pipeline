from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from skimage import exposure
from skimage.filters import sobel

def scale_to_uint8_percentile(
    arr,
    valid_mask,
    lower_percentile=2,
    upper_percentile=98,
    output_nodata=0,
):
    """
    Percentile scale a single-band array to uint8.
    """
    p_low, p_high = np.percentile(
        arr[valid_mask],
        [lower_percentile, upper_percentile]
    )

    if p_high <= p_low:
        raise ValueError(
            f"Invalid percentile range: p_low={p_low}, p_high={p_high}"
        )

    scaled = (arr - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0, 1)

    out = (scaled * 255).astype("uint8")
    out[~valid_mask] = output_nodata

    return out


def normalize_to_01(arr_uint8, valid_mask):
    """
    Convert uint8 image to float image in [0, 1].
    Invalid pixels are set to 0.
    """
    arr01 = arr_uint8.astype("float32") / 255.0
    arr01[~valid_mask] = 0.0
    return arr01


def single_band_to_pseudo_rgb_v2(
    in_path,
    out_path,
    lower_percentile=2,
    upper_percentile=98,
    clahe_clip_limit=0.01,
    output_nodata=0,
):
    """
    Convert a single-band georeferenced image to a 3-band pseudo-RGB GeoTIFF.

    Output bands:
      Band 1 / Red   = percentile-stretched grayscale
      Band 2 / Green = CLAHE-enhanced grayscale
      Band 3 / Blue  = Sobel edge/detail image

    The output preserves CRS, transform, width, height, and geolocation.
    """

    in_path = Path(in_path)
    out_path = Path(out_path)

    with rasterio.open(in_path) as src:
        band = src.read(1).astype("float32")

        nodata = src.nodata

        if nodata is not None:
            valid_mask = (band != nodata) & np.isfinite(band)
        else:
            valid_mask = np.isfinite(band)

        if valid_mask.sum() == 0:
            raise ValueError(f"No valid pixels found in {in_path}")

        # -------------------------------
        # Band 1: Percentile-stretched gray
        # -------------------------------
        red = scale_to_uint8_percentile(
            band,
            valid_mask,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            output_nodata=output_nodata,
        )

        red01 = normalize_to_01(red, valid_mask)

        # -------------------------------
        # Band 2: CLAHE local contrast
        # -------------------------------
        clahe01 = exposure.equalize_adapthist(
            red01,
            clip_limit=clahe_clip_limit
        )

        green = (np.clip(clahe01, 0, 1) * 255).astype("uint8")
        green[~valid_mask] = output_nodata

        # -------------------------------
        # Band 3: Sobel edge/detail channel
        # -------------------------------
        sobel01 = sobel(red01)

        # Scale Sobel result to uint8 using valid pixels only
        if np.any(sobel01[valid_mask] > 0):
            sobel_low, sobel_high = np.percentile(
                sobel01[valid_mask],
                [2, 98]
            )

            if sobel_high > sobel_low:
                sobel_scaled = (sobel01 - sobel_low) / (sobel_high - sobel_low)
            else:
                sobel_scaled = sobel01
        else:
            sobel_scaled = sobel01

        sobel_scaled = np.clip(sobel_scaled, 0, 1)

        blue = (sobel_scaled * 255).astype("uint8")
        blue[~valid_mask] = output_nodata

        # Stack as 3-band image
        rgb = np.stack([red, green, blue], axis=0)

        # Preserve geospatial metadata
        profile = src.profile.copy()

        # Update for 3-band uint8 GeoTIFF
        profile.update(
            driver="GTiff",
            count=3,
            dtype="uint8",
            nodata=output_nodata,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        # Some IMG-specific metadata may not apply cleanly to GeoTIFF
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("photometric", None)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)

            dst.set_band_description(1, "Percentile stretched grayscale")
            dst.set_band_description(2, "CLAHE enhanced grayscale")
            dst.set_band_description(3, "Sobel edge/detail channel")

            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
            )

            # Write internal valid-data mask
            dst.write_mask((valid_mask * 255).astype("uint8"))

    print(f"Saved pseudo-RGB GeoTIFF: {out_path}")


# Example usage
"""
single_band_to_pseudo_rgb_v2(
    in_path="/work/09208/asperera/ls6/DATA/Historical/sample//1948August08_clip_25cm.img",
    out_path="/work/09208/asperera/ls6/DATA/Historical/geotifs/1948August08_clip_25cm_psedo.tif",
    lower_percentile=2,
    upper_percentile=98,
    clahe_clip_limit=0.005,
)
"""
from pathlib import Path

input_dir = Path("/work/09208/asperera/ls6/DATA/historical/sample/")
output_dir = Path("/work/09208/asperera/ls6/DATA/historical/geotifs")

output_dir.mkdir(parents=True, exist_ok=True)

img_files = sorted(input_dir.glob("*.img"))

print(f"Found {len(img_files)} .img files")

for in_path in img_files:
    out_path = output_dir / f"{in_path.stem}_pseudo_rgb.tif"

    try:
        print(f"Converting: {in_path.name} -> {out_path.name}")

        single_band_to_pseudo_rgb_v2(
            in_path=in_path,
            out_path=out_path,
            lower_percentile=2,
            upper_percentile=98,
            clahe_clip_limit=0.01,
        )
    except Exception as e:
        print(f"FAILED: {in_path.name}")
        print(f"  Error: {e}")
print("Done.")


