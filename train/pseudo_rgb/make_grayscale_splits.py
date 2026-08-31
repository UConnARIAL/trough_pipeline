from pathlib import Path

import numpy as np
import rasterio


# --------------------------------------------------
# USER PATHS
# --------------------------------------------------

SPLITS = {
    "train": {
        "img_dir": Path("/path/to/train_img"),
        "out_dir": Path("/path/to/gray/train_img"),
    },
    "val": {
        "img_dir": Path("/path/to/val_img"),
        "out_dir": Path("/path/to/gray/val_img"),
    },
    # Optional:
    # "test": {
    #     "img_dir": Path("/path/to/test_img"),
    #     "out_dir": Path("/path/to/gray/test_img"),
    # },
}

IMAGE_EXTS = [".tif", ".tiff"]

# Use 8-bit grayscale output?
OUTPUT_UINT8 = True

# Percentile stretch only used if OUTPUT_UINT8 = True
LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def list_images(folder: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def rgb_to_gray_luminance(arr):
    """
    arr shape: bands, height, width

    Uses standard luminance conversion:
      gray = 0.299 R + 0.587 G + 0.114 B
    """

    if arr.shape[0] >= 3:
        r = arr[0].astype("float32")
        g = arr[1].astype("float32")
        b = arr[2].astype("float32")

        gray = 0.299 * r + 0.587 * g + 0.114 * b

    elif arr.shape[0] == 1:
        gray = arr[0].astype("float32")

    else:
        raise ValueError(f"Unexpected band count: {arr.shape[0]}")

    return gray


def percentile_to_uint8(gray, valid_mask, lower=2, upper=98, nodata_value=0):
    """
    Percentile stretch grayscale image to uint8.
    """

    if valid_mask.sum() == 0:
        raise ValueError("No valid pixels found.")

    p_low, p_high = np.percentile(gray[valid_mask], [lower, upper])

    if p_high <= p_low:
        raise ValueError(f"Invalid percentile range: {p_low}, {p_high}")

    scaled = (gray - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0, 1)

    out = (scaled * 255).astype("uint8")
    out[~valid_mask] = nodata_value

    return out


def convert_rgb_to_gray(in_path: Path, out_path: Path):
    with rasterio.open(in_path) as src:
        arr = src.read()
        gray = rgb_to_gray_luminance(arr)

        nodata = src.nodata

        valid_mask = np.isfinite(gray)

        if nodata is not None:
            if src.count >= 3:
                valid_mask &= np.all(arr[:3] != nodata, axis=0)
            else:
                valid_mask &= arr[0] != nodata

        profile = src.profile.copy()

        if OUTPUT_UINT8:
            gray_out = percentile_to_uint8(
                gray,
                valid_mask,
                lower=LOWER_PERCENTILE,
                upper=UPPER_PERCENTILE,
                nodata_value=0,
            )

            profile.update(
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=0,
                compress="lzw",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

        else:
            gray_out = gray.astype("float32")
            gray_out[~valid_mask] = -9999

            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=-9999,
                compress="lzw",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

        # Avoid carrying incompatible source metadata
        profile.pop("photometric", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(gray_out, 1)
            dst.set_band_description(1, "grayscale luminance")

            # Preserve valid data mask
            dst.write_mask((valid_mask * 255).astype("uint8"))


def main():
    for split_name, paths in SPLITS.items():
        img_dir = paths["img_dir"]
        out_dir = paths["out_dir"]

        out_dir.mkdir(parents=True, exist_ok=True)

        image_files = list_images(img_dir)

        print(f"\nSplit: {split_name}")
        print(f"Input: {img_dir}")
        print(f"Output: {out_dir}")
        print(f"Found {len(image_files)} images")

        for in_path in image_files:
            out_path = out_dir / f"{in_path.stem}_gray.tif"

            print(f"  {in_path.name} -> {out_path.name}")

            try:
                convert_rgb_to_gray(in_path, out_path)
            except Exception as e:
                print(f"  FAILED: {in_path.name}")
                print(f"    {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()


"""
USAGE


"""