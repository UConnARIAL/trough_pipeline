#########################################
# Code to convert RGB to gray for testing
#########################################
from pathlib import Path
import argparse

import numpy as np
import rasterio


IMAGE_EXTS = [".tif", ".tiff"]


GRAY_MODES = {
    "mean": "Mean of C1/C2/C3",
    "c1_band8": "Channel 1 only; original band 8",
    "c2_band6": "Channel 2 only; original band 6",
    "c3_band1": "Channel 3 only; original band 1",
}


def list_images(folder: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def get_valid_mask(src, arr_masked):
    """
    Build valid mask using GDAL/rasterio mask handling.

    arr_masked: masked array, shape = bands, H, W
    """

    # True where all first 3 bands are valid according to nodata/internal mask
    valid_mask = ~np.ma.getmaskarray(arr_masked[:3]).any(axis=0)

    return valid_mask


def composite_to_gray(arr_masked, gray_mode: str):
    """
    Convert 3-band training composite to grayscale.

    Input channels are not true natural RGB:
      C1 = original band 8
      C2 = original band 6
      C3 = original band 1

    Output is float32 before casting back to uint16.
    """

    if arr_masked.shape[0] < 3:
        raise ValueError(f"Expected at least 3 bands, got {arr_masked.shape[0]}")

    c1 = arr_masked[0].astype("float32")
    c2 = arr_masked[1].astype("float32")
    c3 = arr_masked[2].astype("float32")

    if gray_mode == "mean":
        gray = (c1 + c2 + c3) / 3.0

    elif gray_mode == "c1_band8":
        gray = c1

    elif gray_mode == "c2_band6":
        gray = c2

    elif gray_mode == "c3_band1":
        gray = c3

    else:
        raise ValueError(f"Unknown gray_mode: {gray_mode}")

    return gray


def convert_one_image(
    in_path: Path,
    out_path: Path,
    gray_mode: str,
    output_nodata: int = 65535,
):
    with rasterio.open(in_path) as src:
        if src.count < 3:
            raise ValueError(f"{in_path.name} has {src.count} bands; expected 3+")

        if not all(dt == "uint16" for dt in src.dtypes[:3]):
            print(f"WARNING: {in_path.name} dtypes are {src.dtypes}; output will still be uint16")

        # masked=True respects declared nodata and internal masks if present
        arr = src.read(masked=True)

        valid_mask = get_valid_mask(src, arr)

        gray_masked = composite_to_gray(arr, gray_mode=gray_mode)

        gray_float = gray_masked.filled(0).astype("float32")

        # Cast back to uint16, preserving original-like range
        gray_uint16 = np.round(gray_float)
        gray_uint16 = np.clip(gray_uint16, 0, 65535).astype("uint16")

        # Set invalid pixels to output nodata
        gray_uint16[~valid_mask] = output_nodata

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint16",
            nodata=output_nodata,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        # Avoid carrying over 3-band-specific metadata
        profile.pop("photometric", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(gray_uint16, 1)
            dst.set_band_description(1, f"grayscale_{gray_mode}")

            # Preserve valid-data mask
            dst.write_mask((valid_mask * 255).astype("uint8"))


def main():
    parser = argparse.ArgumentParser(
        description="Create 16-bit grayscale variants from 3-band uint16 training images."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing 3-band uint16 training images.",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output root folder. Variant subfolders will be created inside this.",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mean", "c1_band8", "c2_band6", "c3_band1"],
        choices=list(GRAY_MODES.keys()),
        help="Grayscale variants to create.",
    )

    parser.add_argument(
        "--output-nodata",
        type=int,
        default=65535,
        help="Output nodata value for uint16 grayscale images.",
    )

    parser.add_argument(
        "--suffix",
        default="_gray",
        help="Suffix added to output filenames before .tif.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    image_files = list_images(args.input_dir)

    print(f"Input folder: {args.input_dir}")
    print(f"Output root:  {args.out_root}")
    print(f"Found {len(image_files)} images")
    print(f"Modes: {args.modes}")

    if len(image_files) == 0:
        raise RuntimeError(f"No .tif/.tiff files found in {args.input_dir}")

    for gray_mode in args.modes:
        out_dir = args.out_root / f"gray_{gray_mode}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Creating mode: {gray_mode} - {GRAY_MODES[gray_mode]}")
        print(f"Output dir: {out_dir}")
        print("=" * 80)

        for in_path in image_files:
            out_path = out_dir / f"{in_path.stem}{args.suffix}.tif"

            if out_path.exists() and not args.overwrite:
                print(f"  exists, skipping: {out_path.name}")
                continue

            try:
                print(f"  {in_path.name} -> {out_path.name}")
                convert_one_image(
                    in_path=in_path,
                    out_path=out_path,
                    gray_mode=gray_mode,
                    output_nodata=args.output_nodata,
                )

            except Exception as e:
                print(f"  FAILED: {in_path.name}")
                print(f"    {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
USAGE
python rgb_to_gray.py \
  --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr/train_img \
  --out-root  /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr_gray/train_img
  
for split in train val test; do
  python rgb_to_gray.py \
    --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr/${split}_img \
    --out-root  /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr_gray/${split}_img \
    --modes mean c1_band8 c2_band6 c3_band1 \
    --overwrite
done

"""


