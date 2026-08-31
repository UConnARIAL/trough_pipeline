from pathlib import Path
import argparse
import numpy as np
import rasterio


IMAGE_EXTS = [".tif", ".tiff"]


def list_images(folder: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=np.float32)

    if weights.shape[0] != 3:
        raise ValueError(f"Expected exactly 3 weights, got {weights}")

    if np.any(weights < 0):
        raise ValueError(f"Weights must be non-negative, got {weights}")

    s = weights.sum()

    if s <= 0:
        raise ValueError(f"Weight sum must be > 0, got {weights}")

    return weights / s


def get_valid_mask(arr_masked):
    """
    True where all first three bands are valid according to rasterio/GDAL masks.

    We do NOT automatically treat 65535 as invalid unless the source declares
    it as nodata or the internal mask marks it invalid.
    """
    masks = np.ma.getmaskarray(arr_masked[:3])
    valid_mask = ~masks.any(axis=0)
    return valid_mask


def weighted_gray_uint16(
    arr_masked,
    weights,
    valid_mask,
    output_nodata=65535,
    stretch_percentiles=None,
):
    """
    Convert first 3 bands to one uint16 grayscale band.

    Default:
      preserve raw weighted values.

    Optional:
      stretch_percentiles=(2, 98) rescales valid pixels to 0..65534.
    """

    if arr_masked.shape[0] < 3:
        raise ValueError(f"Expected at least 3 bands, got {arr_masked.shape[0]}")

    c1 = arr_masked[0].astype("float32")
    c2 = arr_masked[1].astype("float32")
    c3 = arr_masked[2].astype("float32")

    gray = weights[0] * c1 + weights[1] * c2 + weights[2] * c3
    gray = gray.filled(0).astype("float32")

    if stretch_percentiles is not None:
        lo, hi = stretch_percentiles
        vals = gray[valid_mask]

        if vals.size == 0:
            raise ValueError("No valid pixels available for percentile stretch.")

        p_low, p_high = np.percentile(vals, [lo, hi])

        if p_high <= p_low:
            raise ValueError(
                f"Invalid percentile range: p{lo}={p_low}, p{hi}={p_high}"
            )

        gray = (gray - p_low) / (p_high - p_low)
        gray = np.clip(gray, 0, 1)
        gray = gray * 65534.0

    # Reserve 65535 for output nodata.
    gray_uint16 = np.round(gray)
    gray_uint16 = np.clip(gray_uint16, 0, 65534).astype("uint16")
    gray_uint16[~valid_mask] = output_nodata

    return gray_uint16


def convert_one(
    in_path: Path,
    out_path: Path,
    weights,
    output_nodata=65535,
    stretch_percentiles=None,
):
    with rasterio.open(in_path) as src:
        if src.count < 3:
            raise ValueError(f"{in_path.name} has {src.count} bands; expected 3+")

        arr = src.read(masked=True)
        valid_mask = get_valid_mask(arr)

        gray = weighted_gray_uint16(
            arr_masked=arr,
            weights=weights,
            valid_mask=valid_mask,
            output_nodata=output_nodata,
            stretch_percentiles=stretch_percentiles,
        )

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

        # Remove metadata that may not apply to single-band output
        profile.pop("photometric", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(gray, 1)
            dst.write_mask((valid_mask * 255).astype("uint8"))

            wtxt = "_".join([f"{w:.3f}" for w in weights])
            dst.set_band_description(1, f"weighted_gray_uint16_w_{wtxt}")


RGB_INDEXES = [5, 3, 2]  # R, G, B — verify for your data
OUTPUT_NODATA = 65535

def rgb_to_gray_from_8band(src):
    """
    Create panchromatic-like grayscale from true RGB bands in an 8-band image.
    """

    if src.count < max(RGB_INDEXES):
        raise ValueError(
            f"Image has {src.count} bands, but RGB_INDEXES={RGB_INDEXES}"
        )

    arr = src.read(indexes=RGB_INDEXES, masked=True)

    valid_mask = ~np.ma.getmaskarray(arr).any(axis=0)

    r = arr[0].astype("float32")
    g = arr[1].astype("float32")
    b = arr[2].astype("float32")

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.filled(0).astype("float32")

    # Preserve uint16-like range, reserve 65535 for nodata
    gray_uint16 = np.round(gray)
    gray_uint16 = np.clip(gray_uint16, 0, 65534).astype("uint16")
    gray_uint16[~valid_mask] = OUTPUT_NODATA

    return gray_uint16, valid_mask


def main():
    parser = argparse.ArgumentParser(
        description="Create clean uint16 weighted grayscale chips for train/val/test."
    )

    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Root containing train_img, val_img, test_img.",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output root. train_img, val_img, test_img will be created.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process. Default: train val test",
    )

    parser.add_argument(
        "--img-dir-suffix",
        default="_img",
        help="Input/output split folder suffix. Default gives train_img, val_img, test_img.",
    )

    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Weights for C1 C2 C3. Default equal mean.",
    )

    parser.add_argument(
        "--output-nodata",
        type=int,
        default=65535,
        help="Output nodata value. Default 65535.",
    )

    parser.add_argument(
        "--stretch-percentiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Optional percentile stretch to 0..65534, e.g. --stretch-percentiles 2 98. "
            "Default is no stretch, preserving raw weighted uint16-like values."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    weights = normalize_weights(args.weights)

    print("Input root:", args.input_root)
    print("Output root:", args.out_root)
    print("Splits:", args.splits)
    print("Weights normalized:", weights)
    print("Stretch percentiles:", args.stretch_percentiles)

    for split in args.splits:
        in_dir = args.input_root / f"{split}{args.img_dir_suffix}"
        out_dir = args.out_root / f"{split}{args.img_dir_suffix}"

        if not in_dir.exists():
            print(f"\nWARNING: missing input directory, skipping: {in_dir}")
            continue

        image_files = list_images(in_dir)

        print("\n" + "=" * 80)
        print(f"Split: {split}")
        print(f"Input:  {in_dir}")
        print(f"Output: {out_dir}")
        print(f"Found:  {len(image_files)} images")
        print("=" * 80)

        if len(image_files) == 0:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = 0
        n_skip = 0
        n_fail = 0

        for in_path in image_files:
            # Keep exact same filename for mask pairing
            out_path = out_dir / in_path.name

            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            try:
                convert_one(
                    in_path=in_path,
                    out_path=out_path,
                    weights=weights,
                    output_nodata=args.output_nodata,
                    stretch_percentiles=args.stretch_percentiles,
                )
                n_done += 1

            except Exception as e:
                n_fail += 1
                print(f"FAILED: {in_path.name}")
                print(f"  {e}")

        print(f"Done split {split}: written={n_done}, skipped={n_skip}, failed={n_fail}")

    print("\nAll done.")


if __name__ == "__main__":
    main()

"""
python make_gray_uint16_weighted.py \
  --input-root /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr \
  --out-root  /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/tcn_mxr_gray_mean_uint16 \
  --splits train val test \
  --weights 1 1 1 \
  --overwrite


"""