from pathlib import Path
import argparse
import numpy as np
import rasterio


IMAGE_EXTS = [".tif", ".tiff", ".img"]
DEFAULT_NODATA = 65535


def list_images(folder: Path, recursive: bool = False):
    files = []
    globber = folder.rglob if recursive else folder.glob

    for ext in IMAGE_EXTS:
        files.extend(globber(f"*{ext}"))
        files.extend(globber(f"*{ext.upper()}"))

    return sorted(set(files))


def normalize_weights(weights, normalize=True):
    weights = np.asarray(weights, dtype=np.float32)

    if len(weights) != 8:
        raise ValueError(f"Expected 8 weights, one per band. Got: {weights}")

    if np.any(weights < 0):
        raise ValueError(f"Weights must be non-negative. Got: {weights}")

    if weights.sum() <= 0:
        raise ValueError(f"At least one weight must be > 0. Got: {weights}")

    if normalize:
        weights = weights / weights.sum()

    return weights


def sample_reference_distribution(
    ref_dir: Path,
    percentiles=(2, 98),
    recursive=False,
    max_pixels_per_image=200_000,
    ignore_zero=False,
    random_seed=42,
):
    """
    Compute target percentile range from real historical/input grayscale images.

    This is used to match synthetic gray values to the reference gray distribution.
    """
    ref_files = list_images(ref_dir, recursive=recursive)

    if len(ref_files) == 0:
        raise RuntimeError(f"No reference images found in: {ref_dir}")

    rng = np.random.default_rng(random_seed)
    samples = []

    print(f"Computing reference distribution from {len(ref_files)} images...")

    for p in ref_files:
        try:
            with rasterio.open(p) as src:
                arr = src.read(1, masked=True)
                vals = arr.compressed().astype("float32")

                if ignore_zero:
                    vals = vals[vals != 0]

                if vals.size == 0:
                    continue

                if vals.size > max_pixels_per_image:
                    idx = rng.choice(vals.size, size=max_pixels_per_image, replace=False)
                    vals = vals[idx]

                samples.append(vals)

        except Exception as e:
            print(f"WARNING: failed reading reference {p}")
            print(f"  {e}")

    if len(samples) == 0:
        raise RuntimeError("No valid reference pixels found.")

    all_vals = np.concatenate(samples)

    p_low, p_high = np.percentile(all_vals, percentiles)

    if p_high <= p_low:
        raise ValueError(f"Bad reference percentiles: {p_low}, {p_high}")

    print("Reference distribution:")
    print(f"  percentiles: {percentiles}")
    print(f"  p_low:       {p_low}")
    print(f"  p_high:      {p_high}")
    print(f"  sample n:    {all_vals.size}")

    return float(p_low), float(p_high)


def valid_mask_from_active_bands(arr_masked, active_band_mask):
    """
    arr_masked shape = 8, H, W
    active_band_mask shape = 8

    Valid where all active bands are valid according to rasterio/GDAL masks.
    """
    active_arr = arr_masked[active_band_mask]

    masks = np.ma.getmaskarray(active_arr)

    if masks.ndim == 2:
        valid_mask = ~masks
    else:
        valid_mask = ~masks.any(axis=0)

    return valid_mask


def weighted_gray_from_8band(src, weights):
    """
    Create weighted grayscale from 8-band image.

    weights correspond to bands 1..8.
    """
    if src.count < 8:
        raise ValueError(f"{src.name} has {src.count} bands; expected at least 8.")

    arr = src.read(indexes=list(range(1, 9)), masked=True)  # 8, H, W

    active = weights > 0
    valid_mask = valid_mask_from_active_bands(arr, active)

    gray = np.zeros((src.height, src.width), dtype=np.float32)

    for i in range(8):
        if weights[i] == 0:
            continue

        band = arr[i].astype("float32").filled(0)
        gray += weights[i] * band

    return gray, valid_mask


def match_gray_to_reference_percentiles(
    gray,
    valid_mask,
    ref_low,
    ref_high,
    source_percentiles=(2, 98),
):
    """
    Match each synthetic gray image to the target/reference percentile range.

    This keeps the relative contrast of each chip but maps its robust low/high
    range to the historical/reference low/high range.
    """
    vals = gray[valid_mask]

    if vals.size == 0:
        raise ValueError("No valid pixels in synthetic gray image.")

    src_low, src_high = np.percentile(vals, source_percentiles)

    if src_high <= src_low:
        raise ValueError(f"Bad source percentiles: {src_low}, {src_high}")

    gray_norm = (gray - src_low) / (src_high - src_low)
    gray_norm = np.clip(gray_norm, 0, 1)

    gray_matched = ref_low + gray_norm * (ref_high - ref_low)

    return gray_matched


def gray_to_uint16(gray, valid_mask, output_nodata=DEFAULT_NODATA):
    """
    Convert gray float image to uint16.

    Reserves output_nodata, default 65535, for invalid pixels.
    """
    if output_nodata is None:
        gray_uint16 = np.round(gray)
        gray_uint16 = np.clip(gray_uint16, 0, 65535).astype("uint16")
    else:
        max_valid = output_nodata - 1
        gray_uint16 = np.round(gray)
        gray_uint16 = np.clip(gray_uint16, 0, max_valid).astype("uint16")
        gray_uint16[~valid_mask] = output_nodata

    return gray_uint16


def clean_profile(src, count, output_nodata):
    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=count,
        dtype="uint16",
        nodata=output_nodata,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    profile.pop("photometric", None)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    return profile


def write_gray_1band(src, out_path, gray_uint16, valid_mask, output_nodata, desc):
    profile = clean_profile(src, count=1, output_nodata=output_nodata)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(gray_uint16, 1)
        dst.write_mask((valid_mask * 255).astype("uint8"))
        dst.set_band_description(1, desc)


def write_gray_copy3(src, out_path, gray_uint16, valid_mask, output_nodata, desc):
    profile = clean_profile(src, count=3, output_nodata=output_nodata)

    rgb = np.stack([gray_uint16, gray_uint16, gray_uint16], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb)
        dst.write_mask((valid_mask * 255).astype("uint8"))
        dst.set_band_description(1, f"{desc}_copy1")
        dst.set_band_description(2, f"{desc}_copy2")
        dst.set_band_description(3, f"{desc}_copy3")


def convert_one(
    in_path,
    out_gray_path,
    out_copy3_path,
    weights,
    output_nodata,
    ref_low=None,
    ref_high=None,
    source_percentiles=(2, 98),
):
    with rasterio.open(in_path) as src:
        gray, valid_mask = weighted_gray_from_8band(src, weights)

        if ref_low is not None and ref_high is not None:
            gray = match_gray_to_reference_percentiles(
                gray=gray,
                valid_mask=valid_mask,
                ref_low=ref_low,
                ref_high=ref_high,
                source_percentiles=source_percentiles,
            )

        gray_uint16 = gray_to_uint16(
            gray=gray,
            valid_mask=valid_mask,
            output_nodata=output_nodata,
        )

        weight_desc = "_".join([f"{w:.4f}" for w in weights])
        desc = f"weighted_gray_b1_to_b8_{weight_desc}"

        if ref_low is not None:
            desc += "_matched"

        if out_gray_path is not None:
            write_gray_1band(
                src=src,
                out_path=out_gray_path,
                gray_uint16=gray_uint16,
                valid_mask=valid_mask,
                output_nodata=output_nodata,
                desc=desc,
            )

        if out_copy3_path is not None:
            write_gray_copy3(
                src=src,
                out_path=out_copy3_path,
                gray_uint16=gray_uint16,
                valid_mask=valid_mask,
                output_nodata=output_nodata,
                desc=desc,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create weighted gray images from 8-band Maxar chips, optionally "
            "matched to a reference grayscale distribution."
        )
    )

    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Input directory containing 8-band image patches.")

    parser.add_argument("--out-gray-dir", type=Path, default=None,
                        help="Output directory for 1-band gray images.")

    parser.add_argument("--out-copy3-dir", type=Path, default=None,
                        help="Output directory for 3-band gray-copy images.")

    parser.add_argument("--weights", nargs=8, type=float, required=True,
                        metavar=("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"),
                        help="Eight weights corresponding to source bands 1..8.")

    parser.add_argument("--no-normalize", action="store_true",
                        help="Do not normalize weights to sum to 1.")

    parser.add_argument("--match-ref-dir", type=Path, default=None,
                        help="Optional reference grayscale image directory to match distribution.")

    parser.add_argument("--match-percentiles", nargs=2, type=float, default=[2, 98],
                        metavar=("LOW", "HIGH"),
                        help="Reference percentiles to match. Default: 2 98.")

    parser.add_argument("--source-percentiles", nargs=2, type=float, default=[2, 98],
                        metavar=("LOW", "HIGH"),
                        help="Source image percentiles used before matching. Default: 2 98.")

    parser.add_argument("--ref-recursive", action="store_true",
                        help="Search reference directory recursively.")

    parser.add_argument("--input-recursive", action="store_true",
                        help="Search input directory recursively.")

    parser.add_argument("--ref-ignore-zero", action="store_true",
                        help="Ignore zero values when computing reference distribution.")

    parser.add_argument("--max-ref-pixels-per-image", type=int, default=200_000,
                        help="Max reference pixels sampled per image.")

    parser.add_argument("--output-nodata", type=int, default=DEFAULT_NODATA,
                        help="Output nodata value. Default: 65535.")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs.")

    args = parser.parse_args()

    if args.out_gray_dir is None and args.out_copy3_dir is None:
        raise ValueError("Provide at least one of --out-gray-dir or --out-copy3-dir.")

    weights = normalize_weights(args.weights, normalize=not args.no_normalize)

    ref_low = None
    ref_high = None

    if args.match_ref_dir is not None:
        ref_low, ref_high = sample_reference_distribution(
            ref_dir=args.match_ref_dir,
            percentiles=tuple(args.match_percentiles),
            recursive=args.ref_recursive,
            max_pixels_per_image=args.max_ref_pixels_per_image,
            ignore_zero=args.ref_ignore_zero,
        )

    image_files = list_images(args.input_dir, recursive=args.input_recursive)

    print("\nConfiguration")
    print("=" * 80)
    print(f"Input dir:              {args.input_dir}")
    print(f"Found input images:     {len(image_files)}")
    print(f"Output gray dir:        {args.out_gray_dir}")
    print(f"Output copy3 dir:       {args.out_copy3_dir}")
    print(f"Input weights:          {args.weights}")
    print(f"Weights used:           {weights}")
    print(f"Normalize weights:      {not args.no_normalize}")
    print(f"Match ref dir:          {args.match_ref_dir}")
    print(f"Reference low/high:     {ref_low}, {ref_high}")
    print(f"Source percentiles:     {args.source_percentiles}")
    print(f"Output nodata:          {args.output_nodata}")
    print("=" * 80)

    if len(image_files) == 0:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    n_done = 0
    n_skip = 0
    n_fail = 0

    for in_path in image_files:
        out_gray_path = args.out_gray_dir / in_path.name if args.out_gray_dir else None
        out_copy3_path = args.out_copy3_dir / in_path.name if args.out_copy3_dir else None

        outputs_exist = True

        if out_gray_path is not None and not out_gray_path.exists():
            outputs_exist = False

        if out_copy3_path is not None and not out_copy3_path.exists():
            outputs_exist = False

        if outputs_exist and not args.overwrite:
            n_skip += 1
            continue

        try:
            convert_one(
                in_path=in_path,
                out_gray_path=out_gray_path,
                out_copy3_path=out_copy3_path,
                weights=weights,
                output_nodata=args.output_nodata,
                ref_low=ref_low,
                ref_high=ref_high,
                source_percentiles=tuple(args.source_percentiles),
            )
            n_done += 1

        except Exception as e:
            n_fail += 1
            print(f"FAILED: {in_path}")
            print(f"  {e}")

    print("\nDone.")
    print(f"Written: {n_done}")
    print(f"Skipped: {n_skip}")
    print(f"Failed:  {n_fail}")


if __name__ == "__main__":
    main()



"""
python make_weighted_gray_matched_from_8band.py \
  --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/images \
  --out-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_train/images \
  --out-copy3-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_x3_train/images \
  --weights 0 0.114 0.587 0 0.299 0 0 0 \
  --match-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/Drew_Point_Ice_Wedge_Time_Series/ \
  --match-percentiles 2 98 \
  --source-percentiles 2 98 \
  --overwrite
  
python make_weighted_gray_matched_from_8band.py \
  --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/images \
  --out-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/b2_7gray_train/images \
  --out-copy3-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_x3_train/images \
  --weights 0 0.22 0.26 0.15 0.22 0.15 0.05 0 \
  --match-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/Drew_Point_Ice_Wedge_Time_Series/ \
  --match-percentiles 2 98 \
  --source-percentiles 2 98 \
  --overwrite

"""