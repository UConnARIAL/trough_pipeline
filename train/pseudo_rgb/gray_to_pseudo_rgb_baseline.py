#!/usr/bin/env python3

"""
gray_pseudo_rgb_baseline.py

Convert single-band grayscale / historical imagery into a 3-band pseudo-RGB
GeoTIFF for inference with a SegFormer model originally trained on selected
Maxar bands, e.g. [8, 6, 1].

Why this version exists
-----------------------
A simple gray-to-RGB conversion such as:

    gray -> [gray, gray, gray]

or even:

    gray -> [stretch, CLAHE, edge]

can fail if the output values are expanded to the full uint16 range
0-65535 while the original model was trained on much smaller raw Maxar
band-value distributions.

This script creates three pseudo channels and then quantile-matches each
pseudo channel to the target distribution of real multiband reference imagery.

Default pseudo model channels
-----------------------------
The intended model input channels are:

    M1 = percentile-stretched grayscale     -> matched to original band 8
    M2 = CLAHE-enhanced grayscale           -> matched to original band 6
    M3 = soft Sobel/detail channel          -> matched to original band 1

Default write order: model321
-----------------------------
By default, the file is written as:

    raster band 1 = M3, matched to original band 1
    raster band 2 = M2, matched to original band 6
    raster band 3 = M1, matched to original band 8

Therefore, if your inference code uses:

    src.read([3, 2, 1])

then the model receives:

    [band8-like, band6-like, band1-like]

which matches a model trained with:

    indexes=[8, 6, 1]

Main modes
----------
1. Use a single multiband reference image to compute target quantiles:

    --target-ref /path/to/original_multiband.tif

2. Use a directory of multiband reference images:

    --target-ref-dir /path/to/train/images

3. Reuse a saved target-quantile CSV:

    --target-q-csv target_quantiles_861.csv

Recommended first test
----------------------
Use one original multiband image and its gray-derived pseudo image counterpart
first. Confirm the output distributions using your comparison script before
running large inference.
"""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import rasterio

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "This script requires OpenCV. Install with: pip install opencv-python-headless"
    ) from exc


DEFAULT_Q_LEVELS = np.array(
    [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5],
    dtype=np.float32,
)


def sample_valid_values(arr, valid_mask, max_samples=1_000_000, seed=42):
    flat_idx = np.flatnonzero(valid_mask.ravel())

    if flat_idx.size == 0:
        raise ValueError("No valid pixels found.")

    rng = np.random.default_rng(seed)

    if flat_idx.size > max_samples:
        flat_idx = rng.choice(flat_idx, size=max_samples, replace=False)

    return arr.ravel()[flat_idx].astype(np.float32)


def robust_stretch_to_uint8(
    arr,
    valid_mask,
    p_low=2.0,
    p_high=98.0,
    max_samples=1_000_000,
):
    vals = sample_valid_values(arr, valid_mask, max_samples=max_samples)

    lo, hi = np.percentile(vals, [p_low, p_high])

    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.uint8)
        return out

    out = (arr.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    out[~valid_mask] = 0

    return out


def apply_clahe(gray_u8, valid_mask, clip_limit=2.0, tile_grid_size=8):
    work = gray_u8.copy()

    if np.any(valid_mask):
        fill_value = int(np.median(work[valid_mask]))
    else:
        fill_value = 0

    # Avoid creating artificial high-contrast nodata boundaries.
    work[~valid_mask] = fill_value

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )

    out = clahe.apply(work)
    out[~valid_mask] = 0

    return out.astype(np.uint8)


def make_soft_detail_channel(
    gray_u8,
    valid_mask,
    edge_weight=0.35,
    blur_ksize=3,
    edge_p_low=1.0,
    edge_p_high=99.0,
):
    work = gray_u8.copy()

    if np.any(valid_mask):
        fill_value = int(np.median(work[valid_mask]))
    else:
        fill_value = 0

    work[~valid_mask] = fill_value

    if blur_ksize and blur_ksize > 1:
        work_blur = cv2.GaussianBlur(work, (blur_ksize, blur_ksize), 0)
    else:
        work_blur = work

    gx = cv2.Sobel(work_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx * gx + gy * gy)

    edge_u8 = robust_stretch_to_uint8(
        mag,
        valid_mask,
        p_low=edge_p_low,
        p_high=edge_p_high,
    )

    # Soft detail channel, not binary Canny-style edge-only.
    detail = (
        (1.0 - edge_weight) * gray_u8.astype(np.float32)
        + edge_weight * edge_u8.astype(np.float32)
    )

    detail = np.clip(detail, 0, 255).astype(np.uint8)
    detail[~valid_mask] = 0

    return detail


def build_valid_mask(src, band):
    valid = np.isfinite(band.astype(np.float32))

    if src.nodata is not None:
        valid &= band != src.nodata

    try:
        dataset_mask = src.dataset_mask()
        valid &= dataset_mask > 0
    except Exception:
        pass

    return valid


def unique_quantile_points(src_q, target_q):
    """
    np.interp expects increasing x positions.

    Pseudo channels may have repeated quantile values, especially near 0.
    This function removes duplicate source-quantile positions and keeps the
    corresponding average target value.
    """
    src_q = np.asarray(src_q, dtype=np.float32)
    target_q = np.asarray(target_q, dtype=np.float32)

    order = np.argsort(src_q)
    src_q = src_q[order]
    target_q = target_q[order]

    unique_src = []
    unique_target = []

    for val in np.unique(src_q):
        mask = src_q == val
        unique_src.append(val)
        unique_target.append(float(np.mean(target_q[mask])))

    unique_src = np.asarray(unique_src, dtype=np.float32)
    unique_target = np.asarray(unique_target, dtype=np.float32)

    return unique_src, unique_target


def quantile_match_to_target(
    src_band,
    valid_mask,
    q_levels,
    target_quantiles,
    out_dtype="uint16",
):
    """
    Match a source pseudo band to a target band distribution using quantile mapping.

    src_band is usually uint8 pseudo data.
    target_quantiles are computed from real Maxar band 8, 6, or 1.
    """
    vals = src_band[valid_mask].astype(np.float32)

    if vals.size == 0:
        raise ValueError("No valid pixels available for quantile matching.")

    src_quantiles = np.percentile(vals, q_levels)

    x_src, y_target = unique_quantile_points(src_quantiles, target_quantiles)

    if x_src.size < 2:
        matched = np.full(src_band.shape, y_target[0], dtype=np.float32)
    else:
        matched = np.interp(
            src_band.astype(np.float32),
            x_src,
            y_target,
            left=y_target[0],
            right=y_target[-1],
        ).astype(np.float32)

    matched[~valid_mask] = 0

    if out_dtype == "uint16":
        matched = np.clip(np.rint(matched), 0, 65535).astype(np.uint16)
    elif out_dtype == "uint8":
        matched = np.clip(np.rint(matched), 0, 255).astype(np.uint8)
    else:
        raise ValueError("out_dtype must be 'uint16' or 'uint8'")

    return matched


def sample_reference_quantiles_from_raster(
    path,
    indexes,
    q_levels,
    max_pixels_per_image=200_000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    with rasterio.open(path) as src:
        if max(indexes) > src.count:
            raise ValueError(
                f"Reference image {path} has {src.count} bands, "
                f"but requested indexes={indexes}"
            )

        nodata = src.nodata
        arr = src.read(indexes)  # [C, H, W]

        valid = np.ones(arr.shape[1:], dtype=bool)
        valid &= np.all(np.isfinite(arr.astype(np.float32)), axis=0)

        if nodata is not None:
            valid &= np.all(arr != nodata, axis=0)

        try:
            dataset_mask = src.dataset_mask()
            valid &= dataset_mask > 0
        except Exception:
            pass

        ys, xs = np.where(valid)

        if len(xs) == 0:
            raise ValueError(f"No valid reference pixels found in {path}")

        n = min(max_pixels_per_image, len(xs))
        pick = rng.choice(len(xs), size=n, replace=False)

        samples = []
        for c in range(len(indexes)):
            samples.append(arr[c, ys[pick], xs[pick]].astype(np.float32))

    return samples


def collect_target_quantiles(
    ref_paths,
    indexes,
    q_levels,
    max_pixels_per_image=200_000,
    seed=42,
):
    """
    Compute target quantiles from one or more real multiband reference images.

    indexes should be [8, 6, 1] for this model.
    Returned dict has keys:
        "q_levels"
        "indexes"
        "target_quantiles"
    where target_quantiles is a list of three lists:
        channel 1 target = band 8 quantiles
        channel 2 target = band 6 quantiles
        channel 3 target = band 1 quantiles
    """
    ref_paths = [Path(p) for p in ref_paths]

    all_samples = [[] for _ in indexes]

    for idx, path in enumerate(ref_paths):
        print(f"Sampling reference target distribution: {path}")
        samples = sample_reference_quantiles_from_raster(
            path,
            indexes=indexes,
            q_levels=q_levels,
            max_pixels_per_image=max_pixels_per_image,
            seed=seed + idx,
        )

        for c in range(len(indexes)):
            all_samples[c].append(samples[c])

    target_quantiles = []

    for c, band_index in enumerate(indexes):
        vals = np.concatenate(all_samples[c])
        q = np.percentile(vals, q_levels)
        target_quantiles.append(q.astype(np.float32))

        print(f"\nTarget distribution for original band {band_index}:")
        print(f"  n={len(vals)}")
        print(f"  min={np.min(vals):.3f}")
        print(f"  p02={np.percentile(vals, 2):.3f}")
        print(f"  p50={np.percentile(vals, 50):.3f}")
        print(f"  p98={np.percentile(vals, 98):.3f}")
        print(f"  max={np.max(vals):.3f}")

    return {
        "q_levels": q_levels.astype(float).tolist(),
        "indexes": list(indexes),
        "target_quantiles": [q.astype(float).tolist() for q in target_quantiles],
    }


def save_target_quantiles_csv(target_info, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    q_levels = np.asarray(target_info["q_levels"], dtype=np.float32)
    indexes = target_info["indexes"]
    target_quantiles = target_info["target_quantiles"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_level"] + [f"band_{b}" for b in indexes])

        for i, q in enumerate(q_levels):
            row = [float(q)]
            for c in range(len(indexes)):
                row.append(float(target_quantiles[c][i]))
            writer.writerow(row)

    print(f"\nSaved target quantiles: {out_csv}")


def load_target_quantiles_csv(path):
    path = Path(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        if len(header) != 4:
            raise ValueError(
                f"Expected CSV with q_level + 3 band columns. Got header: {header}"
            )

        indexes = []
        for col in header[1:]:
            if not col.startswith("band_"):
                raise ValueError(f"Expected band column like band_8, got {col}")
            indexes.append(int(col.replace("band_", "")))

        q_levels = []
        cols = [[], [], []]

        for row in reader:
            q_levels.append(float(row[0]))
            for c in range(3):
                cols[c].append(float(row[c + 1]))

    target_info = {
        "q_levels": q_levels,
        "indexes": indexes,
        "target_quantiles": cols,
    }

    print(f"Loaded target quantiles from: {path}")
    print(f"Target indexes from CSV: {indexes}")

    return target_info


def summarize_output_stack(stack, label):
    print(f"\nOutput summary: {label}")
    for c in range(stack.shape[0]):
        vals = stack[c].astype(np.float32)
        print(
            f"  raster band {c + 1}: "
            f"min={np.min(vals):.3f}, "
            f"p02={np.percentile(vals, 2):.3f}, "
            f"p50={np.percentile(vals, 50):.3f}, "
            f"p98={np.percentile(vals, 98):.3f}, "
            f"max={np.max(vals):.3f}, "
            f"mean={np.mean(vals):.3f}"
        )


def convert_one(
    input_path,
    output_path,
    target_info,
    p_low=2.0,
    p_high=98.0,
    clahe_clip=2.0,
    clahe_grid=8,
    edge_weight=0.35,
    out_dtype="uint16",
    write_order="model321",
    nodata_zero=False,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    q_levels = np.asarray(target_info["q_levels"], dtype=np.float32)
    target_indexes = target_info["indexes"]
    target_quantiles = [
        np.asarray(q, dtype=np.float32) for q in target_info["target_quantiles"]
    ]

    if target_indexes != [8, 6, 1]:
        print(
            f"WARNING: target quantile indexes are {target_indexes}, "
            "not [8, 6, 1]. Continuing anyway."
        )

    with rasterio.open(input_path) as src:
        if src.count < 1:
            raise ValueError(f"No bands found in {input_path}")

        gray = src.read(1)
        valid_mask = build_valid_mask(src, gray)

        c1_stretch_u8 = robust_stretch_to_uint8(
            gray,
            valid_mask,
            p_low=p_low,
            p_high=p_high,
        )

        c2_clahe_u8 = apply_clahe(
            c1_stretch_u8,
            valid_mask,
            clip_limit=clahe_clip,
            tile_grid_size=clahe_grid,
        )

        c3_detail_u8 = make_soft_detail_channel(
            c1_stretch_u8,
            valid_mask,
            edge_weight=edge_weight,
        )

        # Match intended model channels:
        #   M1 -> original band 8 distribution
        #   M2 -> original band 6 distribution
        #   M3 -> original band 1 distribution
        m1_band8_like = quantile_match_to_target(
            c1_stretch_u8,
            valid_mask,
            q_levels,
            target_quantiles[0],
            out_dtype=out_dtype,
        )

        m2_band6_like = quantile_match_to_target(
            c2_clahe_u8,
            valid_mask,
            q_levels,
            target_quantiles[1],
            out_dtype=out_dtype,
        )

        m3_band1_like = quantile_match_to_target(
            c3_detail_u8,
            valid_mask,
            q_levels,
            target_quantiles[2],
            out_dtype=out_dtype,
        )

        if write_order == "model321":
            # Important:
            # src.read([3, 2, 1]) returns:
            # [band8-like, band6-like, band1-like]
            stack = np.stack(
                [
                    m3_band1_like,  # raster band 1
                    m2_band6_like,  # raster band 2
                    m1_band8_like,  # raster band 3
                ],
                axis=0,
            )

            band_descriptions = [
                "M3_soft_detail_matched_to_original_band_1_written_as_band_1",
                "M2_CLAHE_matched_to_original_band_6_written_as_band_2",
                "M1_stretch_matched_to_original_band_8_written_as_band_3",
            ]

        elif write_order == "normal123":
            # src.read([1, 2, 3]) returns:
            # [band8-like, band6-like, band1-like]
            stack = np.stack(
                [
                    m1_band8_like,  # raster band 1
                    m2_band6_like,  # raster band 2
                    m3_band1_like,  # raster band 3
                ],
                axis=0,
            )

            band_descriptions = [
                "M1_stretch_matched_to_original_band_8_written_as_band_1",
                "M2_CLAHE_matched_to_original_band_6_written_as_band_2",
                "M3_soft_detail_matched_to_original_band_1_written_as_band_3",
            ]

        else:
            raise ValueError("write_order must be 'model321' or 'normal123'")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype=out_dtype,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        if nodata_zero:
            profile.update(nodata=0)
        else:
            # Recommended. Avoid treating valid dark pixels as nodata.
            profile.pop("nodata", None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stack)

            for band_i, desc in enumerate(band_descriptions, start=1):
                dst.set_band_description(band_i, desc)

            dst.update_tags(
                pseudo_rgb="true",
                pseudo_method="gray_to_pseudo_rgb_quantile_matched",
                model_expected_indexes="8,6,1",
                write_order=write_order,
                inference_note=(
                    "With write_order=model321, src.read([3,2,1]) returns "
                    "[band8-like,band6-like,band1-like]."
                ),
                target_quantile_indexes=json.dumps(target_indexes),
                target_quantile_levels=json.dumps(
                    [float(x) for x in q_levels.tolist()]
                ),
            )

    print(f"\nWrote: {output_path}")
    summarize_output_stack(stack, output_path.name)


def collect_input_paths(input_dir, patterns):
    input_dir = Path(input_dir)

    paths = []
    for pattern in patterns:
        paths.extend(sorted(input_dir.glob(pattern)))

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


USAGE = r"""
USAGE

1. Single grayscale image, target quantiles from one original multiband reference

   python gray_pseudo_rgb_baseline.py \
     --input /path/to/historical_or_gray.tif \
     --output /path/to/historical_or_gray_pseudoRGB_qmatch.tif \
     --target-ref /path/to/original_multiband_reference.tif \
     --target-indexes 8 6 1 \
     --write-order model321 \
     --out-dtype uint16

2. Batch conversion, target quantiles from one reference image

   python gray_pseudo_rgb_baseline.py \
     --input-dir /path/to/gray_images \
     --output-dir /path/to/pseudoRGB_qmatch_images \
     --patterns "*.tif" "*.img" \
     --target-ref /path/to/original_multiband_reference.tif \
     --target-indexes 8 6 1 \
     --write-order model321 \
     --out-dtype uint16

3. Batch conversion, target quantiles from a directory of original multiband images

   python gray_pseudo_rgb_baseline.py \
     --input-dir /path/to/gray_images \
     --output-dir /path/to/pseudoRGB_qmatch_images \
     --patterns "*.tif" "*.img" \
     --target-ref-dir /path/to/original_multiband_train_images \
     --target-ref-patterns "*.tif" "*.tiff" \
     --target-indexes 8 6 1 \
     --save-target-q-csv target_quantiles_861.csv \
     --write-order model321 \
     --out-dtype uint16

4. Reuse previously saved target quantiles

   python gray_pseudo_rgb_baseline.py \
     --input-dir /path/to/gray_images \
     --output-dir /path/to/pseudoRGB_qmatch_images \
     --patterns "*.tif" "*.img" \
     --target-q-csv target_quantiles_861.csv \
     --write-order model321 \
     --out-dtype uint16

5. Recommended inference pairing

   If your inference code uses:

       indexes = [8, 6, 1] if src.count >= 8 else [3, 2, 1]

   then keep:

       --write-order model321

   The output pseudo image will be read by inference as:

       src.read([3, 2, 1])
       -> [band8-like stretch, band6-like CLAHE, band1-like detail]

6. Parameters worth testing

   Softer edge/detail channel:

       --edge-weight 0.25

   Default:

       --edge-weight 0.35

   Stronger detail channel:

       --edge-weight 0.50

7. Important warning

   Do not use --nodata-zero unless you are certain 0 should be treated as nodata.
   For pseudo images, valid dark pixels can be 0, and nodata=0 can cause the
   inference null mask to erase valid predictions.
"""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert grayscale rasters to quantile-matched pseudo-RGB "
            "for a SegFormer trained on original multiband indexes [8,6,1]."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Single input grayscale raster.")
    input_group.add_argument("--input-dir", type=Path, help="Input directory for batch mode.")

    parser.add_argument("--output", type=Path, help="Single output GeoTIFF.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch mode.")

    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.img", "*.tif", "*.tiff"],
        help="Input filename patterns for batch mode.",
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-ref",
        type=Path,
        help="Single original multiband reference raster used to compute target quantiles.",
    )
    target_group.add_argument(
        "--target-ref-dir",
        type=Path,
        help="Directory of original multiband reference rasters used to compute target quantiles.",
    )
    target_group.add_argument(
        "--target-q-csv",
        type=Path,
        help="Previously saved target quantile CSV.",
    )

    parser.add_argument(
        "--target-ref-patterns",
        nargs="+",
        default=["*.tif", "*.tiff", "*.img"],
        help="Reference filename patterns if using --target-ref-dir.",
    )

    parser.add_argument(
        "--target-indexes",
        nargs=3,
        type=int,
        default=[8, 6, 1],
        help="Original model-training band indexes. Default: 8 6 1.",
    )

    parser.add_argument(
        "--save-target-q-csv",
        type=Path,
        help="Optional path to save computed target quantiles as CSV.",
    )

    parser.add_argument(
        "--max-ref-pixels-per-image",
        type=int,
        default=200_000,
        help="Max valid reference pixels sampled per reference image.",
    )

    parser.add_argument("--p-low", type=float, default=2.0)
    parser.add_argument("--p-high", type=float, default=98.0)

    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="OpenCV CLAHE clip limit. Typical: 1.0 to 3.0.",
    )

    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=8,
        help="CLAHE tile grid size. Typical: 8 or 16.",
    )

    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.35,
        help="How much Sobel/detail to mix into third channel.",
    )

    parser.add_argument(
        "--out-dtype",
        choices=["uint8", "uint16"],
        default="uint16",
        help="Recommended: uint16 for your current inference pipeline.",
    )

    parser.add_argument(
        "--write-order",
        choices=["model321", "normal123"],
        default="model321",
        help="model321 matches inference using indexes=[3,2,1].",
    )

    parser.add_argument(
        "--nodata-zero",
        action="store_true",
        help="Write output nodata=0. Usually NOT recommended for pseudo imagery.",
    )

    args = parser.parse_args()

    if args.target_q_csv:
        target_info = load_target_quantiles_csv(args.target_q_csv)

    else:
        if args.target_ref:
            ref_paths = [args.target_ref]
        else:
            ref_paths = collect_input_paths(args.target_ref_dir, args.target_ref_patterns)

        if not ref_paths:
            raise ValueError("No reference images found for target quantile computation.")

        target_info = collect_target_quantiles(
            ref_paths,
            indexes=args.target_indexes,
            q_levels=DEFAULT_Q_LEVELS,
            max_pixels_per_image=args.max_ref_pixels_per_image,
        )

        if args.save_target_q_csv:
            save_target_quantiles_csv(target_info, args.save_target_q_csv)

    if args.input:
        if args.output is None:
            args.output = args.input.with_name(args.input.stem + "_pseudoRGB_qmatch.tif")

        convert_one(
            args.input,
            args.output,
            target_info=target_info,
            p_low=args.p_low,
            p_high=args.p_high,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            edge_weight=args.edge_weight,
            out_dtype=args.out_dtype,
            write_order=args.write_order,
            nodata_zero=args.nodata_zero,
        )

    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required with --input-dir.")

        input_paths = collect_input_paths(args.input_dir, args.patterns)

        if not input_paths:
            raise ValueError(f"No input files found in {args.input_dir}")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in input_paths:
            output_path = args.output_dir / input_path.name
            #output_path = args.output_dir / f"{input_path.stem}_pseudoRGB_qmatch.tif"

            convert_one(
                input_path,
                output_path,
                target_info=target_info,
                p_low=args.p_low,
                p_high=args.p_high,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid,
                edge_weight=args.edge_weight,
                out_dtype=args.out_dtype,
                write_order=args.write_order,
                nodata_zero=args.nodata_zero,
            )

if __name__ == "__main__":
    main()



"""
python gray_to_pseudo_rgb_baseline.py \
 --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/rgb_gray_train/images \
 --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_rgb_train/images \
 --patterns "*.tif" "*.img" \
 --target-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/images \
 --target-ref-patterns "*.tif" "*.tiff" \
 --target-indexes 8 6 1 \
 --save-target-q-csv target_quantiles_861.csv \
 --write-order model321 \
 --out-dtype uint16


python gray_to_pseudo_rgb_baseline.py \
  --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/rgb_gray_train/images \
  --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_rgb_train/images \
  --patterns "*.img" "*.tif" \
  --write-order model321 \
  --out-dtype uint16
  
python gray_pseudo_rgb_baseline.py \
 --input-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/b2_7gray_train/images \
 --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/b2_7gray_rgb_train/images \
 --patterns "*.tif" "*.img" \
 --target-q-csv target_quantiles_861.csv \
 --write-order model321 \
 --out-dtype uint16
 
"""