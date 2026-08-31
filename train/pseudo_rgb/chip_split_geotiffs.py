#!/usr/bin/env python3

"""
chip_split_geotiffs.py

Split aligned original multiband images, grayscale images, and masks into
GeoTIFF chips while preserving georeferencing.

Designed for the current TCN / gray-adapter workflow:

    original multiband image  -> 1024 chips
    grayscale image           -> matching 1024 chips
    mask                      -> matching 1024 chips

The original image, gray image, and mask are expected to have matching names
or at least matching stems.

Example input:

    original_images/
        tileA.tif
    gray_images/
        tileA.tif
    masks/
        tileA.tif

Example output:

    out/
        train/
            original/tileA_1.tif
            gray/tileA_1.tif
            masks/tileA_1.tif
        val/
            original/...
            gray/...
            masks/...
        test/
            original/...
            gray/...
            masks/...

Important:
    The split is done at the parent-image level before chipping.
    This avoids leakage where tileA_1 is in train and tileA_2 is in val/test.

Default:
    chip_size = 1024
    split = 70 / 15 / 15
    output extension = .tif
    partial chips are skipped unless --keep-partial is used
"""

from pathlib import Path
import argparse
import csv
import random
import math

import numpy as np
import rasterio
from rasterio.windows import Window


def collect_paths(directory, patterns):
    directory = Path(directory)

    paths = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def index_by_stem(paths):
    out = {}
    for p in paths:
        out[p.stem] = p
    return out


def find_matching_path(reference_path, candidate_dir, candidate_index):
    """
    Prefer exact filename match. If not found, fall back to matching by stem.
    """
    candidate_dir = Path(candidate_dir)

    exact = candidate_dir / reference_path.name
    if exact.exists():
        return exact

    by_stem = candidate_index.get(reference_path.stem)
    if by_stem is not None:
        return by_stem

    raise FileNotFoundError(
        f"No matching file found for {reference_path.name} in {candidate_dir}"
    )


def make_scene_split(paths, train_ratio, val_ratio, test_ratio, seed):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Make sure all images are assigned exactly once.
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]

    return {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }


def get_chip_windows(width, height, chip_size, keep_partial=False):
    """
    Return row-major 1024 windows.

    For 2048 x 2048 and chip_size=1024:

        _1 = top-left
        _2 = top-right
        _3 = bottom-left
        _4 = bottom-right
    """
    windows = []
    chip_id = 1

    for y in range(0, height, chip_size):
        for x in range(0, width, chip_size):
            w = min(chip_size, width - x)
            h = min(chip_size, height - y)

            if not keep_partial and (w != chip_size or h != chip_size):
                continue

            windows.append((chip_id, Window(x, y, w, h)))
            chip_id += 1

    return windows


def update_profile_for_window(src, window, compress):
    transform = src.window_transform(window)

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=int(window.height),
        width=int(window.width),
        transform=transform,
        compress=compress,
        BIGTIFF="IF_SAFER",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    # Avoid carrying over incompatible block sizes from source.
    for key in ["interleave"]:
        if key in src.profile:
            profile[key] = src.profile[key]

    return profile


def copy_tags_and_descriptions(src, dst, parent_name, chip_id, window):
    # Dataset-level tags
    tags = src.tags()
    tags.update(
        parent_image=parent_name,
        chip_id=str(chip_id),
        xoff=str(int(window.col_off)),
        yoff=str(int(window.row_off)),
        chip_width=str(int(window.width)),
        chip_height=str(int(window.height)),
    )
    dst.update_tags(**tags)

    # Band descriptions
    for b in range(1, src.count + 1):
        desc = src.descriptions[b - 1]
        if desc:
            dst.set_band_description(b, desc)

    # Colormap, usually only relevant to single-band masks
    try:
        cmap = src.colormap(1)
        if cmap:
            dst.write_colormap(1, cmap)
    except Exception:
        pass


def write_chip(src_path, out_path, window, chip_id, parent_name, compress="lzw"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        data = src.read(window=window)

        profile = update_profile_for_window(src, window, compress=compress)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            copy_tags_and_descriptions(
                src=src,
                dst=dst,
                parent_name=parent_name,
                chip_id=chip_id,
                window=window,
            )


def mask_foreground_count(mask_path, window):
    with rasterio.open(mask_path) as src:
        mask = src.read(1, window=window)

    return int(np.count_nonzero(mask > 0))


def validate_alignment(orig_path, gray_path, mask_path):
    with rasterio.open(orig_path) as orig, rasterio.open(gray_path) as gray, rasterio.open(mask_path) as mask:
        if orig.width != gray.width or orig.height != gray.height:
            raise ValueError(
                f"Size mismatch:\n"
                f"  original: {orig_path} {orig.width}x{orig.height}\n"
                f"  gray:     {gray_path} {gray.width}x{gray.height}"
            )

        if orig.width != mask.width or orig.height != mask.height:
            raise ValueError(
                f"Size mismatch:\n"
                f"  original: {orig_path} {orig.width}x{orig.height}\n"
                f"  mask:     {mask_path} {mask.width}x{mask.height}"
            )

        # CRS/transform can differ slightly in some workflows, so warn instead of failing.
        if orig.crs != gray.crs:
            print(f"WARNING CRS mismatch between original and gray: {orig_path.name}")

        if orig.crs != mask.crs:
            print(f"WARNING CRS mismatch between original and mask: {orig_path.name}")

        if orig.transform != gray.transform:
            print(f"WARNING transform mismatch between original and gray: {orig_path.name}")

        if orig.transform != mask.transform:
            print(f"WARNING transform mismatch between original and mask: {orig_path.name}")


def process_scene(
    orig_path,
    gray_path,
    mask_path,
    split_name,
    out_dir,
    chip_size,
    keep_partial,
    output_ext,
    compress,
    min_mask_fg_pixels,
    manifest_rows,
):
    validate_alignment(orig_path, gray_path, mask_path)

    with rasterio.open(orig_path) as src:
        width = src.width
        height = src.height

    windows = get_chip_windows(
        width=width,
        height=height,
        chip_size=chip_size,
        keep_partial=keep_partial,
    )

    parent_stem = orig_path.stem

    for chip_id, window in windows:
        fg_count = mask_foreground_count(mask_path, window)

        if fg_count < min_mask_fg_pixels:
            continue

        chip_name = f"{parent_stem}_{chip_id}{output_ext}"

        out_orig = Path(out_dir) / split_name / "original" / chip_name
        out_gray = Path(out_dir) / split_name / "gray" / chip_name
        out_mask = Path(out_dir) / split_name / "masks" / chip_name

        write_chip(
            src_path=orig_path,
            out_path=out_orig,
            window=window,
            chip_id=chip_id,
            parent_name=orig_path.name,
            compress=compress,
        )

        write_chip(
            src_path=gray_path,
            out_path=out_gray,
            window=window,
            chip_id=chip_id,
            parent_name=gray_path.name,
            compress=compress,
        )

        write_chip(
            src_path=mask_path,
            out_path=out_mask,
            window=window,
            chip_id=chip_id,
            parent_name=mask_path.name,
            compress=compress,
        )

        manifest_rows.append(
            {
                "split": split_name,
                "parent_stem": parent_stem,
                "chip_id": chip_id,
                "chip_name": chip_name,
                "original_chip": str(out_orig),
                "gray_chip": str(out_gray),
                "mask_chip": str(out_mask),
                "xoff": int(window.col_off),
                "yoff": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
                "mask_fg_pixels": fg_count,
            }
        )


def write_manifest(manifest_rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(manifest_rows) == 0:
        print("WARNING: manifest is empty.")
        return

    fieldnames = list(manifest_rows[0].keys())

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote manifest: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Split original, gray, and mask GeoTIFFs into aligned 1024 chips with train/val/test splits."
    )

    parser.add_argument("--orig-dir", required=True, type=Path, help="Directory of original multiband images.")
    parser.add_argument("--gray-dir", required=True, type=Path, help="Directory of grayscale images.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Directory of masks.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")

    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.tif", "*.tiff", "*.img"],
        help="Image filename patterns to include.",
    )

    parser.add_argument("--chip-size", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-ext",
        default=".tif",
        help="Output extension. Recommended: .tif",
    )

    parser.add_argument(
        "--compress",
        default="lzw",
        help="GeoTIFF compression. Default: lzw",
    )

    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep edge chips smaller than chip-size. Default skips partial chips.",
    )

    parser.add_argument(
        "--min-mask-fg-pixels",
        type=int,
        default=0,
        help=(
            "Minimum foreground pixels required in mask chip. "
            "Default 0 keeps all chips, including background-only chips."
        ),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    orig_paths = collect_paths(args.orig_dir, args.patterns)
    gray_paths = collect_paths(args.gray_dir, args.patterns)
    mask_paths = collect_paths(args.mask_dir, args.patterns)

    if len(orig_paths) == 0:
        raise ValueError(f"No original images found in {args.orig_dir}")

    gray_index = index_by_stem(gray_paths)
    mask_index = index_by_stem(mask_paths)

    split_paths = make_scene_split(
        paths=orig_paths,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("Scene split:")
    for split_name, paths in split_paths.items():
        print(f"  {split_name}: {len(paths)} parent images")

    manifest_rows = []

    for split_name, paths in split_paths.items():
        print(f"\nProcessing split: {split_name}")

        for orig_path in paths:
            gray_path = find_matching_path(orig_path, args.gray_dir, gray_index)
            mask_path = find_matching_path(orig_path, args.mask_dir, mask_index)

            print(f"  {orig_path.name}")

            process_scene(
                orig_path=orig_path,
                gray_path=gray_path,
                mask_path=mask_path,
                split_name=split_name,
                out_dir=args.out_dir,
                chip_size=args.chip_size,
                keep_partial=args.keep_partial,
                output_ext=args.output_ext,
                compress=args.compress,
                min_mask_fg_pixels=args.min_mask_fg_pixels,
                manifest_rows=manifest_rows,
            )

    manifest_csv = args.out_dir / "chip_manifest.csv"
    write_manifest(manifest_rows, manifest_csv)

    print("\nDone.")
    print(f"Total chips written per dataset type: {len(manifest_rows)}")
    print("Output structure:")
    print(f"  {args.out_dir}/train/original")
    print(f"  {args.out_dir}/train/gray")
    print(f"  {args.out_dir}/train/masks")
    print(f"  {args.out_dir}/val/original")
    print(f"  {args.out_dir}/val/gray")
    print(f"  {args.out_dir}/val/masks")
    print(f"  {args.out_dir}/test/original")
    print(f"  {args.out_dir}/test/gray")
    print(f"  {args.out_dir}/test/masks")


if __name__ == "__main__":
    main()

"""
USAGE
python chip_split_geotiffs.py \
  --orig-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/images \
  --gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/b2_7gray_rgb_train/images \
  --mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/masks \
  --out-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split \
  --chip-size 1024 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
"""