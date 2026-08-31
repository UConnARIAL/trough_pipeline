from pathlib import Path
import argparse
import shutil

def build_index(folder: Path, recursive: bool):
    files = list(folder.rglob("*.tif")) if recursive else list(folder.glob("*.tif"))
    index = {}

    for p in files:
        if p.name in index:
            raise RuntimeError(
                f"Duplicate filename found: {p.name}\n"
                f"  {index[p.name]}\n"
                f"  {p}"
            )
        index[p.name] = p

    return index


def get_base_from_augmented(ref_name: str):
    """
    chip_0030_0.tif -> chip_0030.tif, mask_0030.tif
    chip_0030_4.tif -> chip_0030.tif, mask_0030.tif
    """
    p = Path(ref_name)
    stem = p.stem
    suffix = p.suffix

    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Expected augmented name like chip_0030_0.tif, got: {ref_name}")

    # remove final augmentation id
    base_stem = "_".join(parts[:-1])   # chip_0030
    base_id = base_stem.split("_", 1)[1]  # 0030

    image_name = f"chip_{base_id}{suffix}"
    mask_name = f"mask_{base_id}{suffix}"

    return image_name, mask_name


def main():
    parser = argparse.ArgumentParser(
        description="Copy unique original image/mask pairs based on augmented reference filenames."
    )

    parser.add_argument("--ref-img-dir", required=True, type=Path,
                        help="Reference augmented image dir, e.g. chip_0030_0.tif")
    parser.add_argument("--all-img-dir", required=True, type=Path,
                        help="Large original image dir, e.g. chip_0030.tif")
    parser.add_argument("--all-mask-dir", required=True, type=Path,
                        help="Large original mask dir, e.g. mask_0030.tif")
    parser.add_argument("--out-img-dir", required=True, type=Path)
    parser.add_argument("--out-mask-dir", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")

    args = parser.parse_args()

    args.out_img_dir.mkdir(parents=True, exist_ok=True)
    args.out_mask_dir.mkdir(parents=True, exist_ok=True)

    ref_files = sorted(args.ref_img_dir.glob("*.tif"))

    img_index = build_index(args.all_img_dir, args.recursive)
    mask_index = build_index(args.all_mask_dir, args.recursive)

    # Deduplicate augmented references into unique original base pairs
    unique_pairs = {}

    for ref in ref_files:
        image_name, mask_name = get_base_from_augmented(ref.name)
        unique_pairs[image_name] = mask_name

    copied_images = 0
    copied_masks = 0
    missing_images = []
    missing_masks = []

    manifest_path = args.out_img_dir.parent / "copy_unique_original_manifest.csv"

    with open(manifest_path, "w") as mf:
        mf.write("base_image,base_mask,src_image,src_mask,out_image,out_mask\n")

        for image_name, mask_name in sorted(unique_pairs.items()):
            src_img = img_index.get(image_name)
            src_mask = mask_index.get(mask_name)

            out_img_path = args.out_img_dir / image_name
            out_mask_path = args.out_mask_dir / mask_name

            if src_img is None:
                missing_images.append(image_name)
                out_img_written = ""
            else:
                shutil.copy2(src_img, out_img_path)
                copied_images += 1
                out_img_written = str(out_img_path)

            if src_mask is None:
                missing_masks.append(mask_name)
                out_mask_written = ""
            else:
                shutil.copy2(src_mask, out_mask_path)
                copied_masks += 1
                out_mask_written = str(out_mask_path)

            mf.write(
                f"{image_name},{mask_name},"
                f"{src_img if src_img else ''},"
                f"{src_mask if src_mask else ''},"
                f"{out_img_written},{out_mask_written}\n"
            )

    actual_images = len(list(args.out_img_dir.glob("*.tif")))
    actual_masks = len(list(args.out_mask_dir.glob("*.tif")))

    print("\nDone.")
    print(f"Reference augmented files: {len(ref_files)}")
    print(f"Unique original pairs:     {len(unique_pairs)}")
    print(f"Copied images:             {copied_images}")
    print(f"Copied masks:              {copied_masks}")
    print(f"Actual output images:      {actual_images}")
    print(f"Actual output masks:       {actual_masks}")
    print(f"Missing images:            {len(missing_images)}")
    print(f"Missing masks:             {len(missing_masks)}")
    print(f"Manifest:                  {manifest_path}")

    if missing_images:
        print("\nFirst missing images:")
        for x in missing_images[:30]:
            print(" ", x)

    if missing_masks:
        print("\nFirst missing masks:")
        for x in missing_masks[:30]:
            print(" ", x)


if __name__ == "__main__":
    main()




"""
python copy_subset_from_ref_dir.py \
  --ref-img-dir /scratch2/projects/PDG_shared/TCN_Training/test_1024/images \
  --all-img-dir /scratch2/projects/PDG_shared/TCN_Training/georef/images \
  --all-mask-dir /scratch2/projects/PDG_shared/TCN_Training/georef/masks \
  --out-img-dir /scratch2/projects/PDG_shared/TCN_Training_GRAY/subsets/test/images \
  --out-mask-dir /scratch2/projects/PDG_shared/TCN_Training_GRAY/subsets/test/masks

"""