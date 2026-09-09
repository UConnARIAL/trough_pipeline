#!/usr/bin/env python3

import argparse
import csv
import os
import re
import shutil
from pathlib import Path


def clean_line(line: str) -> str:
    line = line.strip()
    line = line.strip('"').strip("'").strip()
    if line.endswith(","):
        line = line[:-1].strip()
    return line


def get_basename(path_or_name: str) -> str:
    return path_or_name.rstrip("/").split("/")[-1]


def infer_parent_from_name(name: str) -> str | None:
    """
    Finds parent tile from names like:
      ArcticMosaic_34_33_1_2_TCN.gpkg
      ArcticMosaic_34_33_1_2_mask.tif
      34_33_1_2_TCN.gpkg
    Returns:
      34_33
    """
    m = re.search(r"(?:ArcticMosaic_)?(\d+)_(\d+)", name)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def gpkg_to_mask_name(gpkg_name: str) -> str:
    """
    Expected conversion:
      ArcticMosaic_34_33_1_2_TCN.gpkg
        -> ArcticMosaic_34_33_1_2_mask.tif
    """
    if gpkg_name.endswith("_TCN.gpkg"):
        return gpkg_name.replace("_TCN.gpkg", "_mask.tif")

    if gpkg_name.endswith(".gpkg"):
        stem = gpkg_name[:-5]
        if stem.endswith("_TCN"):
            stem = stem[:-4]
        return stem + "_mask.tif"

    raise ValueError(f"Not a .gpkg name: {gpkg_name}")


def build_mask_index(mask_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in mask_root.rglob("*_mask.tif"):
        index.setdefault(p.name, []).append(p)
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Create a pseudo input mask directory by symlinking masks for missing output GPKGs."
    )
    parser.add_argument(
        "--missing-gpkgs",
        required=True,
        help="Text file containing missing output GPKG file names or full paths, one per line.",
    )
    parser.add_argument(
        "--mask-root",
        required=True,
        help="Original full input mask root, e.g. /scratch/.../masks_3338",
    )
    parser.add_argument(
        "--pseudo-root",
        required=True,
        help="Output pseudo mask root that will contain symlinks only for missing files.",
    )
    parser.add_argument(
        "--manifest",
        default="missing_mask_symlink_manifest.csv",
        help="CSV manifest output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be linked without creating symlinks.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete pseudo-root before creating links. Use carefully.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination symlinks/files.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Create relative symlinks instead of absolute symlinks.",
    )

    args = parser.parse_args()

    missing_file = Path(args.missing_gpkgs)
    mask_root = Path(args.mask_root).resolve()
    pseudo_root = Path(args.pseudo_root).resolve()
    manifest_path = Path(args.manifest)

    if not missing_file.exists():
        raise FileNotFoundError(f"Missing list not found: {missing_file}")

    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    if args.clean and pseudo_root.exists():
        print(f"CLEANING pseudo root: {pseudo_root}")
        if not args.dry_run:
            shutil.rmtree(pseudo_root)

    if not args.dry_run:
        pseudo_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading missing GPKG list: {missing_file}")
    print(f"Input mask root        : {mask_root}")
    print(f"Pseudo mask root       : {pseudo_root}")
    print(f"Manifest               : {manifest_path}")
    print(f"Dry run                : {args.dry_run}")
    print()

    print("Building mask filename index...")
    mask_index = build_mask_index(mask_root)
    print(f"Indexed mask files: {sum(len(v) for v in mask_index.values())}")
    print()

    stats = {
        "lines": 0,
        "created": 0,
        "exists_ok": 0,
        "overwritten": 0,
        "missing_mask": 0,
        "bad_name": 0,
        "ambiguous": 0,
        "skipped_existing_conflict": 0,
    }

    rows = []

    with missing_file.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = clean_line(raw)

            if not line or line.startswith("#"):
                continue

            stats["lines"] += 1

            gpkg_name = get_basename(line)

            try:
                mask_name = gpkg_to_mask_name(gpkg_name)
            except ValueError as e:
                stats["bad_name"] += 1
                rows.append({
                    "status": "BAD_NAME",
                    "input_line": line,
                    "output_gpkg": gpkg_name,
                    "parent": "",
                    "mask_name": "",
                    "source": "",
                    "destination": "",
                    "message": str(e),
                })
                continue

            parent = infer_parent_from_name(mask_name)

            if parent is None:
                stats["bad_name"] += 1
                rows.append({
                    "status": "BAD_NAME",
                    "input_line": line,
                    "output_gpkg": gpkg_name,
                    "parent": "",
                    "mask_name": mask_name,
                    "source": "",
                    "destination": "",
                    "message": "Could not infer parent tile from file name.",
                })
                continue

            expected_src = mask_root / parent / mask_name

            if expected_src.exists():
                src = expected_src
            else:
                matches = mask_index.get(mask_name, [])

                if len(matches) == 1:
                    src = matches[0]
                elif len(matches) > 1:
                    stats["ambiguous"] += 1
                    rows.append({
                        "status": "AMBIGUOUS",
                        "input_line": line,
                        "output_gpkg": gpkg_name,
                        "parent": parent,
                        "mask_name": mask_name,
                        "source": ";".join(str(m) for m in matches),
                        "destination": "",
                        "message": "Multiple masks with same basename found.",
                    })
                    continue
                else:
                    stats["missing_mask"] += 1
                    rows.append({
                        "status": "MISSING_MASK",
                        "input_line": line,
                        "output_gpkg": gpkg_name,
                        "parent": parent,
                        "mask_name": mask_name,
                        "source": str(expected_src),
                        "destination": "",
                        "message": "Mask not found at expected path or anywhere under mask root.",
                    })
                    continue

            dest_dir = pseudo_root / parent
            dest = dest_dir / mask_name

            status = ""
            message = ""

            if dest.exists() or dest.is_symlink():
                if dest.is_symlink() and dest.resolve() == src.resolve():
                    stats["exists_ok"] += 1
                    status = "EXISTS_OK"
                    message = "Correct symlink already exists."
                elif args.overwrite:
                    stats["overwritten"] += 1
                    status = "OVERWRITTEN"
                    message = "Existing destination replaced."
                    if not args.dry_run:
                        dest.unlink()
                else:
                    stats["skipped_existing_conflict"] += 1
                    rows.append({
                        "status": "SKIPPED_EXISTING_CONFLICT",
                        "input_line": line,
                        "output_gpkg": gpkg_name,
                        "parent": parent,
                        "mask_name": mask_name,
                        "source": str(src),
                        "destination": str(dest),
                        "message": "Destination exists but does not point to source. Use --overwrite to replace.",
                    })
                    continue

            if status != "EXISTS_OK":
                link_target = src
                if args.relative:
                    link_target = Path(os.path.relpath(src, start=dest_dir))

                if not args.dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    os.symlink(link_target, dest)

                stats["created"] += 1
                if not status:
                    status = "CREATED"
                    message = "Symlink created."

            rows.append({
                "status": status,
                "input_line": line,
                "output_gpkg": gpkg_name,
                "parent": parent,
                "mask_name": mask_name,
                "source": str(src),
                "destination": str(dest),
                "message": message,
            })

    fieldnames = [
        "status",
        "input_line",
        "output_gpkg",
        "parent",
        "mask_name",
        "source",
        "destination",
        "message",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Summary")
    print("-------")
    for k, v in stats.items():
        print(f"{k:28s}: {v}")

    print()
    print(f"Manifest written: {manifest_path}")

    print()
    print("Pseudo-root count:")
    if pseudo_root.exists():
        n_links = sum(1 for p in pseudo_root.rglob("*_mask.tif"))
        print(f"{pseudo_root}: {n_links} mask links/files")
    else:
        print(f"{pseudo_root}: does not exist yet")


if __name__ == "__main__":
    main()

"""
USAGE
python make_missing_mask_symlinks.py \
  --missing-gpkgs missing_gpkgs.txt \
  --mask-root /home1/09208/asperera/PDG_shared2/CanadaTundraMosaicMasks \
  --pseudo-root /scratch2/projects/PDG_shared/CanadaTundraMosaicMasks_missing_only/ \
  --manifest missing_mask_symlink_manifest.csv \
  --dry-run


  --clean


"""