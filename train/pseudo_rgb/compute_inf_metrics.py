from pathlib import Path
import argparse
import csv

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


IMAGE_EXTS = [".tif", ".tiff"]

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


def has_valid_georef(src):
    """
    Return False for non-georeferenced / default-georeferenced chips.
    """
    if src.crs is None:
        return False

    if src.transform is None:
        return False

    # Common default transform for non-georeferenced rasters
    if src.transform.almost_equals(Affine.identity()):
        return False

    return True


def read_pred_mask_aligned(pred_path: Path, gt_path: Path, pred_threshold: float):
    """
    Read prediction mask and align to GT.

    Important behavior:
    1. If prediction and GT have the same array shape, compare directly.
       This is safest for augmented/non-georeferenced chips.
    2. Only reproject if shapes differ AND both rasters have valid georeferencing.
    3. If shapes differ but georeferencing is invalid, fail loudly.
    """

    with rasterio.open(gt_path) as gt_src, rasterio.open(pred_path) as pred_src:
        gt_height = gt_src.height
        gt_width = gt_src.width

        pred_arr = pred_src.read(1)

        # Best path for chip-based evaluation:
        # same pixel dimensions means compare directly, ignore CRS/transform.
        if pred_arr.shape == (gt_height, gt_width):
            aligned = pred_arr

        else:
            pred_has_geo = has_valid_georef(pred_src)
            gt_has_geo = has_valid_georef(gt_src)

            if pred_has_geo and gt_has_geo:
                aligned = np.zeros((gt_height, gt_width), dtype=pred_arr.dtype)

                reproject(
                    source=pred_arr,
                    destination=aligned,
                    src_transform=pred_src.transform,
                    src_crs=pred_src.crs,
                    dst_transform=gt_src.transform,
                    dst_crs=gt_src.crs,
                    resampling=Resampling.nearest,
                )

            else:
                raise ValueError(
                    "Prediction and GT shapes differ, but one or both rasters do not "
                    "have valid georeferencing.\n"
                    f"Pred: {pred_path}\n"
                    f"  shape={pred_arr.shape}, crs={pred_src.crs}, transform={pred_src.transform}\n"
                    f"GT:   {gt_path}\n"
                    f"  shape=({gt_height}, {gt_width}), crs={gt_src.crs}, transform={gt_src.transform}"
                )

    return (aligned > pred_threshold).astype(bool)

def list_tifs(folder: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def parse_pred_arg(pred_arg: str):
    """
    Parse name=folder.
    Example:
      pseudo_rgb=/path/to/preds
    """
    if "=" not in pred_arg:
        raise ValueError(
            f"Prediction argument must be name=folder, got: {pred_arg}"
        )

    name, folder = pred_arg.split("=", 1)
    return name.strip(), Path(folder.strip())


def read_gt_mask(gt_path: Path, gt_threshold: float):
    with rasterio.open(gt_path) as src:
        arr = src.read(1)
        nodata = src.nodata

        valid = np.isfinite(arr)

        if nodata is not None:
            valid &= arr != nodata

        mask = arr > gt_threshold

    return mask.astype(bool), valid.astype(bool)


def read_pred_mask_aligned(pred_path: Path, gt_path: Path, pred_threshold: float):
    """
    Read prediction mask and align to GT grid if needed.
    Uses nearest-neighbor resampling, appropriate for binary masks.
    """

    with rasterio.open(gt_path) as gt_src:
        gt_crs = gt_src.crs
        gt_transform = gt_src.transform
        gt_height = gt_src.height
        gt_width = gt_src.width

        with rasterio.open(pred_path) as pred_src:
            pred_arr = pred_src.read(1)

            same_grid = (
                pred_src.crs == gt_crs
                and pred_src.transform == gt_transform
                and pred_src.height == gt_height
                and pred_src.width == gt_width
            )

            if same_grid:
                aligned = pred_arr
            else:
                aligned = np.zeros((gt_height, gt_width), dtype=pred_arr.dtype)

                reproject(
                    source=pred_arr,
                    destination=aligned,
                    src_transform=pred_src.transform,
                    src_crs=pred_src.crs,
                    dst_transform=gt_transform,
                    dst_crs=gt_crs,
                    resampling=Resampling.nearest,
                )

    return (aligned > pred_threshold).astype(bool)


def build_pred_index(pred_dir: Path):
    pred_files = list_tifs(pred_dir)
    return {p.stem: p for p in pred_files}

import re


def extract_numeric_id(stem: str):
    """
    Extract the main numeric ID from filenames like:
      mask_1234
      chip_1234_mask
      chip_000123_mask
    Returns normalized string without leading zeros.
    """
    nums = re.findall(r"\d+", stem)

    if not nums:
        return None

    # Usually the chip ID is the last/only number.
    # Normalize leading zeros: 000123 -> 123
    return str(int(nums[-1]))


def find_matching_prediction(gt_path: Path, pred_index: dict):
    """
    Match GT masks to prediction masks.

    Handles:
      mask_1234.tif       -> chip_1234_mask.tif
      mask_1234.tif       -> 1234.tif
      chip_1234.tif       -> chip_1234_mask.tif
      image_1234_mask.tif -> image_1234_pred.tif
    """

    gt_stem = gt_path.stem

    # 1. Exact stem match
    if gt_stem in pred_index:
        return pred_index[gt_stem]

    # 2. Common suffix/prefix cases
    candidates = [
        f"{gt_stem}_pred",
        f"{gt_stem}_mask",
        f"{gt_stem}_binary",
        f"{gt_stem}_prediction",
        f"{gt_stem}_seg",
    ]

    for c in candidates:
        if c in pred_index:
            return pred_index[c]

    # 3. Match by numeric ID
    gt_id = extract_numeric_id(gt_stem)

    if gt_id is not None:
        id_matches = []

        for pred_stem, pred_path in pred_index.items():
            pred_id = extract_numeric_id(pred_stem)

            if pred_id == gt_id:
                id_matches.append(pred_path)

        if len(id_matches) == 1:
            return id_matches[0]

        elif len(id_matches) > 1:
            print(
                f"WARNING: multiple prediction matches for {gt_path.name}: "
                f"{[p.name for p in id_matches]}"
            )
            return id_matches[0]

    # 4. Fallback: prediction stem starts with GT stem
    startswith_matches = [
        p for stem, p in pred_index.items()
        if stem.startswith(gt_stem)
    ]

    if len(startswith_matches) == 1:
        return startswith_matches[0]

    return None

def find_matching_prediction_old(gt_path: Path, pred_index: dict):
    """
    Attempts to match GT file to prediction file by stem.

    Handles common cases:
      image_001.tif -> image_001.tif
      image_001.tif -> image_001_pred.tif
      image_001.tif -> image_001_mask.tif
    """

    gt_stem = gt_path.stem

    if gt_stem in pred_index:
        return pred_index[gt_stem]

    candidates = [
        f"{gt_stem}_pred",
        f"{gt_stem}_mask",
        f"{gt_stem}_binary",
        f"{gt_stem}_prediction",
        f"{gt_stem}_seg",
    ]

    for c in candidates:
        if c in pred_index:
            return pred_index[c]

    startswith_matches = [
        p for stem, p in pred_index.items()
        if stem.startswith(gt_stem)
    ]

    if len(startswith_matches) == 1:
        return startswith_matches[0]

    return None


def compute_counts(pred_mask, gt_mask, valid_mask):
    pred = pred_mask & valid_mask
    gt = gt_mask & valid_mask

    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.int64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.int64)
    tn = np.logical_and(~pred, ~gt).sum(dtype=np.int64)

    return int(tp), int(fp), int(fn), int(tn)

def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-9

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }

def evaluate_one_prediction_folder(
    gt_dir: Path,
    pred_dir: Path,
    scenario_name: str,
    out_dir: Path,
    gt_threshold: float,
    pred_threshold: float,
):
    gt_files = list_tifs(gt_dir)
    pred_index = build_pred_index(pred_dir)

    if len(gt_files) == 0:
        raise RuntimeError(f"No GT masks found in {gt_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = out_dir / f"{scenario_name}_per_image.csv"

    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    missing = 0

    for gt_path in gt_files:
        pred_path = find_matching_prediction(gt_path, pred_index)

        if pred_path is None:
            missing += 1
            print(f"[{scenario_name}] missing prediction for {gt_path.name}")
            continue

        gt_mask, valid_mask = read_gt_mask(
            gt_path=gt_path,
            gt_threshold=gt_threshold,
        )

        pred_mask = read_pred_mask_aligned(
            pred_path=pred_path,
            gt_path=gt_path,
            pred_threshold=pred_threshold,
        )

        tp, fp, fn, tn = compute_counts(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            valid_mask=valid_mask,
        )

        metrics = metrics_from_counts(tp, fp, fn, tn)

        gt_fg_px = int((gt_mask & valid_mask).sum())
        pred_fg_px = int((pred_mask & valid_mask).sum())

        row = {
            "scenario": scenario_name,
            "gt_file": gt_path.name,
            "pred_file": pred_path.name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "gt_fg_px": gt_fg_px,
            "pred_fg_px": pred_fg_px,
            **metrics,
        }

        rows.append(row)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    with open(per_image_csv, "w", newline="") as f:
        fieldnames = [
            "scenario",
            "gt_file",
            "pred_file",
            "tp",
            "fp",
            "fn",
            "tn",
            "gt_fg_px",
            "pred_fg_px",
            "precision",
            "recall",
            "f1",
            "iou",
            "accuracy",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    micro_metrics = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    if rows:
        macro_precision = float(np.mean([r["precision"] for r in rows]))
        macro_recall = float(np.mean([r["recall"] for r in rows]))
        macro_f1 = float(np.mean([r["f1"] for r in rows]))
        macro_iou = float(np.mean([r["iou"] for r in rows]))
        macro_accuracy = float(np.mean([r["accuracy"] for r in rows]))
    else:
        macro_precision = macro_recall = macro_f1 = macro_iou = macro_accuracy = 0.0

    summary = {
        "scenario": scenario_name,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "num_gt": len(gt_files),
        "num_pred": len(pred_index),
        "num_evaluated": len(rows),
        "num_missing": missing,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "micro_precision": micro_metrics["precision"],
        "micro_recall": micro_metrics["recall"],
        "micro_f1": micro_metrics["f1"],
        "micro_iou": micro_metrics["iou"],
        "micro_accuracy": micro_metrics["accuracy"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "macro_accuracy": macro_accuracy,
        "per_image_csv": str(per_image_csv),
    }

    print(
        f"[{scenario_name}] "
        f"micro F1={summary['micro_f1']:.4f}, "
        f"IoU={summary['micro_iou']:.4f}, "
        f"P={summary['micro_precision']:.4f}, "
        f"R={summary['micro_recall']:.4f}, "
        f"evaluated={summary['num_evaluated']}, "
        f"missing={summary['num_missing']}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple TCN prediction folders against GT masks."
    )

    parser.add_argument(
        "--gt-dir",
        required=True,
        type=Path,
        help="Folder containing ground-truth binary masks.",
    )

    parser.add_argument(
        "--pred",
        required=True,
        action="append",
        help=(
            "Prediction folder in name=path format. "
            "Can be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output folder for CSV reports.",
    )

    parser.add_argument(
        "--gt-threshold",
        type=float,
        default=0,
        help="GT pixels > threshold are treated as foreground.",
    )

    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0,
        help="Prediction pixels > threshold are treated as foreground.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for pred_arg in args.pred:
        scenario_name, pred_dir = parse_pred_arg(pred_arg)

        summary = evaluate_one_prediction_folder(
            gt_dir=args.gt_dir,
            pred_dir=pred_dir,
            scenario_name=scenario_name,
            out_dir=args.out_dir,
            gt_threshold=args.gt_threshold,
            pred_threshold=args.pred_threshold,
        )

        summaries.append(summary)

    summary_csv = args.out_dir / "summary_metrics.csv"

    fieldnames = [
        "scenario",
        "gt_dir",
        "pred_dir",
        "num_gt",
        "num_pred",
        "num_evaluated",
        "num_missing",
        "tp",
        "fp",
        "fn",
        "tn",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_iou",
        "micro_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_iou",
        "macro_accuracy",
        "per_image_csv",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()

"""
USAGE
conda activate /scratch2/projects/PDG_shared/CONDA_ENV/segformer-env-gpkg3
python -m inference.segf_inference_tcn \
    --input_dir=/scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/val_1024/images/ \
    --output_dir=/scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/inf/val_org

python compute_inf_metrics.py  \
--gt-dir /scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/val_mask/ --pred original_rgb=/scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/inf/val_org --out-dir ./metrics

python compute_inf_metrics.py \
  --gt-dir /home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/org_testset/test/masks \
  --pred original_rgb=/home1/09208/asperera/PDG_shared2/TCN_Training_GRAY/inf_geo_test321 \
  --out-dir ./metrics_geo_test321
    
python compute_inf_metrics.py --gt-dir /scratch2/projects/PDG_shared/TCN_Training/test_1024/masks \
 --pred original_rgb=/scratch2/projects/PDG_shared/TCN_Training_GRAY/inf \
 --out-dir ./metrics_test_only

python compute_inf_metrics.py --gt-dir /scratch2/projects/PDG_shared/TCN_Training/test_1024/masks \
 --pred original_rgb=/scratch2/projects/PDG_shared/TCN_Training_GRAY/inf8band \
 --out-dir ./metrics_inf_8band

python compute_inf_metrics.py --gt-dir /scratch2/projects/PDG_shared/TCN_Training/TCN_train_All_bands/masks/ \
--pred original_rgb=/scratch2/projects/PDG_shared/TCN_Training_GRAY/inf_gray_rgb/ --out-dir ./metrics_inf_g_rgb

"""
