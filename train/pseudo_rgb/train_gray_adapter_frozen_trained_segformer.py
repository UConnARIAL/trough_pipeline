#!/usr/bin/env python3

"""
train_gray_adapter_from_inference_model.py

Train a learnable grayscale-to-3-band adapter in front of an existing trained
SegFormer inference model.

Main idea
---------
Instead of recreating the SegFormer architecture manually, this script imports
your existing inference script and calls its own load_model() function.

Pipeline:

    gray image chip
        -> learnable GrayToBandsAdapter
        -> 3-channel [8,6,1]-like tensor
        -> frozen SegFormer loaded from existing inference code
        -> segmentation logits
        -> mask loss

The SegFormer is frozen. Only the adapter is trained.

Why this version is safer
-------------------------
If your original training/inference code had small differences in:

    - ENCODER name
    - checkpoint loading
    - num_labels
    - model configuration
    - device handling
    - hard-coded MODEL_PATH
    - HuggingFace SegFormer construction

then recreating the model in a new script can accidentally create a mismatch.

This script instead uses:

    --inference-script /path/to/your_existing_inference.py
    --load-model-fn load_model

and calls that function directly.

Expected input structure
------------------------

This script assumes you already chipped the data to 1024 x 1024, with matching
filenames across gray images, masks, and original multiband reference images.

Example:

    chipped_1024_split/
        train/
            gray/
                tileA_1.tif
                tileA_2.tif
            masks/
                tileA_1.tif
                tileA_2.tif
            original/
                tileA_1.tif
                tileA_2.tif

        val/
            gray/
            masks/
            original/

        test/
            gray/
            masks/
            original/

The original folder is used only to estimate the raw value range of the model's
expected [8,6,1] input bands. It can also be used for an optional weak
reconstruction loss.

Output
------

The script saves:

    output_dir/
        adapter_best.pt
        adapter_last.pt
        training_log.csv

        best_inspection/
            adapter_images/
                tileA_1.tif       # learned 3-band adapter output
            pred_masks/
                tileA_1.tif       # predicted binary mask
            gt_masks/
                tileA_1.tif       # copied/written ground-truth mask
            prob_masks/           # optional float32 probability raster

Adapter output band order
-------------------------

Inside PyTorch, the adapter output is always:

    channel 0 = band8-like
    channel 1 = band6-like
    channel 2 = band1-like

For saved inspection GeoTIFFs, default --export-write-order model321 writes:

    raster band 1 = band1-like
    raster band 2 = band6-like
    raster band 3 = band8-like

So if your old inference fallback uses:

    src.read([3,2,1])

then it receives:

    [band8-like, band6-like, band1-like]

This matches the model trained on [8,6,1].
"""

from pathlib import Path
import argparse
import csv
import importlib.util
import random
import shutil
import sys

import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Basic raster utilities
# -----------------------------------------------------------------------------

def collect_paths(directory, patterns=("*.tif", "*.tiff", "*.img")):
    directory = Path(directory)
    paths = []
    for pattern in patterns:
        paths.extend(sorted(directory.glob(pattern)))

    seen = set()
    unique = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def index_by_stem(paths):
    return {p.stem: p for p in paths}


def find_matching_path(reference_path, candidate_dir, candidate_index):
    candidate_dir = Path(candidate_dir)

    exact = candidate_dir / reference_path.name
    if exact.exists():
        return exact

    by_stem = candidate_index.get(reference_path.stem)
    if by_stem is not None:
        return by_stem

    raise FileNotFoundError(
        f"No matching file for {reference_path.name} in {candidate_dir}"
    )


def read_single_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = 0

    return arr


def read_multiband(path, indexes):
    with rasterio.open(path) as src:
        if max(indexes) > src.count:
            raise ValueError(
                f"{path} has {src.count} bands, but requested indexes={indexes}"
            )

        arr = src.read(indexes).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = 0

    return arr


def normalize_gray(gray):
    """
    Normalize gray image to 0-1 for the adapter input.

    This is only for the adapter. The adapter output is mapped back into the
    raw [8,6,1]-like band range expected by the frozen SegFormer.
    """
    valid = np.isfinite(gray) & (gray > 0)

    if np.count_nonzero(valid) < 10:
        return np.zeros_like(gray, dtype=np.float32)

    p2, p98 = np.percentile(gray[valid], [2, 98])

    if p98 <= p2:
        return np.zeros_like(gray, dtype=np.float32)

    out = (gray - p2) / (p98 - p2)
    out = np.clip(out, 0, 1)
    out[~valid] = 0

    return out.astype(np.float32)


def write_single_band_geotiff_like(src_path, out_path, arr, dtype, nodata=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile.update(nodata=nodata)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def write_adapter_geotiff_like(
    gray_src_path,
    out_path,
    adapter_out_chw,
    export_write_order="model321",
):
    """
    adapter_out_chw is [3,H,W] in normal model order:
        [band8-like, band6-like, band1-like]

    model321 output is compatible with old inference fallback src.read([3,2,1]).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_out_chw = np.clip(np.rint(adapter_out_chw), 0, 65535).astype(np.uint16)

    if export_write_order == "model321":
        stack = np.stack(
            [
                adapter_out_chw[2],  # raster band 1 = band1-like
                adapter_out_chw[1],  # raster band 2 = band6-like
                adapter_out_chw[0],  # raster band 3 = band8-like
            ],
            axis=0,
        )
        descriptions = [
            "adapter_band1_like_written_as_band1",
            "adapter_band6_like_written_as_band2",
            "adapter_band8_like_written_as_band3",
        ]
        note = "src.read([3,2,1]) returns [band8-like,band6-like,band1-like]."

    elif export_write_order == "normal123":
        stack = adapter_out_chw
        descriptions = [
            "adapter_band8_like_written_as_band1",
            "adapter_band6_like_written_as_band2",
            "adapter_band1_like_written_as_band3",
        ]
        note = "src.read([1,2,3]) returns [band8-like,band6-like,band1-like]."

    else:
        raise ValueError("export_write_order must be 'model321' or 'normal123'")

    with rasterio.open(gray_src_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=3,
        dtype="uint16",
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("nodata", None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)

        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)

        dst.update_tags(
            adapter_output="true",
            intended_model_input="8,6,1-like",
            export_write_order=export_write_order,
            note=note,
        )


# -----------------------------------------------------------------------------
# Reference range estimation
# -----------------------------------------------------------------------------

def compute_ref_ranges(
    ref_dir,
    indexes=(8, 6, 1),
    max_files=100,
    max_pixels_per_file=100000,
    low_pct=1.0,
    high_pct=99.0,
    seed=42,
):
    """
    Estimate per-band raw-value output range for adapter.

    For model trained on [8,6,1], this returns lows/highs for:

        channel 0 -> band 8
        channel 1 -> band 6
        channel 2 -> band 1
    """
    ref_paths = collect_paths(ref_dir)

    if len(ref_paths) == 0:
        raise ValueError(f"No reference images found in {ref_dir}")

    ref_paths = ref_paths[:max_files]

    rng = np.random.default_rng(seed)
    samples = [[] for _ in indexes]

    for path in ref_paths:
        arr = read_multiband(path, indexes)
        valid = np.all(np.isfinite(arr), axis=0)
        valid &= np.all(arr > 0, axis=0)

        ys, xs = np.where(valid)
        if len(xs) == 0:
            continue

        n = min(max_pixels_per_file, len(xs))
        pick = rng.choice(len(xs), size=n, replace=False)

        for c in range(len(indexes)):
            samples[c].append(arr[c, ys[pick], xs[pick]])

    lows = []
    highs = []

    print("\nEstimated target value ranges from original reference images:")

    for c, band_index in enumerate(indexes):
        if len(samples[c]) == 0:
            raise ValueError(f"No valid samples collected for band {band_index}")

        vals = np.concatenate(samples[c]).astype(np.float32)
        lo, hi = np.percentile(vals, [low_pct, high_pct])

        lows.append(float(lo))
        highs.append(float(hi))

        print(
            f"  original band {band_index}: "
            f"p{low_pct:g}={lo:.3f}, "
            f"median={np.median(vals):.3f}, "
            f"p{high_pct:g}={hi:.3f}, "
            f"mean={np.mean(vals):.3f}, std={np.std(vals):.3f}"
        )

    return torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class GrayAdapterChipDataset(Dataset):
    def __init__(
        self,
        gray_dir,
        mask_dir,
        ref_dir=None,
        ref_indexes=(8, 6, 1),
        augment=False,
        patterns=("*.tif", "*.tiff", "*.img"),
    ):
        self.gray_dir = Path(gray_dir)
        self.mask_dir = Path(mask_dir)
        self.ref_dir = Path(ref_dir) if ref_dir is not None else None
        self.ref_indexes = list(ref_indexes)
        self.augment = augment

        self.gray_paths = collect_paths(self.gray_dir, patterns)
        if len(self.gray_paths) == 0:
            raise ValueError(f"No gray images found in {self.gray_dir}")

        self.mask_paths = collect_paths(self.mask_dir, patterns)
        self.mask_index = index_by_stem(self.mask_paths)

        if self.ref_dir is not None:
            self.ref_paths = collect_paths(self.ref_dir, patterns)
            self.ref_index = index_by_stem(self.ref_paths)
        else:
            self.ref_index = {}

    def __len__(self):
        return len(self.gray_paths)

    def _augment(self, gray, mask, ref=None):
        # gray [H,W], mask [H,W], ref [C,H,W]
        if random.random() < 0.5:
            gray = np.flip(gray, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            if ref is not None:
                ref = np.flip(ref, axis=2).copy()

        if random.random() < 0.5:
            gray = np.flip(gray, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
            if ref is not None:
                ref = np.flip(ref, axis=1).copy()

        k = random.randint(0, 3)
        if k > 0:
            gray = np.rot90(gray, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            if ref is not None:
                ref = np.rot90(ref, k, axes=(1, 2)).copy()

        return gray, mask, ref

    def __getitem__(self, idx):
        gray_path = self.gray_paths[idx]
        mask_path = find_matching_path(gray_path, self.mask_dir, self.mask_index)

        gray_raw = read_single_band(gray_path)
        gray = normalize_gray(gray_raw)

        mask = read_single_band(mask_path)
        mask = (mask > 0).astype(np.float32)

        ref = None
        if self.ref_dir is not None:
            ref_path = find_matching_path(gray_path, self.ref_dir, self.ref_index)
            ref = read_multiband(ref_path, self.ref_indexes)

        if self.augment:
            gray, mask, ref = self._augment(gray, mask, ref)

        sample = {
            "gray": torch.from_numpy(gray[None, :, :]).float(),
            "mask": torch.from_numpy(mask[None, :, :]).float(),
            "name": gray_path.name,
            "gray_path": str(gray_path),
            "mask_path": str(mask_path),
        }

        if ref is not None:
            sample["ref"] = torch.from_numpy(ref).float()

        return sample


# -----------------------------------------------------------------------------
# Adapter model
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class GrayToBandsAdapter(nn.Module):
    """
    Capacity-adjustable gray -> 3-channel adapter.

    Output channel order inside PyTorch:

        channel 0 = band8-like
        channel 1 = band6-like
        channel 2 = band1-like
    """

    def __init__(
        self,
        band_lows,
        band_highs,
        width=32,
        blocks=2,
        use_dilation=False,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
        ]

        for i in range(blocks):
            dilation = 2 if use_dilation and (i % 2 == 1) else 1
            layers.append(ResidualBlock(width, dilation=dilation))

        layers.extend(
            [
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, 3, 1),
            ]
        )

        self.net = nn.Sequential(*layers)

        self.register_buffer("band_lows", band_lows.view(1, 3, 1, 1))
        self.register_buffer("band_highs", band_highs.view(1, 3, 1, 1))

    def forward(self, gray):
        raw = self.net(gray)
        x01 = torch.sigmoid(raw)
        out = self.band_lows + x01 * (self.band_highs - self.band_lows)
        return out


# -----------------------------------------------------------------------------
# Import existing inference model
# -----------------------------------------------------------------------------

def load_model_from_inference_script(
    inference_script,
    load_model_fn,
    device,
):
    """
    Import user's existing inference script and call its load_model() function.

    The inference script should be import-safe, meaning its CLI/main code should
    be protected by:

        if __name__ == "__main__":
            main()
    """
    inference_script = Path(inference_script).resolve()

    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found: {inference_script}")

    module_name = "user_segformer_inference_module"
    spec = importlib.util.spec_from_file_location(module_name, inference_script)
    module = importlib.util.module_from_spec(spec)

    # Make the inference script's folder importable if it uses local imports.
    sys.path.insert(0, str(inference_script.parent))

    spec.loader.exec_module(module)

    # Many of your scripts use a global DEVICE variable. Set it before calling load_model().
    if hasattr(module, "DEVICE"):
        setattr(module, "DEVICE", device)

    if not hasattr(module, load_model_fn):
        raise AttributeError(
            f"{inference_script} does not contain function {load_model_fn}()"
        )

    loader = getattr(module, load_model_fn)

    print(f"\nLoading frozen SegFormer using:")
    print(f"  script: {inference_script}")
    print(f"  function: {load_model_fn}()")

    model = loader()

    # Some loaders return (model, processor) or similar.
    if isinstance(model, tuple):
        model = model[0]

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Frozen inference model loaded. Parameters: {n_params:,}")

    return model


def forward_segformer(model, pixel_values):
    """
    Robust forward wrapper for HuggingFace-style SegFormer.

    Your current model likely supports:

        model(pixel_values=batch).logits
    """
    try:
        out = model(pixel_values=pixel_values)
    except TypeError:
        out = model(pixel_values)

    if hasattr(out, "logits"):
        return out.logits

    if isinstance(out, dict) and "logits" in out:
        return out["logits"]

    if torch.is_tensor(out):
        return out

    raise RuntimeError("Could not extract logits from SegFormer output.")


# -----------------------------------------------------------------------------
# Losses and metrics
# -----------------------------------------------------------------------------

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits, mask, pos_weight=None):
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            mask,
            pos_weight=pos_weight,
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, mask)

    dloss = dice_loss_from_logits(logits, mask)

    return bce + dloss


def compute_counts_from_logits(logits, mask, threshold):
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    target = mask > 0.5

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return f1, iou, precision, recall


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def run_epoch(
    adapter,
    segformer,
    loader,
    optimizer,
    device,
    train,
    threshold,
    pos_weight,
    lambda_recon,
    amp,
):
    if train:
        adapter.train()
    else:
        adapter.eval()

    segformer.eval()

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in loader:
        gray = batch["gray"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        grad_enabled = train

        with torch.set_grad_enabled(grad_enabled):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=(amp and device.type == "cuda"),
            ):
                pseudo3 = adapter(gray)

                # Important:
                # Do not use torch.no_grad() around the frozen SegFormer during training.
                # We need gradients to flow through it back into the adapter.
                logits = forward_segformer(segformer, pseudo3)

                logits = F.interpolate(
                    logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = segmentation_loss(
                    logits,
                    mask,
                    pos_weight=pos_weight,
                )

                if lambda_recon > 0 and "ref" in batch:
                    ref = batch["ref"].to(device, non_blocking=True)
                    recon = F.smooth_l1_loss(pseudo3, ref)
                    loss = loss + lambda_recon * recon

            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())

        tp, fp, fn = compute_counts_from_logits(
            logits.detach(),
            mask,
            threshold=threshold,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    f1, iou, precision, recall = metrics_from_counts(total_tp, total_fp, total_fn)
    mean_loss = total_loss / max(len(loader), 1)

    return {
        "loss": mean_loss,
        "f1": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def save_training_log_row(log_csv, row):
    log_csv = Path(log_csv)
    write_header = not log_csv.exists()

    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Export best inspection set
# -----------------------------------------------------------------------------

def export_best_inspection_set(
    adapter,
    segformer,
    gray_dir,
    mask_dir,
    out_dir,
    device,
    threshold,
    max_images,
    export_write_order,
    export_probs,
    patterns=("*.tif", "*.tiff", "*.img"),
):
    out_dir = Path(out_dir)
    adapter_img_dir = out_dir / "adapter_images"
    pred_mask_dir = out_dir / "pred_masks"
    gt_mask_dir = out_dir / "gt_masks"
    prob_mask_dir = out_dir / "prob_masks"

    adapter_img_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    gt_mask_dir.mkdir(parents=True, exist_ok=True)

    if export_probs:
        prob_mask_dir.mkdir(parents=True, exist_ok=True)

    gray_paths = collect_paths(gray_dir, patterns)
    mask_paths = collect_paths(mask_dir, patterns)
    mask_index = index_by_stem(mask_paths)

    if max_images > 0:
        gray_paths = gray_paths[:max_images]

    adapter.eval()
    segformer.eval()

    print(f"\nExporting best inspection set to: {out_dir}")
    print(f"  images: {len(gray_paths)}")

    for gray_path in gray_paths:
        mask_path = find_matching_path(gray_path, mask_dir, mask_index)

        gray_raw = read_single_band(gray_path)
        gray_norm = normalize_gray(gray_raw)

        gray_tensor = torch.from_numpy(gray_norm[None, None, :, :]).float().to(device)

        with torch.no_grad():
            pseudo3 = adapter(gray_tensor)
            logits = forward_segformer(segformer, pseudo3)
            logits = F.interpolate(
                logits,
                size=gray_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (probs > threshold).astype(np.uint8)

            pseudo_np = pseudo3[0].detach().cpu().numpy()

        # Save learned 3-band adapter output.
        write_adapter_geotiff_like(
            gray_src_path=gray_path,
            out_path=adapter_img_dir / gray_path.name,
            adapter_out_chw=pseudo_np,
            export_write_order=export_write_order,
        )

        # Save prediction mask.
        write_single_band_geotiff_like(
            src_path=gray_path,
            out_path=pred_mask_dir / gray_path.name,
            arr=pred,
            dtype="uint8",
            nodata=None,
        )

        # Save / copy GT mask with same basename.
        # Copying preserves its original values/style if possible.
        gt_out = gt_mask_dir / gray_path.name
        shutil.copy2(mask_path, gt_out)

        # Optional probability raster.
        if export_probs:
            write_single_band_geotiff_like(
                src_path=gray_path,
                out_path=prob_mask_dir / gray_path.name,
                arr=probs.astype(np.float32),
                dtype="float32",
                nodata=None,
            )

    print("Inspection export complete.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

USAGE = r"""
USAGE

1. Train adapter using exact SegFormer from existing inference script

   python train_gray_adapter_from_inference_model.py \
     --inference-script /path/to/segf_inference_tcn_RGB2.py \
     --load-model-fn load_model \
     --train-gray-dir /path/to/chipped_1024_split/train/gray \
     --train-mask-dir /path/to/chipped_1024_split/train/masks \
     --train-ref-dir  /path/to/chipped_1024_split/train/original \
     --val-gray-dir   /path/to/chipped_1024_split/val/gray \
     --val-mask-dir   /path/to/chipped_1024_split/val/masks \
     --val-ref-dir    /path/to/chipped_1024_split/val/original \
     --output-dir /path/to/adapter_run_v2 \
     --target-indexes 8 6 1 \
     --epochs 30 \
     --batch-size 1 \
     --lr 1e-4 \
     --adapter-width 32 \
     --adapter-blocks 2 \
     --threshold 0.05 \
     --export-max-images 25 \
     --export-probs

2. Increase adapter capacity

   Try this after the small adapter is stable:

     --adapter-width 64 --adapter-blocks 4 --use-dilation

3. Optional weak reconstruction loss against original [8,6,1]

   Start small:

     --lambda-recon 0.001

   or:

     --lambda-recon 0.01

   Keep this weak. The goal is segmentation performance, not perfect
   reconstruction of [8,6,1].

4. Export all validation chips for inspection

   Use:

     --export-max-images 0

5. Output adapter image order

   Default:

     --export-write-order model321

   This means saved adapter images can be read by old fallback logic:

     src.read([3,2,1]) -> [band8-like, band6-like, band1-like]

   For simpler visual/debug reading, use:

     --export-write-order normal123

   Then:

     src.read([1,2,3]) -> [band8-like, band6-like, band1-like]

IMPORTANT

The inference script must be import-safe. Its command-line execution should be
inside:

    if __name__ == "__main__":
        main()

If importing the inference script starts running inference immediately, move the
load_model() and ensure_model_weights() functions into a small model_loader.py
file and import that instead.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Train gray-to-3-band adapter using exact model from existing inference script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE,
    )

    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--load-model-fn", default="load_model")

    parser.add_argument("--train-gray-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--train-ref-dir", type=Path, required=True)

    parser.add_argument("--val-gray-dir", type=Path, required=True)
    parser.add_argument("--val-mask-dir", type=Path, required=True)
    parser.add_argument("--val-ref-dir", type=Path, required=True)

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--target-indexes", nargs=3, type=int, default=[8, 6, 1])

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--adapter-width", type=int, default=32)
    parser.add_argument("--adapter-blocks", type=int, default=2)
    parser.add_argument("--use-dilation", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.0)

    parser.add_argument("--range-low-pct", type=float, default=1.0)
    parser.add_argument("--range-high-pct", type=float, default=99.0)
    parser.add_argument("--max-ref-files", type=int, default=100)
    parser.add_argument("--max-ref-pixels-per-file", type=int, default=100000)

    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--export-max-images",
        type=int,
        default=25,
        help="Number of validation images to export for inspection. Use 0 for all.",
    )

    parser.add_argument(
        "--export-write-order",
        choices=["model321", "normal123"],
        default="model321",
    )

    parser.add_argument("--export-probs", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load exact frozen model from existing inference code.
    segformer = load_model_from_inference_script(
        inference_script=args.inference_script,
        load_model_fn=args.load_model_fn,
        device=device,
    )

    # Estimate adapter output range from original multiband training chips.
    band_lows, band_highs = compute_ref_ranges(
        ref_dir=args.train_ref_dir,
        indexes=tuple(args.target_indexes),
        max_files=args.max_ref_files,
        max_pixels_per_file=args.max_ref_pixels_per_file,
        low_pct=args.range_low_pct,
        high_pct=args.range_high_pct,
    )

    band_lows = band_lows.to(device)
    band_highs = band_highs.to(device)

    train_ds = GrayAdapterChipDataset(
        gray_dir=args.train_gray_dir,
        mask_dir=args.train_mask_dir,
        ref_dir=args.train_ref_dir if args.lambda_recon > 0 else None,
        ref_indexes=args.target_indexes,
        augment=True,
    )

    val_ds = GrayAdapterChipDataset(
        gray_dir=args.val_gray_dir,
        mask_dir=args.val_mask_dir,
        ref_dir=args.val_ref_dir if args.lambda_recon > 0 else None,
        ref_indexes=args.target_indexes,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    adapter = GrayToBandsAdapter(
        band_lows=band_lows,
        band_highs=band_highs,
        width=args.adapter_width,
        blocks=args.adapter_blocks,
        use_dilation=args.use_dilation,
    ).to(device)

    n_adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"\nAdapter trainable parameters: {n_adapter_params:,}")
    print(f"Adapter width: {args.adapter_width}")
    print(f"Adapter blocks: {args.adapter_blocks}")
    print(f"Use dilation: {args.use_dilation}")

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)

    best_f1 = -1.0
    log_csv = args.output_dir / "training_log.csv"

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            adapter=adapter,
            segformer=segformer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            threshold=args.threshold,
            pos_weight=pos_weight,
            lambda_recon=args.lambda_recon,
            amp=args.amp,
        )

        val_m = run_epoch(
            adapter=adapter,
            segformer=segformer,
            loader=val_loader,
            optimizer=None,
            device=device,
            train=False,
            threshold=args.threshold,
            pos_weight=pos_weight,
            lambda_recon=args.lambda_recon,
            amp=args.amp,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_m['loss']:.4f} "
            f"F1={train_m['f1']:.4f} IoU={train_m['iou']:.4f} "
            f"P={train_m['precision']:.4f} R={train_m['recall']:.4f} | "
            f"val loss={val_m['loss']:.4f} "
            f"F1={val_m['f1']:.4f} IoU={val_m['iou']:.4f} "
            f"P={val_m['precision']:.4f} R={val_m['recall']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_f1": train_m["f1"],
            "train_iou": train_m["iou"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "val_loss": val_m["loss"],
            "val_f1": val_m["f1"],
            "val_iou": val_m["iou"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "threshold": args.threshold,
            "adapter_width": args.adapter_width,
            "adapter_blocks": args.adapter_blocks,
            "use_dilation": args.use_dilation,
            "lambda_recon": args.lambda_recon,
        }
        save_training_log_row(log_csv, row)

        ckpt = {
            "epoch": epoch,
            "adapter_state_dict": adapter.state_dict(),
            "band_lows": band_lows.detach().cpu(),
            "band_highs": band_highs.detach().cpu(),
            "target_indexes": args.target_indexes,
            "adapter_width": args.adapter_width,
            "adapter_blocks": args.adapter_blocks,
            "use_dilation": args.use_dilation,
            "threshold": args.threshold,
            "val_f1": val_m["f1"],
            "val_iou": val_m["iou"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "inference_script": str(args.inference_script),
            "load_model_fn": args.load_model_fn,
        }

        torch.save(ckpt, args.output_dir / "adapter_last.pt")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(ckpt, args.output_dir / "adapter_best.pt")
            print(f"Saved new best adapter: F1={best_f1:.4f}")

    # Reload best adapter before final export.
    best_ckpt = torch.load(args.output_dir / "adapter_best.pt", map_location=device)
    adapter.load_state_dict(best_ckpt["adapter_state_dict"])
    adapter.eval()

    export_best_inspection_set(
        adapter=adapter,
        segformer=segformer,
        gray_dir=args.val_gray_dir,
        mask_dir=args.val_mask_dir,
        out_dir=args.output_dir / "best_inspection",
        device=device,
        threshold=args.threshold,
        max_images=args.export_max_images,
        export_write_order=args.export_write_order,
        export_probs=args.export_probs,
    )

    print("\nDone.")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Saved best checkpoint: {args.output_dir / 'adapter_best.pt'}")
    print(f"Saved inspection outputs: {args.output_dir / 'best_inspection'}")


if __name__ == "__main__":
    main()

"""
USAGE
CUDA_VISIBLE_DEVICES=1 python train_gray_adapter_frozen_trained_segformer.py \
  --inference-script /home1/09208/asperera/PDG_shared2/TCN_mosaic_pipline/git_ver/inference/segf_inference_tcn.py \
  --train-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/gray \
  --train-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/masks \
  --train-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/original \
  --val-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/gray \
  --val-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/masks \
  --val-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/original \
  --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/gray_adapter_w32_b2 \
  --target-indexes 8 6 1 \
  --epochs 30 \
  --batch-size 1 \
  --lr 1e-4 \
  --adapter-width 32 \
  --adapter-blocks 2 \
  --threshold 0.05 \
  --export-max-images 25 \
  --export-probs

python train_gray_adapter_frozen_trained_segformer.py \
  --inference-script /home1/09208/asperera/PDG_shared2/TCN_mosaic_pipline/git_ver/inference/segf_inference_tcn.py \
  --train-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/gray \
  --train-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/masks \
  --train-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/original \
  --val-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/gray \
  --val-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/masks \
  --val-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/original \
  --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/gray_adapter_w64_b4 \
  --target-indexes 8 6 1 \
  --epochs 30 \
  --batch-size 1 \
  --lr 1e-4 \
  --adapter-width 64 \
  --adapter-blocks 4 \
  --use-dilation \
  --threshold 0.05 \
  --export-max-images 25 \
  --export-probs

"""
