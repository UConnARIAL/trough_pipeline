#!/usr/bin/env python3

"""
train_gray_adapter_frozen_segformer.py

Train a small learnable adapter that converts grayscale imagery into a
3-band representation usable by an already-trained SegFormer model.

Pipeline:

    gray image
        -> GrayToBandsAdapter
        -> 3-band [8,6,1]-like tensor
        -> frozen SegFormer
        -> segmentation logits
        -> mask loss

The SegFormer weights are frozen. Only the adapter is trained.

This is intended to test whether a learned gray->3-band mapping can improve
over the handcrafted pseudo-RGB baseline.

Expected input:
    gray_dir: 1-band gray images
    mask_dir: binary masks
    ref_dir: original multiband images used only to estimate [8,6,1]
             value ranges and optionally add a weak reconstruction loss.

Important:
    Keep preprocessing consistent with the original SegFormer training/inference.
    If the trained model expects raw uint16-like values, keep raw scale.
    If it expects normalized values, apply the same normalization in
    preprocess_for_segformer().
"""

from pathlib import Path
import argparse
import random
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation


# -----------------------------
# Utilities
# -----------------------------

def save_adapter_output_geotiff(adapter, gray_path, out_path, device, band_descriptions=None):
    """
    Save adapter-produced 3-band output as a GeoTIFF.

    Input:
        gray_path = 1-band grayscale GeoTIFF
        out_path  = output 3-band adapter GeoTIFF

    Output order:
        band 1 = adapter output channel 1, intended band8-like
        band 2 = adapter output channel 2, intended band6-like
        band 3 = adapter output channel 3, intended band1-like

    If your inference reads adapter outputs directly with [1,2,3], this is fine.
    If you want compatibility with old [3,2,1] fallback logic, we can reverse-write it.
    """

    adapter.eval()
    gray_path = Path(gray_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(gray_path) as src:
        gray = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    gray_norm = normalize_gray(gray)

    gray_tensor = torch.from_numpy(gray_norm[None, None, :, :]).float().to(device)

    with torch.no_grad():
        pred3 = adapter(gray_tensor)[0].detach().cpu().numpy()  # [3, H, W]

    pred3 = np.clip(np.rint(pred3), 0, 65535).astype(np.uint16)

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
        dst.write(pred3)

        if band_descriptions is None:
            band_descriptions = [
                "adapter_band8_like",
                "adapter_band6_like",
                "adapter_band1_like",
            ]

        for i, desc in enumerate(band_descriptions, start=1):
            dst.set_band_description(i, desc)

        dst.update_tags(
            adapter_output="true",
            intended_model_input="8,6,1-like",
            note="Channels are written in normal order [band8-like, band6-like, band1-like].",
        )

    print(f"Saved adapter output: {out_path}")

def find_matching_file(directory, name):
    directory = Path(directory)
    exact = directory / name
    if exact.exists():
        return exact

    stem = Path(name).stem
    matches = list(directory.glob(stem + ".*"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No match for {name} in {directory}")
    return matches[0]


def read_band(path, band_index=1):
    with rasterio.open(path) as src:
        arr = src.read(band_index).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = 0

    return arr


def read_multiband(path, indexes):
    with rasterio.open(path) as src:
        arr = src.read(indexes).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = 0

    return arr


def normalize_gray(gray):
    """
    Normalize gray input to 0-1 for the adapter.
    This does not control the SegFormer input scale;
    the adapter output does that.
    """
    valid = gray > 0
    if valid.sum() < 10:
        return np.zeros_like(gray, dtype=np.float32)

    p2, p98 = np.percentile(gray[valid], [2, 98])
    if p98 <= p2:
        return np.zeros_like(gray, dtype=np.float32)

    out = (gray - p2) / (p98 - p2)
    out = np.clip(out, 0, 1)
    out[~valid] = 0
    return out.astype(np.float32)


def compute_ref_ranges(ref_dir, indexes=(8, 6, 1), max_files=50, max_pixels=100000):
    """
    Estimate p01/p99 ranges for the original model input bands [8,6,1].
    Adapter output will be constrained to these ranges.
    """
    ref_dir = Path(ref_dir)
    paths = sorted(list(ref_dir.glob("*.tif")) + list(ref_dir.glob("*.tiff")) + list(ref_dir.glob("*.img")))

    if len(paths) == 0:
        raise ValueError(f"No reference images found in {ref_dir}")

    paths = paths[:max_files]
    rng = np.random.default_rng(42)

    samples = [[] for _ in indexes]

    for path in paths:
        arr = read_multiband(path, indexes)
        valid = np.all(arr > 0, axis=0)

        ys, xs = np.where(valid)
        if len(xs) == 0:
            continue

        n = min(max_pixels, len(xs))
        pick = rng.choice(len(xs), size=n, replace=False)

        for c in range(len(indexes)):
            samples[c].append(arr[c, ys[pick], xs[pick]])

    lows = []
    highs = []

    for c, idx in enumerate(indexes):
        vals = np.concatenate(samples[c])
        lo, hi = np.percentile(vals, [1, 99])
        lows.append(lo)
        highs.append(hi)
        print(f"Band {idx}: p01={lo:.3f}, p99={hi:.3f}, median={np.median(vals):.3f}")

    return torch.tensor(lows).float(), torch.tensor(highs).float()


# -----------------------------
# Dataset
# -----------------------------

class GraySegDataset(Dataset):
    def __init__(
        self,
        gray_dir,
        mask_dir,
        ref_dir=None,
        ref_indexes=(8, 6, 1),
        chip_size=1024,
        samples_per_epoch=2000,
        random_crop=True,
    ):
        self.gray_dir = Path(gray_dir)
        self.mask_dir = Path(mask_dir)
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.ref_indexes = list(ref_indexes)
        self.chip_size = chip_size
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop

        self.gray_paths = sorted(
            list(self.gray_dir.glob("*.tif")) +
            list(self.gray_dir.glob("*.tiff")) +
            list(self.gray_dir.glob("*.img"))
        )

        if len(self.gray_paths) == 0:
            raise ValueError(f"No gray images found in {self.gray_dir}")

    def __len__(self):
        return self.samples_per_epoch

    def _crop(self, arr, x, y):
        return arr[..., y:y+self.chip_size, x:x+self.chip_size]

    def __getitem__(self, idx):
        gray_path = random.choice(self.gray_paths)
        mask_path = find_matching_file(self.mask_dir, gray_path.name)

        gray = read_band(gray_path, 1)
        mask = read_band(mask_path, 1)
        mask = (mask > 0).astype(np.float32)

        h, w = gray.shape
        cs = self.chip_size

        if h < cs or w < cs:
            raise ValueError(f"{gray_path} is smaller than chip_size={cs}: {gray.shape}")

        if self.random_crop:
            x = random.randint(0, w - cs)
            y = random.randint(0, h - cs)
        else:
            x = max((w - cs) // 2, 0)
            y = max((h - cs) // 2, 0)

        gray = gray[y:y+cs, x:x+cs]
        mask = mask[y:y+cs, x:x+cs]

        gray = normalize_gray(gray)

        sample = {
            "gray": torch.from_numpy(gray[None, :, :]).float(),
            "mask": torch.from_numpy(mask[None, :, :]).float(),
        }

        if self.ref_dir is not None:
            ref_path = find_matching_file(self.ref_dir, gray_path.name)
            ref = read_multiband(ref_path, self.ref_indexes)
            ref = self._crop(ref, x, y)
            sample["ref"] = torch.from_numpy(ref).float()

        return sample


# -----------------------------
# Adapter model
# -----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class GrayToBandsAdapter(nn.Module):
    """
    Converts 1-channel normalized gray input into raw-scale 3-channel output.

    Output is constrained between per-band lo/hi values estimated from
    the original [8,6,1] training imagery.
    """
    def __init__(self, band_lows, band_highs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

        self.register_buffer("band_lows", band_lows.view(1, 3, 1, 1))
        self.register_buffer("band_highs", band_highs.view(1, 3, 1, 1))

    def forward(self, gray):
        x = self.net(gray)
        x01 = torch.sigmoid(x)
        out = self.band_lows + x01 * (self.band_highs - self.band_lows)
        return out


# -----------------------------
# Losses and metrics
# -----------------------------

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


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


def compute_batch_metrics(logits, mask, threshold=0.05):
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    target = mask > 0.5

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    return tp, fp, fn


def f1_from_counts(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou, precision, recall


# -----------------------------
# SegFormer loading
# -----------------------------

def load_frozen_segformer(encoder, model_path, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{encoder}",
        num_labels=1,
    )

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove possible DDP prefixes.
    clean_state = {}
    for k, v in state.items():
        k2 = k.replace("module.", "")
        clean_state[k2] = v

    model.load_state_dict(clean_state, strict=False)

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print("Frozen SegFormer loaded.")
    return model


def preprocess_for_segformer(x):
    """
    Put the exact preprocessing used during original SegFormer training here.

    In your current inference code, it looked like:

        batch = batch.to(DEVICE).float()

    So this function currently returns x unchanged.

    If original training used normalization, add it here.
    """
    return x


# -----------------------------
# Training / validation
# -----------------------------

def run_epoch(
    adapter,
    segformer,
    loader,
    optimizer,
    device,
    train=True,
    lambda_recon=0.0,
    pos_weight=None,
    threshold=0.05,
):
    adapter.train(train)

    total_loss = 0.0
    total_tp = total_fp = total_fn = 0

    for batch in loader:
        gray = batch["gray"].to(device)
        mask = batch["mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pseudo3 = adapter(gray)
        model_input = preprocess_for_segformer(pseudo3)

        with torch.set_grad_enabled(train):
            logits = segformer(pixel_values=model_input).logits

            logits = F.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = segmentation_loss(logits, mask, pos_weight=pos_weight)

            if lambda_recon > 0 and "ref" in batch:
                ref = batch["ref"].to(device)
                loss_recon = F.smooth_l1_loss(pseudo3, ref)
                loss = loss + lambda_recon * loss_recon

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        tp, fp, fn = compute_batch_metrics(logits.detach(), mask, threshold=threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    f1, iou, precision, recall = f1_from_counts(total_tp, total_fp, total_fn)
    mean_loss = total_loss / max(len(loader), 1)

    return mean_loss, f1, iou, precision, recall


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-gray-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--train-ref-dir", required=True)

    parser.add_argument("--val-gray-dir", required=True)
    parser.add_argument("--val-mask-dir", required=True)
    parser.add_argument("--val-ref-dir", required=True)

    parser.add_argument("--encoder", required=True, help="Example: mit-b3 or whatever ENCODER you used")
    parser.add_argument("--model-path", required=True, help="Path to trained SegFormer .pth")

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--target-indexes", nargs=3, type=int, default=[8, 6, 1])
    parser.add_argument("--chip-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-recon", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=1.0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    band_lows, band_highs = compute_ref_ranges(
        args.train_ref_dir,
        indexes=tuple(args.target_indexes),
    )

    band_lows = band_lows.to(device)
    band_highs = band_highs.to(device)

    train_ds = GraySegDataset(
        gray_dir=args.train_gray_dir,
        mask_dir=args.train_mask_dir,
        ref_dir=args.train_ref_dir,
        ref_indexes=args.target_indexes,
        chip_size=args.chip_size,
        samples_per_epoch=args.samples_per_epoch,
        random_crop=True,
    )

    val_ds = GraySegDataset(
        gray_dir=args.val_gray_dir,
        mask_dir=args.val_mask_dir,
        ref_dir=args.val_ref_dir,
        ref_indexes=args.target_indexes,
        chip_size=args.chip_size,
        samples_per_epoch=min(500, args.samples_per_epoch),
        random_crop=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    segformer = load_frozen_segformer(
        encoder=args.encoder,
        model_path=args.model_path,
        device=device,
    )

    adapter = GrayToBandsAdapter(band_lows, band_highs).to(device)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    pos_weight = torch.tensor([args.pos_weight], device=device)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, train_iou, train_p, train_r = run_epoch(
            adapter,
            segformer,
            train_loader,
            optimizer,
            device,
            train=True,
            lambda_recon=args.lambda_recon,
            pos_weight=pos_weight,
            threshold=args.threshold,
        )

        val_loss, val_f1, val_iou, val_p, val_r = run_epoch(
            adapter,
            segformer,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            lambda_recon=args.lambda_recon,
            pos_weight=pos_weight,
            threshold=args.threshold,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} F1={train_f1:.4f} IoU={train_iou:.4f} P={train_p:.4f} R={train_r:.4f} | "
            f"val loss={val_loss:.4f} F1={val_f1:.4f} IoU={val_iou:.4f} P={val_p:.4f} R={val_r:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "adapter_state_dict": adapter.state_dict(),
            "band_lows": band_lows.detach().cpu(),
            "band_highs": band_highs.detach().cpu(),
            "target_indexes": args.target_indexes,
            "encoder": args.encoder,
            "segformer_model_path": args.model_path,
        }

        torch.save(ckpt, output_dir / "adapter_last.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ckpt, output_dir / "adapter_best.pt")
            print(f"Saved new best adapter: F1={best_f1:.4f}")
            # Save a few adapter outputs for visual QA.
            qa_dir = output_dir / "adapter_outputs_epoch_best"
            qa_dir.mkdir(parents=True, exist_ok=True)

            example_paths = sorted(Path(args.val_gray_dir).glob("*.tif"))[:5]

            for gray_path in example_paths:
                out_path = qa_dir / gray_path.name
                save_adapter_output_geotiff(
                    adapter=adapter,
                    gray_path=gray_path,
                    out_path=out_path,
                    device=device,
                )

if __name__ == "__main__":
    main()
"""
USAGE
python train_gray_adapter_frozen_segformer.py \
  --train-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/gray \
  --train-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/masks \
  --train-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/train/original \
  --val-gray-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/gray \
  --val-mask-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/masks \
  --val-ref-dir /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/1024_split/val/original \
  --encoder mit-b3 \
  --model-path /home1/09208/asperera/PDG_shared2/TCN_mosaic_pipline/git_ver/inference/segf_mit_b3_tcn_finetuned.pth \
  --output-dir /home1/09208/asperera/PDG_shared2/TCN_Training/gray_adapter \
  --target-indexes 8 6 1 \
  --chip-size 1024 \
  --batch-size 1 \
  --epochs 20 \
  --samples-per-epoch 2000 \
  --lr 1e-4 \
  --threshold 0.05 \
  --lambda-recon 0.0

"""
