import os
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.distributed as dist
import torch.nn.functional as F
from albumentations import Compose, Resize
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

# Headless-safe settings for HPC/SLURM compute nodes.
# Prevent matplotlib/OpenCV/Qt from trying to open a GUI display.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        image_paths = sorted(
            [
                os.path.join(image_dir, filename)
                for filename in os.listdir(image_dir)
                if filename.lower().endswith(".tif")
            ]
        )

        mask_paths = sorted(
            [
                os.path.join(mask_dir, filename)
                for filename in os.listdir(mask_dir)
                if filename.lower().endswith(".tif")
            ]
        )

        if not image_paths:
            raise ValueError(f"No .tif images were found in: {image_dir}")

        if not mask_paths:
            raise ValueError(f"No .tif masks were found in: {mask_dir}")

        image_paths_by_stem = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in image_paths
        }

        mask_paths_by_stem = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in mask_paths
        }

        image_stems = set(image_paths_by_stem.keys())
        mask_stems = set(mask_paths_by_stem.keys())

        if image_stems != mask_stems:
            missing_masks = sorted(image_stems - mask_stems)
            missing_images = sorted(mask_stems - image_stems)

            raise ValueError(
                "Image and mask filenames do not match by filename stem.\n"
                f"Images without masks: {missing_masks[:10]}\n"
                f"Masks without images: {missing_images[:10]}"
            )

        self.samples = [
            (image_paths_by_stem[stem], mask_paths_by_stem[stem])
            for stem in sorted(image_stems)
        ]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__OLD(self, idx):
        image_path, mask_path = self.samples[idx]

        with rasterio.open(image_path) as image_src:
            image = image_src.read(1).astype(np.float32)

        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.int64)

        invalid_mask_values = np.logical_and(mask != 0, mask != 1)

        if invalid_mask_values.any():
            invalid_values = np.unique(mask[invalid_mask_values])

            raise ValueError(
                f"Mask contains values other than 0 and 1: {mask_path}. "
                f"Invalid values: {invalid_values[:10]}"
            )

        # The pretrained MiT backbone expects three channels.
        # Preserve the original single-band input values and repeat the band.
        image = np.repeat(image[..., None], repeats=3, axis=2)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        with rasterio.open(image_path) as image_src:
            if image_src.count >= 3:
                # Read first three bands as RGB/pseudo-RGB.
                # rasterio gives shape: (bands, H, W)
                image = image_src.read([1, 2, 3]).astype(np.float32)

                # Convert to shape expected by albumentations:
                # (bands, H, W) -> (H, W, bands)
                image = np.transpose(image, (1, 2, 0))

            else:
                # Fall back to single-band input and repeat to 3 channels,
                # because the pretrained MiT backbone expects 3-channel input.
                image = image_src.read(1).astype(np.float32)
                image = np.repeat(image[..., None], repeats=3, axis=2)


        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.int64)

        # Allow common binary mask encodings:
        #   0/1   -> already class labels
        #   0/255 -> convert foreground to class label 1
        valid_mask_values = np.isin(mask, [0, 1, 255])

        if not valid_mask_values.all():
            invalid_values = np.unique(mask[~valid_mask_values])

            raise ValueError(
                f"Mask contains values other than 0, 1, and 255: {mask_path}. "
                f"Invalid values: {invalid_values[:10]}"
            )

        # Convert binary mask to class labels expected by training:
        #   background = 0
        #   TCN        = 1
        mask = (mask > 0).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

def calculate_confusion_counts(predictions, targets):
    preds = torch.argmax(predictions, dim=1)

    confusion_counts = torch.zeros(
        (2, 4),
        dtype=torch.float64,
        device=predictions.device,
    )

    for label in (0, 1):
        true_positive = ((preds == label) & (targets == label)).sum()
        false_positive = ((preds == label) & (targets != label)).sum()
        false_negative = ((preds != label) & (targets == label)).sum()
        true_negative = ((preds != label) & (targets != label)).sum()

        confusion_counts[label, 0] = true_positive
        confusion_counts[label, 1] = false_positive
        confusion_counts[label, 2] = false_negative
        confusion_counts[label, 3] = true_negative

    return confusion_counts


def calculate_metrics_from_counts(confusion_counts, epsilon=1e-7):
    metrics = {}

    for label in (0, 1):
        true_positive = confusion_counts[label, 0]
        false_positive = confusion_counts[label, 1]
        false_negative = confusion_counts[label, 2]
        true_negative = confusion_counts[label, 3]

        accuracy = (
            (true_positive + true_negative)
            / (
                true_positive
                + false_positive
                + false_negative
                + true_negative
                + epsilon
            )
        )

        precision = true_positive / (
            true_positive + false_positive + epsilon
        )

        recall = true_positive / (
            true_positive + false_negative + epsilon
        )

        f1 = (
            2 * precision * recall
            / (precision + recall + epsilon)
        )

        iou = true_positive / (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        )

        metrics[f"{label}_accuracy"] = accuracy.item()
        metrics[f"{label}_precision"] = precision.item()
        metrics[f"{label}_recall"] = recall.item()
        metrics[f"{label}_f1"] = f1.item()
        metrics[f"{label}_iou"] = iou.item()

    metrics["mean_iou"] = (
        metrics["0_iou"] + metrics["1_iou"]
    ) / 2

    return metrics


def aggregate_epoch_results(
    total_loss,
    number_of_batches,
    confusion_counts,
    device,
):
    loss_counts = torch.tensor(
        [total_loss, number_of_batches],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(confusion_counts, op=dist.ReduceOp.SUM)

    if loss_counts[1].item() == 0:
        raise ValueError("The DataLoader produced zero batches.")

    average_loss = (
        loss_counts[0] / loss_counts[1]
    ).item()

    metrics = calculate_metrics_from_counts(confusion_counts)

    return average_loss, metrics


def save_metrics(metrics, output_dir, epoch):
    csv_path = os.path.join(output_dir, "metrics.csv")
    dataframe = pd.DataFrame([metrics])

    if epoch == 1:
        dataframe.to_csv(csv_path, index=False)
    else:
        dataframe.to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False,
        )


def plot_metrics(metrics_history, output_dir):
    for key, values in metrics_history.items():
        plt.figure()
        plt.plot(values, label=key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}.png"))
        plt.close()


def save_checkpoint(
    model,
    optimizer,
    scaler,
    args,
    epoch,
    checkpoint_output_dir,
):
    checkpoint_path = os.path.join(
        checkpoint_output_dir,
        f"checkpoint_epoch_{epoch}.pt",
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scaler,
    loss_fn,
    rank,
):
    model.train()

    total_loss = 0.0

    confusion_counts = torch.zeros(
        (2, 4),
        dtype=torch.float64,
        device=device,
    )

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        disable=rank != 0,
    )

    for images, masks in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(pixel_values=images).logits

            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = loss_fn(outputs, masks.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        confusion_counts += calculate_confusion_counts(
            outputs.detach(),
            masks,
        )

        if rank == 0:
            progress_bar.set_postfix(
                loss=total_loss / (progress_bar.n + 1)
            )

    return aggregate_epoch_results(
        total_loss=total_loss,
        number_of_batches=len(dataloader),
        confusion_counts=confusion_counts,
        device=device,
    )


def validate_one_epoch(
    model,
    dataloader,
    device,
    loss_fn,
    rank,
):
    model.eval()

    total_loss = 0.0

    confusion_counts = torch.zeros(
        (2, 4),
        dtype=torch.float64,
        device=device,
    )

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        disable=rank != 0,
    )

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(pixel_values=images).logits

                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = loss_fn(outputs, masks.long())

            total_loss += loss.item()

            confusion_counts += calculate_confusion_counts(
                outputs,
                masks,
            )

            if rank == 0:
                progress_bar.set_postfix(
                    loss=total_loss / (progress_bar.n + 1)
                )

    return aggregate_epoch_results(
        total_loss=total_loss,
        number_of_batches=len(dataloader),
        confusion_counts=confusion_counts,
        device=device,
    )


def main_worker(rank, args):
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank,
    )

    id2label = {
        0: "Background",
        1: "Trough",
    }

    label2id = {
        "Background": 0,
        "Trough": 1,
    }

    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit-{args.model_size}",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    scaler = torch.amp.GradScaler("cuda")

    train_dataset = SegmentationDataset(
        args.train_path,
        args.train_mask_path,
        transform=Compose(
            [
                Resize(args.chip_size, args.chip_size),
                ToTensorV2(),
            ]
        ),
    )

    val_dataset = SegmentationDataset(
        args.val_path,
        args.val_mask_path,
        transform=Compose(
            [
                Resize(args.chip_size, args.chip_size),
                ToTensorV2(),
            ]
        ),
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    metrics_output_dir = os.path.join(
        args.output_dir,
        "metrics",
    )

    checkpoint_output_dir = os.path.join(
        args.output_dir,
        "checkpoints",
    )

    if rank == 0:
        os.makedirs(metrics_output_dir, exist_ok=True)
        os.makedirs(checkpoint_output_dir, exist_ok=True)

    dist.barrier()

    metrics_history = {}

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            loss_fn=loss_fn,
            rank=rank,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            rank=rank,
        )

        if rank == 0:
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            epoch_metrics.update(
                {
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                }
            )

            epoch_metrics.update(
                {
                    f"val_{key}": value
                    for key, value in val_metrics.items()
                }
            )

            save_metrics(
                metrics=epoch_metrics,
                output_dir=metrics_output_dir,
                epoch=epoch,
            )

            for key, value in epoch_metrics.items():
                if key == "epoch":
                    continue

                if key not in metrics_history:
                    metrics_history[key] = []

                metrics_history[key].append(value)

            plot_metrics(
                metrics_history=metrics_history,
                output_dir=metrics_output_dir,
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                checkpoint_output_dir=checkpoint_output_dir,
            )

            print(
                f"[Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Train Mean IoU: {train_metrics['mean_iou']:.4f}, "
                f"Val Mean IoU: {val_metrics['mean_iou']:.4f}"
            )

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", required=True)
    parser.add_argument("--chip_size", type=int, required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--train_mask_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--val_mask_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dist_url", required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--base_output_dir", required=True)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args.output_dir = os.path.join(
        args.base_output_dir,
        f"segformer_{args.model_size}_chip{args.chip_size}_{timestamp}",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Results will be saved to: {args.output_dir}")

    torch.multiprocessing.spawn(
        main_worker,
        args=(args,),
        nprocs=args.world_size,
    )


"""
USAGE
python segformer_train.py \
  --model_size mit-b3 \
  --chip_size 1024 \
  --train_path /scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/train_1024/images \
  --train_mask_path /scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/train_1024/masks \
  --val_path /scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/val_1024/images \
  --val_mask_path /scratch2/projects/PDG_shared/TCN_Training/tcn_mxr/val_1024/masks \
  --batch_size 1 \
  --epochs 20 \
  --lr 1e-4 \
  --dist_url tcp://127.0.0.1:29500 \
  --world_size 4 \
  --base_output_dir ./outputs/segformer_tcn_mitb3_1024
"""