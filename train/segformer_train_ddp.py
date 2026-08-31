import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


NUM_CLASSES = 2
SUPPORTED_EXTENSIONS = (".png", ".tiff", ".jpg", ".jpeg", ".tif")
MODEL_NAMES = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
    "b3": "nvidia/mit-b3",
    "b4": "nvidia/mit-b4",
    "b5": "nvidia/mit-b5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SegFormer model for binary trough segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required directories: no defaults are provided.
    parser.add_argument(
        "--train_img_dir",
        required=True,
        type=Path,
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--train_mask_dir",
        required=True,
        type=Path,
        help="Directory containing training masks.",
    )
    parser.add_argument(
        "--val_img_dir",
        required=True,
        type=Path,
        help="Directory containing validation images.",
    )
    parser.add_argument(
        "--val_mask_dir",
        required=True,
        type=Path,
        help="Directory containing validation masks.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Root directory in which the timestamped run folder will be created.",
    )

    # Model and chip settings.
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_NAMES),
        default="b0",
        help="SegFormer MiT backbone. Type only b0, b1, b2, b3, b4, or b5.",
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=128,
        help="Square input-chip size in pixels. Images and masks are resized to chip_size x chip_size.",
    )

    # Training settings.
    parser.add_argument("--batch_size", type=int, default=1, help="Training and validation batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes per GPU.")

    # Focal-loss settings.
    parser.add_argument("--focal_alpha", type=float, default=0.9, help="Focal-loss alpha.")
    parser.add_argument("--focal_gamma", type=float, default=3.0, help="Focal-loss gamma.")

    # Scheduler and optimization settings.
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=10,
        help="Epoch interval between StepLR learning-rate reductions.",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.1,
        help="StepLR learning-rate multiplier.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm used for clipping.",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    required_input_directories = {
        "--train_img_dir": args.train_img_dir,
        "--train_mask_dir": args.train_mask_dir,
        "--val_img_dir": args.val_img_dir,
        "--val_mask_dir": args.val_mask_dir,
    }

    for argument_name, directory in required_input_directories.items():
        if not directory.is_dir():
            raise NotADirectoryError(f"{argument_name} does not exist or is not a directory: {directory}")

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(f"--output_dir exists but is not a directory: {args.output_dir}")

    if args.chip_size <= 0:
        raise ValueError("--chip_size must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be greater than 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay cannot be negative.")
    if args.patience <= 0:
        raise ValueError("--patience must be greater than 0.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.focal_alpha < 0:
        raise ValueError("--focal_alpha cannot be negative.")
    if args.focal_gamma < 0:
        raise ValueError("--focal_gamma cannot be negative.")
    if args.scheduler_step_size <= 0:
        raise ValueError("--scheduler_step_size must be greater than 0.")
    if args.scheduler_gamma <= 0:
        raise ValueError("--scheduler_gamma must be greater than 0.")
    if args.max_grad_norm <= 0:
        raise ValueError("--max_grad_norm must be greater than 0.")


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    """
    Initialize Distributed Data Parallel when launched with torchrun.

    Returns:
        distributed: Whether DDP is active.
        rank: Global process rank.
        local_rank: GPU index on the current node.
        world_size: Total number of DDP processes.
        device: Device assigned to the current process.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA GPUs.")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


class DistributedEvaluationSampler(Sampler[int]):
    """
    Split validation samples across DDP ranks without padding or duplication.

    PyTorch's standard DistributedSampler pads datasets that are not evenly
    divisible by the number of ranks. Padding is desirable for synchronized
    training batches but would duplicate validation samples and slightly bias
    validation metrics.
    """

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.world_size) + 1


class ImageSegmentationDataset(Dataset):
    """
    Dataset for paired image and mask files.

    Images are normalized to [0, 1], converted to three channels when needed,
    and resized to chip_size x chip_size. Masks are resized with nearest-neighbor
    interpolation and converted to binary labels: 0 for background and 1 for troughs.
    """

    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path,
        image_processor: SegformerImageProcessor,
        chip_size: int,
    ) -> None:
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_processor = image_processor
        self.chip_size = chip_size

        image_files = self._index_files(self.img_dir)
        mask_files = self._index_files(self.mask_dir)

        matching_bases = sorted(set(image_files).intersection(mask_files))
        if not matching_bases:
            raise ValueError(
                f"No matching image-mask pairs found between {self.img_dir} and {self.mask_dir}."
            )

        self.pairs = [(image_files[base], mask_files[base]) for base in matching_bases]
        logging.info(
            "Found %d paired files in images=%s and masks=%s.",
            len(self.pairs),
            self.img_dir,
            self.mask_dir,
        )

    @staticmethod
    def _index_files(directory: Path) -> dict[str, Path]:
        indexed_files: dict[str, Path] = {}

        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            base = path.stem
            if base in indexed_files:
                raise ValueError(
                    f"Multiple supported files share the same base name '{base}' in {directory}."
                )
            indexed_files[base] = path

        return indexed_files

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        image = self._prepare_image(image, image_path)
        mask = self._prepare_mask(mask)

        encoding = self.image_processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        return {key: value.squeeze(0) for key, value in encoding.items()}

    def _prepare_image(self, image: np.ndarray, image_path: Path) -> np.ndarray:
        if image.shape[:2] != (self.chip_size, self.chip_size):
            interpolation = cv2.INTER_AREA if max(image.shape[:2]) > self.chip_size else cv2.INTER_LINEAR
            image = cv2.resize(
                image,
                (self.chip_size, self.chip_size),
                interpolation=interpolation,
            )

        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            min_value = float(image.min())
            max_value = float(image.max())
            if min_value < 0.0 or max_value > 1.0:
                raise ValueError(
                    f"Unsupported floating-point image range for {image_path}: "
                    f"[{min_value}, {max_value}]. Expected [0, 1]."
                )

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected a one-channel or three-channel image, but received shape "
                f"{image.shape} for {image_path}."
            )

        return image

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] != (self.chip_size, self.chip_size):
            mask = cv2.resize(
                mask,
                (self.chip_size, self.chip_size),
                interpolation=cv2.INTER_NEAREST,
            )

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return np.where(mask > 127, 1, 0).astype(np.int64)


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = self.cross_entropy(logits, targets)
        probability_of_true_class = torch.exp(-cross_entropy)
        focal_loss = self.alpha * (1.0 - probability_of_true_class) ** self.gamma * cross_entropy
        return focal_loss.mean()


class SegmentationMetrics:
    """Accumulate a confusion matrix and compute classwise accuracy and IoU."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        true_labels = true_labels.reshape(-1)
        predicted_labels = predicted_labels.reshape(-1)

        valid = (
            (true_labels >= 0)
            & (true_labels < self.num_classes)
            & (predicted_labels >= 0)
            & (predicted_labels < self.num_classes)
        )

        encoded = self.num_classes * true_labels[valid] + predicted_labels[valid]
        self.confusion_matrix += np.bincount(
            encoded,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def synchronize(self, device: torch.device, distributed: bool) -> None:
        if not distributed:
            return

        confusion_matrix = torch.as_tensor(
            self.confusion_matrix,
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        self.confusion_matrix = confusion_matrix.cpu().numpy()

    def class_accuracy(self, class_id: int) -> float:
        denominator = self.confusion_matrix[class_id, :].sum()
        if denominator == 0:
            return float("nan")
        return float(self.confusion_matrix[class_id, class_id] / denominator)

    def class_iou(self, class_id: int) -> float:
        true_positive = self.confusion_matrix[class_id, class_id]
        false_negative = self.confusion_matrix[class_id, :].sum() - true_positive
        false_positive = self.confusion_matrix[:, class_id].sum() - true_positive
        denominator = true_positive + false_negative + false_positive

        if denominator == 0:
            return float("nan")
        return float(true_positive / denominator)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
    description: str,
    distributed: bool,
    rank: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    metrics = SegmentationMetrics(NUM_CLASSES)

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
        disable=not is_main_process(rank),
    )

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(pixel_values=pixel_values)

            resized_labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=outputs.logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()

            loss = loss_function(outputs.logits, resized_labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        predictions = outputs.logits.argmax(dim=1).detach().cpu().numpy()
        true_labels = resized_labels.detach().cpu().numpy()
        metrics.update(true_labels=true_labels, predicted_labels=predictions)

        if is_main_process(rank):
            progress_bar.set_postfix(loss=f"{total_loss / total_batches:.4f}")

    loss_statistics = torch.tensor(
        [total_loss, float(total_batches)],
        dtype=torch.float64,
        device=device,
    )

    if distributed:
        dist.all_reduce(loss_statistics, op=dist.ReduceOp.SUM)

    global_total_loss = float(loss_statistics[0].item())
    global_total_batches = int(loss_statistics[1].item())

    if global_total_batches == 0:
        raise ValueError(f"{description} DataLoader is empty.")

    metrics.synchronize(device=device, distributed=distributed)

    return {
        "loss": global_total_loss / global_total_batches,
        "background_accuracy": metrics.class_accuracy(0),
        "trough_accuracy": metrics.class_accuracy(1),
        "background_iou": metrics.class_iou(0),
        "trough_iou": metrics.class_iou(1),
    }


def create_metrics_dictionary() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_background_accuracy": [],
        "train_trough_accuracy": [],
        "train_background_iou": [],
        "train_trough_iou": [],
        "val_background_accuracy": [],
        "val_trough_accuracy": [],
        "val_background_iou": [],
        "val_trough_iou": [],
    }


def append_epoch_metrics(
    metrics: dict[str, list[float]],
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    metrics["epoch"].append(epoch)

    for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
        for metric_name, metric_value in split_metrics.items():
            metrics[f"{split_name}_{metric_name}"].append(metric_value)


def save_metrics_plot(
    metrics: dict[str, list[float]],
    save_directory: Path,
    filename: str,
) -> None:
    figure = plt.figure(figsize=(20, 10))
    metric_names = [
        ("background_accuracy", "Background Accuracy"),
        ("trough_accuracy", "Trough Accuracy"),
        ("background_iou", "Background IoU"),
        ("trough_iou", "Trough IoU"),
    ]

    for index, (metric_key, label) in enumerate(metric_names, start=1):
        axis = figure.add_subplot(2, 2, index)
        axis.plot(metrics["epoch"], metrics[f"train_{metric_key}"], label=f"Train {label}")
        axis.plot(metrics["epoch"], metrics[f"val_{metric_key}"], label=f"Validation {label}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_title(f"{label} over Epochs")
        axis.legend()
        axis.grid(True)

    figure.tight_layout()
    plot_path = save_directory / filename
    figure.savefig(plot_path)
    plt.close(figure)
    logging.info("Saved metrics plot: %s", plot_path)


def save_metrics_table(metrics: dict[str, list[float]], save_directory: Path) -> None:
    metrics_path = save_directory / "training_metrics.xlsx"
    pd.DataFrame(metrics).to_excel(metrics_path, index=False)
    logging.info("Saved metrics table: %s", metrics_path)


def configure_logging(save_directory: Path, rank: int) -> None:
    if is_main_process(rank):
        log_path = save_directory / "training.log"
        handlers: list[logging.Handler] = [
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
        level = logging.INFO
    else:
        handlers = [logging.NullHandler()]
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def create_save_directory(
    output_dir: Path,
    model: str,
    chip_size: int,
    distributed: bool,
    rank: int,
) -> Path:
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = output_dir / f"segformer_{model}_{chip_size}px_{timestamp}"
        save_directory.mkdir(parents=True, exist_ok=False)
        save_directory_string: list[str | None] = [str(save_directory)]
    else:
        save_directory_string = [None]

    if distributed:
        dist.broadcast_object_list(save_directory_string, src=0)

    if save_directory_string[0] is None:
        raise RuntimeError("Failed to obtain the shared output directory.")

    return Path(save_directory_string[0])


def save_run_configuration(
    args: argparse.Namespace,
    save_directory: Path,
    distributed: bool,
    world_size: int,
) -> None:
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["ddp_enabled"] = distributed
    configuration["world_size"] = world_size

    configuration_path = save_directory / "run_configuration.json"
    with configuration_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    logging.info("Saved run configuration: %s", configuration_path)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed()

    try:
        save_directory = create_save_directory(
            output_dir=args.output_dir,
            model=args.model,
            chip_size=args.chip_size,
            distributed=distributed,
            rank=rank,
        )
        configure_logging(save_directory=save_directory, rank=rank)

        if is_main_process(rank):
            logging.info("Outputs will be saved to: %s", save_directory)
            logging.info("Selected SegFormer backbone: %s -> %s", args.model, MODEL_NAMES[args.model])
            logging.info("Square chip size: %d x %d pixels", args.chip_size, args.chip_size)
            logging.info("DDP enabled: %s | World size: %d", distributed, world_size)
            save_run_configuration(
                args=args,
                save_directory=save_directory,
                distributed=distributed,
                world_size=world_size,
            )

        image_processor = SegformerImageProcessor(
            do_rescale=False,
            do_resize=False,
        )

        train_dataset = ImageSegmentationDataset(
            img_dir=args.train_img_dir,
            mask_dir=args.train_mask_dir,
            image_processor=image_processor,
            chip_size=args.chip_size,
        )
        val_dataset = ImageSegmentationDataset(
            img_dir=args.val_img_dir,
            mask_dir=args.val_mask_dir,
            image_processor=image_processor,
            chip_size=args.chip_size,
        )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            if distributed
            else None
        )
        val_sampler = (
            DistributedEvaluationSampler(
                val_dataset,
                rank=rank,
                world_size=world_size,
            )
            if distributed
            else None
        )

        pin_memory = device.type == "cuda"

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        id2label = {0: "background", 1: "troughs"}
        label2id = {"background": 0, "troughs": 1}

        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAMES[args.model],
            ignore_mismatched_sizes=True,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            reshape_last_stage=True,
        )
        model.to(device)

        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        if is_main_process(rank):
            logging.info("Using device: %s", device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        loss_function = FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )

        metrics = create_metrics_dictionary()
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main_process(rank):
                logging.info("Epoch %d/%d", epoch, args.epochs)

            train_metrics = run_epoch(
                model=model,
                dataloader=train_dataloader,
                loss_function=loss_function,
                device=device,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
                description="Training",
                distributed=distributed,
                rank=rank,
            )
            val_metrics = run_epoch(
                model=model,
                dataloader=val_dataloader,
                loss_function=loss_function,
                device=device,
                optimizer=None,
                max_grad_norm=args.max_grad_norm,
                description="Validation",
                distributed=distributed,
                rank=rank,
            )

            append_epoch_metrics(
                metrics=metrics,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            if is_main_process(rank):
                logging.info("Train loss: %.4f | Validation loss: %.4f", train_metrics["loss"], val_metrics["loss"])
                logging.info(
                    "Train background accuracy: %.4f | Train trough accuracy: %.4f",
                    train_metrics["background_accuracy"],
                    train_metrics["trough_accuracy"],
                )
                logging.info(
                    "Train background IoU: %.4f | Train trough IoU: %.4f",
                    train_metrics["background_iou"],
                    train_metrics["trough_iou"],
                )
                logging.info(
                    "Validation background accuracy: %.4f | Validation trough accuracy: %.4f",
                    val_metrics["background_accuracy"],
                    val_metrics["trough_accuracy"],
                )
                logging.info(
                    "Validation background IoU: %.4f | Validation trough IoU: %.4f",
                    val_metrics["background_iou"],
                    val_metrics["trough_iou"],
                )

                save_metrics_plot(
                    metrics=metrics,
                    save_directory=save_directory,
                    filename=f"training_metrics_epoch_{epoch}.png",
                )

            scheduler.step()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0

                if is_main_process(rank):
                    model_path = save_directory / f"segformer_{args.model}_best_epoch_{epoch}.pth"
                    torch.save(unwrap_model(model).state_dict(), model_path)
                    logging.info("Saved new best model: %s", model_path)
            else:
                epochs_without_improvement += 1

                if is_main_process(rank):
                    logging.info(
                        "No validation-loss improvement for %d epoch(s).",
                        epochs_without_improvement,
                    )

            should_stop = epochs_without_improvement >= args.patience

            if distributed:
                dist.barrier()

            if should_stop:
                if is_main_process(rank):
                    logging.info("Early stopping triggered.")
                break

        if is_main_process(rank):
            save_metrics_plot(
                metrics=metrics,
                save_directory=save_directory,
                filename="training_metrics_final.png",
            )
            save_metrics_table(metrics=metrics, save_directory=save_directory)

        if distributed:
            dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Training failed.")
        raise
