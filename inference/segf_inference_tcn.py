#!/usr/bin/env python3
"""
segf_inference_tcn.py

Is the gpu pipeline to detect the troughs from the VHRS images.
Will produce the model detected TCN masks that can be fed to the Graph Theoretic pipeline
Model used is Segformer model finetuned for detecting TCNs with a corresponding weight file

It is executed at tile level to recreate a tree of tile/subtiles of masks

Helper script is avaialble to submit multiple SLURM jobs on HPC
create_multi_slurm_jobs.py

Code keeps tmp files in the node(local) to optimized on I/O

USSAGE
 python ./inference/segf_inference_tcn.py --input_dir ../data/ --output_dir ./test

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera
"""

import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, Resampling
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
from skimage import exposure
from torch.utils.data import Dataset, DataLoader
import argparse


# -----------------------
# Configuration
# -----------------------

ENCODER = "mit-b3"
CHIP_SIZE = 1024
OVERLAP = 128
THRESHOLD = 0.01
TARGET_RES = 0.5   # 0.5-meter resolution
NULL_VALUE = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use a small batch size to reduce peak memory usage based on architecture avaialble.
BATCH_SIZE = 1

INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"

#--------------------Helpers to get model---------------
import os
import time
import hashlib
import logging
import tempfile
import urllib.request
from pathlib import Path
from contextlib import contextmanager

# --- some hard-coded defaults live in inference only ---
MODEL_PATH = Path(__file__).resolve().parent / "segf-mit-b3_tcn_finetuned.pth"

# Release URL from PGC google drive
MODEL_URL = "https://drive.google.com/file/d/1pI8n9PZYPzWXm5h1kpnn2XPKdv_DYoeb/view?usp=sharing"

# Model integrity check
MODEL_SHA256 = "bb92f7471d2a3a145f45509bffe177e5c016a22154cebd11e0eba07602d5510b"  # SHA-256 (hex) for "segf-mit-b3_tcn_finetuned.pth"

@contextmanager
def _file_lock(lock_path: Path, poll_s: float = 0.25):
    """
    Prevents multiple processes (or SLURM tasks) downloading the same file at once.
    Linux-only (HPC-friendly). If fcntl missing, lock becomes a no-op.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        try:
            import fcntl
        except Exception:
            yield
            return

        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(poll_s)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()

def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_model_weights(
    *,
    dst_path: Path = MODEL_PATH,
    url: str = MODEL_URL,
    sha256: str | None = MODEL_SHA256,
    force: bool = False,
    timeout_s: int = 60,
) -> Path:
    """
    If dst_path doesn't exist, download it. Safe for multi-worker / SLURM.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = dst_path.with_suffix(dst_path.suffix + ".lock")
    with _file_lock(lock_path):
        if dst_path.exists() and not force:
            if sha256:
                got = _sha256_file(dst_path)
                if got.lower() != sha256.lower():
                    raise RuntimeError(f"SHA256 mismatch for {dst_path} (got {got}, expected {sha256})")
            return dst_path

        logging.info("Weights not found. Downloading -> %s", dst_path)

        # download to temp then atomic rename
        with tempfile.NamedTemporaryFile(dir=str(dst_path.parent), delete=False) as tf:
            tmp = Path(tf.name)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tcn-inference/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r, open(tmp, "wb") as out:
                out.write(r.read())

            if sha256:
                got = _sha256_file(tmp)
                if got.lower() != sha256.lower():
                    raise RuntimeError(f"SHA256 mismatch after download (got {got}, expected {sha256})")

            os.replace(tmp, dst_path)  # atomic on POSIX
            logging.info("Downloaded weights: %.1f MB", dst_path.stat().st_size / (1024**2))
            return dst_path
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

#--------------------</Helpers to get model>---------------
# -----------------------
# Helper Functions
# -----------------------

def resample_raster(src_raster, target_res):
    transform, width, height = calculate_default_transform(
        src_raster.crs, src_raster.crs, src_raster.width, src_raster.height,
        *src_raster.bounds, resolution=target_res
    )
    kwargs = src_raster.meta.copy()
    kwargs.update({'transform': transform, 'width': width, 'height': height})
    data = src_raster.read(out_shape=(src_raster.count, height, width), resampling=Resampling.bilinear)
    return data, kwargs

def reflect_pad(image, pad_size):
    return np.pad(image, ((0,0),(pad_size, pad_size),(pad_size, pad_size)), mode='reflect')

def generate_chips(w, h, chip_size, overlap):
    step = chip_size - overlap
    return [(x, y) for y in range(0, h, step) for x in range(0, w, step)]

def apply_ahe_chip(chip, grid_size=(8,8)):
    """
    Apply AHE per band on a 1024x1024 chip.
    chip shape: (bands, CHIP_SIZE, CHIP_SIZE)
    """
    chip_out = np.zeros_like(chip, dtype=np.float32)
    for i in range(chip.shape[0]):
        band = chip[i]
        max_val = band.max() if band.max() > 0 else 1e-8
        band_norm = band / max_val
        eq = exposure.equalize_adapthist(band_norm, kernel_size=grid_size)
        chip_out[i] = eq * max_val
    return chip_out

# Custom collate to keep positions as tuples.
def custom_collate(batch):
    chips, positions = zip(*batch)
    chips = torch.stack([torch.as_tensor(chip).float() for chip in chips])
    return chips, positions

# -----------------------
# Inference Function
# -----------------------

def infer_tif(input_path, output_path, model, batch_size=BATCH_SIZE):
    with rasterio.open(input_path) as src:
        # Resample to target resolution and select only bands 8, 6, and 1.
        img_resampled, profile = resample_raster(src, TARGET_RES)
        img_selected = img_resampled[[7, 5, 0], :, :]
        # Create a null mask: pixels where all selected bands equal NULL_VALUE.
        null_mask = (img_selected == NULL_VALUE).all(axis=0)
        # Get original dimensions.
        orig_h, orig_w = img_selected.shape[1:]
    
    # Pad the full image.
    pad = CHIP_SIZE // 2
    padded_img = reflect_pad(img_selected, pad)
    # Generate chips based on original dimensions.
    chips = generate_chips(orig_w, orig_h, CHIP_SIZE, OVERLAP)

    # Instead of large in‑memory arrays, create memmaps for predictions.
    #sum_preds = np.memmap("temp_sum_preds.dat", dtype=np.float32, mode="w+", shape=(orig_h, orig_w))
    #counts = np.memmap("temp_counts.dat", dtype=np.uint16, mode="w+", shape=(orig_h, orig_w))

    # Instead of large in‑memory arrays, create memmaps for predictions on the local storage
    temp_dir = os.getenv("SLURM_TMPDIR", "/tmp")  # Use node-local storage if available
    sum_preds_path = os.path.join(temp_dir, "temp_sum_preds.dat")
    counts_path = os.path.join(temp_dir, "temp_counts.dat")

    sum_preds = np.memmap(sum_preds_path, dtype=np.float32, mode="w+", shape=(orig_h, orig_w))
    counts = np.memmap(counts_path, dtype=np.uint16, mode="w+", shape=(orig_h, orig_w))

    # Initialize memmaps to zero.
    sum_preds[:] = 0
    counts[:] = 0

    # Dataset: extract chips from padded image using the pad offset.
    class ChipDataset(Dataset):
        def __init__(self, image, chips, chip_size, pad):
            self.image = image
            self.chips = chips
            self.chip_size = chip_size
            self.pad = pad  # Offset due to padding.
    
        def __len__(self):
            return len(self.chips)
    
        def __getitem__(self, idx):
            x, y = self.chips[idx]
            # Extract from padded image: add pad offset.
            chip = self.image[:, y + self.pad:y + self.pad + self.chip_size,
                              x + self.pad:x + self.pad + self.chip_size]
            # Apply AHE on this chip.
            chip = apply_ahe_chip(chip, grid_size=(8,8))
            return chip, (x, y)
    
    dataset = ChipDataset(padded_img, chips, CHIP_SIZE, pad)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    
    model.eval()
    with torch.no_grad():
        for batch, positions in tqdm(loader, desc="Inferencing", total=len(loader)):
            batch = batch.to(DEVICE).float()
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(pixel_values=batch).logits
                preds = torch.sigmoid(F.interpolate(logits, size=(CHIP_SIZE, CHIP_SIZE), mode='bilinear'))
            preds = preds.cpu().numpy()
            for pred, pos in zip(preds, positions):
                if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                    raise ValueError("Expected a tuple of (x, y), got: {}".format(pos))
                x, y = pos
                # Use the original image coordinates.
                x_end, y_end = min(x + CHIP_SIZE, orig_w), min(y + CHIP_SIZE, orig_h)
                chip_h, chip_w = y_end - y, x_end - x
                sum_preds[y:y_end, x:x_end] += pred[0, :chip_h, :chip_w]
                counts[y:y_end, x:x_end] += 1
            torch.cuda.empty_cache()
    
    # Write final output window-by-window to avoid loading full array into RAM.
    profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw', 'BIGTIFF': 'YES'})
    # Remove nodata if exists.
    profile.pop('nodata', None)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Use block windows if available, else define a tile size (e.g., 1024x1024)
        tile_size = 1024
        for i in range(0, orig_h, tile_size):
            for j in range(0, orig_w, tile_size):
                h_tile = min(tile_size, orig_h - i)
                w_tile = min(tile_size, orig_w - j)
                window = Window(j, i, w_tile, h_tile)
                window_sum = np.array(sum_preds[i:i+h_tile, j:j+w_tile])
                window_count = np.array(counts[i:i+h_tile, j:j+w_tile])
                window_avg = window_sum / np.maximum(window_count, 1)
                window_mask = (window_avg > THRESHOLD).astype(np.uint8)
                # Apply null mask.
                window_null = null_mask[i:i+h_tile, j:j+w_tile]
                window_mask[window_null] = 0
                dst.write(window_mask, 1, window=window)
    
    # Clean up temporary memmap files.
    del sum_preds
    del counts
    if os.path.exists(sum_preds_path):
        os.remove(sum_preds_path)
    if os.path.exists(counts_path):
        os.remove(counts_path)
    """
    if os.path.exists("temp_sum_preds.dat"):
        os.remove("temp_sum_preds.dat")
    if os.path.exists("temp_counts.dat"):
        os.remove("temp_counts.dat")
    """
# -----------------------
# Model Loading
# -----------------------

def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(f'nvidia/{ENCODER}', num_labels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model

# -----------------------
# Main Execution
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Execute inference on input dir and write out dir")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to input directory containing image files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory where results should be stored.")
    args = parser.parse_args()
    INPUT_FOLDER = args.input_dir
    OUTPUT_FOLDER = args.output_dir
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = load_model()
    tif_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")]
    for tif_file in tif_files:
        input_path = os.path.join(INPUT_FOLDER, tif_file)
        output_file = os.path.splitext(tif_file)[0] + "_mask.tif"
        output_path = os.path.join(OUTPUT_FOLDER, output_file)
        print(f"Processing: {tif_file}")
        infer_tif(input_path, output_path, model, batch_size=BATCH_SIZE)
        print(f"Finished: {output_file}")

if __name__ == "__main__":
    main()
