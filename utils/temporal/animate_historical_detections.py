from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio

# -----------------------------
# User paths
# -----------------------------
input_dir = Path("/work/09208/asperera/ls6/DATA/historical/tcn_selected")
out_dir = Path("/work/09208/asperera/ls6/DATA/historical/tcn_animation")

out_dir.mkdir(parents=True, exist_ok=True)

# Match your binary detection GeoTIFFs
tif_files = sorted(input_dir.glob("*.tif"))

print(f"Found {len(tif_files)} tif files")

if len(tif_files) == 0:
    raise RuntimeError("No .tif files found.")


# -----------------------------
# Helper: extract year/date from filename
# -----------------------------
def label_from_filename(path: Path) -> str:
    """
    Example:
      1948August08_detection.tif -> 1948 August 08
      1976july18_clip_pred.tif   -> 1976 July 18
    """
    name = path.stem

    m = re.search(
        r"(19\d{2}|20\d{2})\s*([A-Za-z]+)?\s*(\d{1,2})?",
        name
    )

    if m:
        year = m.group(1)
        month = m.group(2) or ""
        day = m.group(3) or ""
        return f"{year} {month} {day}".strip()

    return name


# -----------------------------
# Use first raster as reference grid
# -----------------------------
with rasterio.open(tif_files[0]) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_height = ref.height
    ref_width = ref.width
    ref_bounds = array_bounds(ref_height, ref_width, ref_transform)

extent = [
    ref_bounds[0],  # left
    ref_bounds[2],  # right
    ref_bounds[1],  # bottom
    ref_bounds[3],  # top
]

frame_paths = []


# -----------------------------
# Create PNG frames
# -----------------------------
for i, tif_path in enumerate(tif_files, start=1):
    print(f"Rendering frame {i}: {tif_path.name}")

    with rasterio.open(tif_path) as src:
        arr = src.read(1)

        # Reproject/resample if grid does not match reference
        if (
            src.crs != ref_crs
            or src.transform != ref_transform
            or src.height != ref_height
            or src.width != ref_width
        ):
            aligned = np.zeros((ref_height, ref_width), dtype=np.uint8)

            reproject(
                source=arr,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )

            arr = aligned

    # Convert to clean binary mask
    mask = arr > 0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Background
    ax.imshow(
        np.zeros(mask.shape),
        cmap="gray",
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=1,
    )

    # Detection mask
    ax.imshow(
        mask.astype(float),
        cmap="Reds",
        extent=extent,
        origin="upper",
        alpha=0.85,
        vmin=0,
        vmax=1,
    )

    ax.set_title(label_from_filename(tif_path), fontsize=16)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")

    frame_path = out_dir / f"frame_{i:03d}.png"
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)

    frame_paths.append(frame_path)


# -----------------------------
# Create animated GIF
# -----------------------------
gif_path = out_dir / "historical_detections.gif"

frames = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(
    gif_path,
    frames,
    duration=1000.0,  # seconds per frame
    loop=2,
)

print(f"Saved GIF: {gif_path}")

# -----------------------------
# Create MP4
# -----------------------------
mp4_path = out_dir / "historical_detections.mp4"

print(f"Writing MP4 to: {mp4_path}")
print(f"MP4 suffix: {mp4_path.suffix}")

with imageio.get_writer(
    str(mp4_path),
    format="FFMPEG",
    mode="I",
    fps=1,
    codec="libx264",
    macro_block_size=16,
) as writer:
    for p in frame_paths:
        frame = imageio.imread(str(p))

        # MP4 prefers RGB, not RGBA
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        writer.append_data(frame)

print(f"Saved MP4: {mp4_path}")
