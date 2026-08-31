#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt


def sample_raster(path, indexes, max_pixels=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    samples = [[] for _ in indexes]

    with rasterio.open(path) as src:
        nodata = src.nodata

        # Randomly sample block windows to avoid reading huge rasters fully.
        windows = [w for _, w in src.block_windows(1)]

        if len(windows) == 0:
            windows = [rasterio.windows.Window(0, 0, src.width, src.height)]

        rng.shuffle(windows)

        collected = 0

        for window in windows:
            arr = src.read(indexes, window=window)  # [C, H, W]

            valid = np.ones(arr.shape[1:], dtype=bool)

            if nodata is not None:
                valid &= np.all(arr != nodata, axis=0)

            valid &= np.all(np.isfinite(arr.astype(np.float32)), axis=0)

            ys, xs = np.where(valid)
            if len(xs) == 0:
                continue

            remaining = max_pixels - collected
            if remaining <= 0:
                break

            n = min(len(xs), remaining)
            pick = rng.choice(len(xs), size=n, replace=False)

            for c in range(len(indexes)):
                samples[c].append(arr[c, ys[pick], xs[pick]].astype(np.float32))

            collected += n

    out = []
    for s in samples:
        if len(s) == 0:
            out.append(np.array([], dtype=np.float32))
        else:
            out.append(np.concatenate(s))

    return out


def summarize(samples, label, indexes):
    rows = []

    for vals, idx in zip(samples, indexes):
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        pct = np.percentile(vals, [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100])

        rows.append({
            "image_set": label,
            "band_index_read": idx,
            "n": len(vals),
            "min": pct[0],
            "p01": pct[1],
            "p02": pct[2],
            "p05": pct[3],
            "p10": pct[4],
            "p25": pct[5],
            "p50": pct[6],
            "p75": pct[7],
            "p90": pct[8],
            "p95": pct[9],
            "p98": pct[10],
            "p99": pct[11],
            "max": pct[12],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        })

    return pd.DataFrame(rows)


def plot_hists(orig_samples, pseudo_samples, orig_indexes, pseudo_indexes, out_png):
    for k in range(3):
        plt.figure(figsize=(7, 4))

        o = orig_samples[k]
        p = pseudo_samples[k]

        if len(o) > 0:
            plt.hist(o, bins=100, alpha=0.5, density=True, label=f"Original band {orig_indexes[k]}")
        if len(p) > 0:
            plt.hist(p, bins=100, alpha=0.5, density=True, label=f"Pseudo band {pseudo_indexes[k]}")

        plt.title(f"Distribution comparison: channel {k+1}")
        plt.xlabel("Pixel value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        out_path = out_png.with_name(out_png.stem + f"_channel{k+1}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True, help="Original multiband Maxar image")
    parser.add_argument("--pseudo", required=True, help="New gray-to-3-band pseudo RGB image")
    parser.add_argument("--orig-indexes", nargs=3, type=int, default=[8, 5, 1])
    parser.add_argument("--pseudo-indexes", nargs=3, type=int, default=[3, 2, 1])
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--out-prefix", default="band_distribution_compare")

    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)

    print("Sampling original:", args.orig)
    orig_samples = sample_raster(args.orig, args.orig_indexes, args.max_pixels)

    print("Sampling pseudo:", args.pseudo)
    pseudo_samples = sample_raster(args.pseudo, args.pseudo_indexes, args.max_pixels)

    df_orig = summarize(orig_samples, "original", args.orig_indexes)
    df_pseudo = summarize(pseudo_samples, "pseudo", args.pseudo_indexes)

    df = pd.concat([df_orig, df_pseudo], ignore_index=True)

    out_csv = out_prefix.with_suffix(".csv")
    df.to_csv(out_csv, index=False)

    print("\nSummary:")
    print(df.to_string(index=False))

    print(f"\nWrote {out_csv}")

    plot_hists(
        orig_samples,
        pseudo_samples,
        args.orig_indexes,
        args.pseudo_indexes,
        out_prefix.with_suffix(".png"),
    )


if __name__ == "__main__":
    main()

"""
USAGE
python compare_band_distributions.py \
  --orig /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_All_bands/images/Jago_23.tif \
  --pseudo /home1/09208/asperera/PDG_shared2/TCN_Training/TCN_train_GRAY/gray_rgb_train/images/Jago_23.tif \
  --orig-indexes 8 5 1 \
  --pseudo-indexes 3 2 1 \
  --out-prefix compare_851_vs_pseudo


"""