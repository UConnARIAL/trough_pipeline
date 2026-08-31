from pathlib import Path
import rasterio
import numpy as np
import sys


def inspect(path):
    path = Path(path)

    with rasterio.open(path) as src:
        arr = src.read()

        print("\n", "=" * 80)
        print(path)
        print("count:", src.count)
        print("shape:", arr.shape)
        print("dtype:", arr.dtype)
        print("crs:", src.crs)
        print("transform:", src.transform)
        print("nodata:", src.nodata)

        for b in range(arr.shape[0]):
            band = arr[b]
            finite = np.isfinite(band)

            if finite.sum() == 0:
                print(f"band {b+1}: no finite pixels")
                continue

            vals = band[finite]
            print(
                f"band {b+1}: "
                f"min={vals.min()}, "
                f"p1={np.percentile(vals, 1):.3f}, "
                f"p50={np.percentile(vals, 50):.3f}, "
                f"p99={np.percentile(vals, 99):.3f}, "
                f"max={vals.max()}, "
                f"zero_frac={(vals == 0).mean():.4f}"
            )

        if arr.shape[0] >= 3:
            all_zero = (arr[:3] == 0).all(axis=0)
            print("all first-3-bands zero fraction:", all_zero.mean())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)

""""
python inspect_tif_stats.py \
  /home1/09208/asperera/PDG_shared2/TCN_Training/georef/images/chip_0070.tif \
  /home1/09208/asperera/PDG_shared2/TCN_Training/test_1024/images/chip_0070_0.tif
"""