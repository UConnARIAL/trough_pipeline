#!/usr/bin/env python3
"""
Report remaining GPKG work per tile:
  rank, tile_id, total_tiff, gpkg, balance[, seconds], pending_subtiles

Requires:
    from gt_gpkg_tile_resume_check_sub3 import filter_done_tiffs
which returns (remaining_tifs_list, skipped_count)
"""

import os
import glob
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from gt_gpkg_common import filter_done_tiffs

def discover_tiles(master_dir: str):
    return [e.name for e in os.scandir(master_dir) if e.is_dir()]


def fast_list(dirpath: str, pattern: str):
    simple_ext = (
        pattern.startswith("*.") and
        pattern.count("*") == 1 and
        "?" not in pattern and
        "/" not in pattern
    )
    if simple_ext:
        suffix = pattern[1:]  # ".tif"
        return [
            os.path.join(dirpath, e.name)
            for e in os.scandir(dirpath)
            if e.is_file() and e.name.endswith(suffix)
        ]
    else:
        return glob.glob(os.path.join(dirpath, pattern))


def _countish(x):
    if isinstance(x, int):
        return x
    try:
        return len(x)
    except Exception:
        try:
            return int(x)
        except Exception:
            return 0


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def extract_subtile_id(path: str) -> str:
    """
    Try to extract the sub-tile id from '*<subtile>_mask.tif'.
    1) Prefer digits/underscores block before '_mask.tif' (e.g., '12_7')
    2) Fallback: last token before '_mask.tif'
    3) Final fallback: basename without extension
    """
    name = os.path.basename(path)
    m = re.search(r'(\d+(?:_\d+)*)_mask\.tif$', name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'([^_]+)_mask\.tif$', name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(name)[0]

def process_one_tile(tile_id, master_dir, output_tiles_dir, ext, verify, verbose, profile):
    import time, os
    start = time.time()

    tile_img_dir = os.path.join(master_dir, tile_id)
    tile_out_dir = os.path.join(output_tiles_dir, tile_id)
    outdir_exists = os.path.isdir(tile_out_dir)  # <-- read-only guard

    tifs = fast_list(tile_img_dir, ext)
    total = len(tifs)

    if total == 0:
        remaining = []
        done = 0
        bal = 0
    elif not outdir_exists:
        # Do NOT create anything; just report that nothing is done yet.
        remaining = tifs
        done = 0
        bal = total
    else:
        remaining, done_raw = filter_done_tiffs(
            tifs, tile_out_dir, verify=verify, verbose=verbose
        )
        # _countish handles list/int return shapes
        done = _countish(done_raw)
        bal = len(remaining)

    # If your script also prints pending sub-tiles, keep this bit; otherwise remove:
    pending_ids = [extract_subtile_id(f) for f in remaining]
    pending_ids.sort(key=_natural_key)
    pending_str = ",".join(pending_ids)

    secs = time.time() - start
    return (tile_id, total, done, bal, secs if profile else None, pending_str)

def process_one_tile_old(tile_id, master_dir, output_tiles_dir, ext, verify, verbose, profile):
    import time
    start = time.time()

    tile_img_dir = os.path.join(master_dir, tile_id)
    tile_out_dir = os.path.join(output_tiles_dir, tile_id)

    tifs = fast_list(tile_img_dir, ext)
    total = len(tifs)

    if total == 0:
        done = 0
        remaining = []
        bal = 0
    else:
        remaining, done_raw = filter_done_tiffs(
            tifs, tile_out_dir, verify=verify, verbose=verbose
        )
        done = _countish(done_raw)
        bal = len(remaining)

    # Build a natural-sorted, comma-separated list of pending sub-tile ids
    pending_ids = []
    for f in remaining:
        sid = extract_subtile_id(f)
        pending_ids.append(sid)
    pending_ids = sorted(pending_ids, key=_natural_key)
    pending_str = ",".join(pending_ids)

    secs = time.time() - start
    return (tile_id, total, done, bal, secs if profile else None, pending_str)


def main():
    ap = argparse.ArgumentParser(description="Report remaining GPKG work per tile, including pending sub-tiles.")
    ap.add_argument("--master_dir", required=True, help="Root containing tile folders with imagery files")
    ap.add_argument("--output_tiles_dir", required=True, help="Root where per-tile outputs (GPKGs) are written")
    ap.add_argument("--ext", default="*.tif", help="Glob for imagery files inside each tile (default: *.tif)")
    ap.add_argument("--verbose_filter", action="store_true", help="Have filter_done_tiffs print its per-tile summary")
    ap.add_argument("--csv", default=None, help="Optional path to write CSV report")
    ap.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 8) * 2),
                    help="Number of parallel worker threads (default: ~2x CPU)")
    ap.add_argument("--verify", action="store_true", help="Enable expensive verification in filter_done_tiffs")
    ap.add_argument("--profile", action="store_true", help="Include per-tile seconds in output")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tiles = discover_tiles(args.master_dir)
    if not tiles:
        logging.error(f"No tile folders found in {args.master_dir}")
        return

    rows = []
    total_tiff = total_done = total_bal = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(
                process_one_tile,
                tile_id,
                args.master_dir,
                args.output_tiles_dir,
                args.ext,
                args.verify,
                args.verbose_filter,
                args.profile,
            ): tile_id
            for tile_id in tiles
        }
        for fut in as_completed(futs):
            tile = futs[fut]
            try:
                tile_id, total, done, bal, secs, pending_str = fut.result()
                rows.append((tile_id, total, done, bal, secs, pending_str))
                total_tiff += total
                total_done += done
                total_bal += bal
            except Exception as e:
                logging.error(f"[{tile}] ERROR: {e}")
                rows.append((tile, 0, 0, 0, None, ""))

    rows.sort(key=lambda r: _natural_key(r[0]))

    # Pretty print table with rank
    if args.profile:
        header = ("rank", "tile_id", "total_tiff", "gpkg", "balance", "seconds", "pending_subtiles")
        colw = (6, 28, 12, 8, 8, 10, 60)  # widen last as needed
        fmt = f"{{:>{colw[0]}}} {{:<{colw[1]}}} {{:>{colw[2]}}} {{:>{colw[3]}}} {{:>{colw[4]}}} {{:>{colw[5]}.2f}} {{:<{colw[6]}}}"
    else:
        header = ("rank", "tile_id", "total_tiff", "gpkg", "balance", "pending_subtiles")
        colw = (6, 28, 12, 8, 8, 60)
        fmt = f"{{:>{colw[0]}}} {{:<{colw[1]}}} {{:>{colw[2]}}} {{:>{colw[3]}}} {{:>{colw[4]}}} {{:<{colw[5]}}}"

    print(fmt.format(*header))
    print("-" * sum(colw))
    if args.profile:
        for i, (tile_id, total, done, bal, secs, pending) in enumerate(rows, 1):
            print(fmt.format(i, tile_id, total, done, bal, secs if secs is not None else 0.0, pending))
        print("-" * sum(colw))
        print(fmt.format("", "SUM", total_tiff, total_done, total_bal, 0.0, ""))
    else:
        for i, (tile_id, total, done, bal, _, pending) in enumerate(rows, 1):
            print(fmt.format(i, tile_id, total, done, bal, pending))
        print("-" * sum(colw))
        print(fmt.format("", "SUM", total_tiff, total_done, total_bal, ""))

    # Optional CSV (now with 'pending_subtiles')
    if args.csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            if args.profile:
                writer.writerow(["rank", "tile_id", "total_tiff", "gpkg", "balance", "seconds", "pending_subtiles"])
                for i, (tile_id, total, done, bal, secs, pending) in enumerate(rows, 1):
                    writer.writerow([i, tile_id, total, done, bal, f"{secs:.4f}" if secs else "", pending])
            else:
                writer.writerow(["rank", "tile_id", "total_tiff", "gpkg", "balance", "pending_subtiles"])
                for i, (tile_id, total, done, bal, _, pending) in enumerate(rows, 1):
                    writer.writerow([i, tile_id, total, done, bal, pending])

if __name__ == "__main__":
    main()
