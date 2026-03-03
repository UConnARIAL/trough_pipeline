#!/usr/bin/env python
"""
Generate 1 SLURM job per input subdirectory. to take advantage of multiple small jobs
Each job allocates 4 GPUs on a single node and shards the .tif list into 4
node-local shard dirs (symlinks), then runs 4 independent inference processes (one per GPU)

This is coded to for specific hpc node type with 4x RTX5000 gpus with 16GB gpu and 128GB CPU memory
If used for other configs test for OOM issues.

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera
"""

import os
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]  # one level up from /inference
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gt_gpkg_common import load_toml, cfg_get

def create_slurm_script(sub_dir, output_dir, job_name, script_dir):
    """Generates a SLURM job script for a given subdirectory to execute."""
    script_content = f"""#!/bin/bash

#SBATCH -J {job_name}
#SBATCH -o {script_dir}/out/{job_name}.out
#SBATCH -e {script_dir}/err/{job_name}.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH -t 16:00:00
#SBATCH -A TG-NAIRR240088

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch2/projects/PDG_shared/CONDA_ENV/segformer-env-gpkg3

IN_DIR="{sub_dir}"
OUT_DIR="{output_dir}"
mkdir -p "$OUT_DIR"

BASE_TMP="${{SLURM_TMPDIR:-/tmp}}"
JOB_TMP="$BASE_TMP/tcn_${{SLURM_JOB_ID}}_{job_name}"
mkdir -p "$JOB_TMP"

# 4 shard dirs
for r in 0 1 2 3; do
  mkdir -p "$JOB_TMP/in_shard_$r"
done

# Collect + sort tif files
mapfile -t FILES < <(find "$IN_DIR" -maxdepth 1 -type f -name '*.tif' | sort)
if [[ ${{#FILES[@]}} -eq 0 ]]; then
  echo "No .tif files found in $IN_DIR"
  exit 0
fi

# Round-robin symlink into shards
for i in "${{!FILES[@]}}"; do
  r=$(( i % 4 ))
  ln -sf "${{FILES[$i]}}" "$JOB_TMP/in_shard_$r/"
done

# Launch 4 independent processes (one per GPU)
for r in 0 1 2 3; do
  if compgen -G "$JOB_TMP/in_shard_$r/*.tif" > /dev/null; then
    srun --exclusive -n 1 bash -lc \\
      "export CUDA_VISIBLE_DEVICES=$r; \\
       python segf_inference_tcn.py --input_dir='$JOB_TMP/in_shard_$r' --output_dir='$OUT_DIR'" &
  else
    echo "Shard $r empty; skipping."
  fi
done

wait
echo "Done: $IN_DIR -> $OUT_DIR"
"""
    script_path = os.path.join(script_dir, f"{job_name}.job")
    with open(script_path, "w") as f:
        f.write(script_content)
    return script_path






def main(input_dir, output_dir, script_dir, st, end, run_job):
    sub_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    for i, sub_dir in enumerate(sub_dirs, start=0):
        print(f"{i}. {sub_dir}")

    selected_sub_dirs = sub_dirs[st:end]

    # Ensure script_dir + out/err exist
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(os.path.join(script_dir, "out"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "err"), exist_ok=True)

    for i, sub_dir in enumerate(selected_sub_dirs):
        sub_dir_path = os.path.join(input_dir, sub_dir)
        out_dir_path = os.path.join(output_dir, sub_dir)  # preserves tile/subtile tree
        job_name = f"j{i + st}_{sub_dir}"

        script_path = create_slurm_script(sub_dir_path, out_dir_path, job_name, script_dir)
        print(f"Created SLURM script: {script_path}")
        if run_job:
            os.system(f"sbatch {script_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate SLURM jobs for subdirectories.")

    p.add_argument("--config", type=str, default=None, help="Optional config.toml (only IO defaults are used).")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Input directory containing subdirectories (default from config if provided).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory where results should be stored (default from config if provided).")

    p.add_argument("--script_dir", type=str, required=True, help="Directory where SLURM scripts should be saved.")
    p.add_argument("--st", type=int, required=True, help="Start index of subdirectories.")
    p.add_argument("--end", type=int, required=True, help="End index (exclusive) of subdirectories.")
    p.add_argument("--run_job", action="store_true", help="Execute the script after creating.")

    return p.parse_args()


def _get_io_defaults_from_cfg(cfg: dict):
    input_dir = cfg_get(cfg, "io", "input_img_dir", default=None)
    output_dir = cfg_get(cfg, "io", "model_mask_dir", default=None)
    return input_dir, output_dir


def resolve_io(args):
    cfg_input = cfg_output = None
    if args.config:
        cfg = load_toml(args.config)
        cfg_input, cfg_output = _get_io_defaults_from_cfg(cfg)

    input_dir = args.input_dir or cfg_input
    output_dir = args.output_dir or cfg_output

    missing = []
    if not input_dir:
        missing.append("input_dir (via --input_dir or config)")
    if not output_dir:
        missing.append("output_dir (via --output_dir or config)")
    if missing:
        raise SystemExit("Missing required IO: " + ", ".join(missing))

    return input_dir, output_dir


if __name__ == "__main__":
    args = parse_args()
    input_dir, output_dir = resolve_io(args)
    main(input_dir, output_dir, args.script_dir, args.st, args.end, args.run_job)

"""
python create_new_slurm_job_mult_gpu.py --input_dir /scratch1/projects/PDG_shared/AlaskaTundraMosaic/imagery/ --output_dir /scratch1/projects/PDG_shared/AlaskaTundraMosaicMasks/ --script_dir /scratch1/projects/PDG_shared/Troughs_Segformer/trough_pipe --st 2 --end 3
"""
