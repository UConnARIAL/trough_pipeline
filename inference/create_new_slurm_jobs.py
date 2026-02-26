#!/usr/bin/env python
"""
Code configured to execute multiple inference sessions on tacc frontera 4 GPU nodes
Based on a passing the directory of images to the model
Taking advantage of running multiple small jobs in parallel DDP
Depending on the Alaska Tundra mosiac data tiling structure for parsing the input

Usage
-----
bash -lc '
python create_multi_job.py --input_dir ./AlaskaTundraMosaic/imagery/ \
                    --output_dir ./AlaskaTundraMosaicMasks/ \
                    --script_dir ./TCN_pipe \
                    --st 2 --end 3
'
bash -lc '
python create_multi_job.py --config .config.toml \
                    --script_dir ./TCN_pipe \
                    --st 2 --end 3
'

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

# ----------------------------- TOML helpers (local fallback) -----------------------------
from gt_gpkg_common import load_toml, cfg_get

def create_slurm_script(sub_dir, output_dir, job_name, script_dir):
    """Generates a SLURM job script for a given subdirectory to execute."""
    script_content = f"""#!/bin/bash

#SBATCH -J {job_name}
#SBATCH -o {script_dir}/out/{job_name}.out
#SBATCH -e {script_dir}/err/{job_name}.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -A DPP20001

source /home1/09208/asperera/.bashrc
conda activate /scratch1/projects/PDG_shared/Troughs_Segformer/segformer-env-org

CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the distributed inference script
torchrun --nproc_per_node=1 inference_new_in_dir.py --input_dir={sub_dir} --output_dir={output_dir}
"""
    script_path = os.path.join(script_dir, f"{job_name}.job")
    with open(script_path, "w") as f:
        f.write(script_content)
    return script_path


def main(input_dir, output_dir, script_dir, st, end, run_job):
    sub_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    for i, sub_dir in enumerate(sub_dirs, start=0): print(f"{i}. {sub_dir}")
    selected_sub_dirs = sub_dirs[st:end]

    if not os.path.exists(script_dir):
        os.makedirs(script_dir)

    for i, sub_dir in enumerate(selected_sub_dirs):
        sub_dir_path = os.path.join(input_dir, sub_dir)
        out_dir_path = os.path.join(output_dir, sub_dir)
        job_name = f"j{i + st}_{sub_dir}"
        script_path = create_slurm_script(sub_dir_path, out_dir_path, job_name, script_dir)
        print(f"Created SLURM script: {script_path}")
        if (run_job):os.system("sbatch %s" % script_path)


def parse_args():
    p = argparse.ArgumentParser(description="Generate SLURM jobs for subdirectories.")

    # Optional config: only used to supply defaults for input/output dirs
    p.add_argument("--config", type=str, default=None, help="Optional config.toml (only IO defaults are used).")

    # These can come from config if not provided on CLI
    p.add_argument("--input_dir", type=str, default=None,
                   help="Input directory containing subdirectories (default from config if provided).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory where results should be stored (default from config if provided).")

    # These remain CLI-only (always required)
    p.add_argument("--script_dir", type=str, required=True,
                   help="Directory where SLURM scripts should be saved.")
    p.add_argument("--st", type=int, required=True,
                   help="Start index of subdirectories.")
    p.add_argument("--end", type=int, required=True,
                   help="End index (exclusive) of subdirectories.")

    # CLI only
    p.add_argument("--run_job", action="store_true", help="Execute the script after creating.")

    return p.parse_args()

def _get_io_defaults_from_cfg(cfg: dict):
    """
    Will get I/O from config.toml to match rest of the pipeline
    """
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

