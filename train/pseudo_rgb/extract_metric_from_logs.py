#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE = Path("/home1/09208/asperera/PDG_shared2/TCN_Training/")

runs = [
    "gray_adapter_w32_b2",
    "gray_adapter_w64_b4_dil",
    "gray_adapter_w64_b4_nodil",
    "gray_adapter_w64_b4_dil_recon001",
    "gray_adapter_w96_b4_dil",
    "gray_adapter_w96_b6_dil",
    "gray_adapter_w128_b4_dil",
    "gray_adapter_w64_b6_dil",
]

rows = []

for run in runs:
    log_path = BASE / run / "training_log.csv"

    if not log_path.exists():
        print(f"Missing: {log_path}")
        continue

    df = pd.read_csv(log_path)
    df["run"] = run

    best = df.loc[df["val_f1"].idxmax()].copy()
    rows.append(best)

summary = pd.DataFrame(rows)

cols = [
    "run",
    "epoch",
    "val_f1",
    "val_iou",
    "val_precision",
    "val_recall",
    "train_f1",
    "train_iou",
    "threshold",
    "adapter_width",
    "adapter_blocks",
    "use_dilation",
    "lambda_recon",
]

summary = summary[cols].sort_values("val_f1", ascending=False)

out_csv = BASE / "gray_adapter_4run_summary.csv"
summary.to_csv(out_csv, index=False)

print(summary.to_string(index=False))
print(f"\nWrote: {out_csv}")