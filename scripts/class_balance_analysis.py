"""
Class balance analysis for SVAMITVA dataset.

This script analyzes the frequency of each class in the masks across all dataset samples.
Run this to identify rare classes and inform loss weighting or sampling strategies.
"""

import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from data.dataset import SVAMITVADataset

# Set your dataset directory and config
DATA_DIR = Path("data")
IMAGE_SIZE = 512

# Instantiate dataset (modify as needed for your setup)
dataset = SVAMITVADataset(
    root_dir=DATA_DIR,
    image_size=IMAGE_SIZE,
    split="train",
)

class_counts = {}

for idx in tqdm(range(len(dataset)), desc="Analyzing class balance"):
    sample = dataset[idx]
    for key, mask in sample.items():
        if key.endswith("_mask") and isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
            unique, counts = np.unique(mask_np, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts.setdefault(key, {}).setdefault(int(u), 0)
                class_counts[key][int(u)] += int(c)

print("\nClass balance summary:")
for task, counts in class_counts.items():
    print(f"{task}:")
    total = sum(counts.values())
    for cls, cnt in counts.items():
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  Class {cls}: {cnt} pixels ({pct:.2f}%)")
