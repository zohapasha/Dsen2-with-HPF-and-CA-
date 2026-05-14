"""Diagnostics: time per-sample reads and print asset hrefs used by WaldProtocolDataset.

Run on real STAC data to identify slow COG assets.
Example:
  python diagnostics.py --max-items 2 --max-patches-per-item 8 --max-total-patches 16 --num-samples 4

For quick local test with synthetic data:
  python diagnostics.py --synthetic --synthetic-samples 4 --num-samples 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_loader import WaldProtocolDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-patches-per-item", type=int, default=0)
    p.add_argument("--max-total-patches", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=4, help="Number of dataset samples to time")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-samples", type=int, default=16)
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.synthetic:
        # import local SyntheticDataset from train to avoid circular imports
        from train import SyntheticDataset

        ds = SyntheticDataset(num_samples=args.synthetic_samples)
        print(f"Using synthetic dataset with {len(ds)} samples")
        n = min(len(ds), args.num_samples)
        for i in range(n):
            t0 = time.perf_counter()
            _ = ds[i]
            t1 = time.perf_counter()
            print(f"sample {i} read time: {t1-t0:.3f}s")
        return

    max_patches = args.max_patches_per_item if args.max_patches_per_item > 0 else None
    max_items = args.max_items if args.max_items > 0 else None
    max_total = args.max_total_patches if args.max_total_patches > 0 else None

    ds = WaldProtocolDataset(
        max_patches_per_item=max_patches,
        max_items_to_use=max_items,
        max_total_patches=max_total,
        cache_dir=args.cache_dir,
        verbose=True,
    )
    print(f"Dataset built: {len(ds)} samples, patch_index len {len(ds.patch_index)}")

    n = min(len(ds), args.num_samples)
    for i in range(n):
        patch = ds.patch_index[i]
        item = ds.items[patch.item_index]
        print(f"\nTiming sample {i} -> scene id: {item.id} (item_index {patch.item_index})")
        # print candidate asset hrefs for inspection
        hrefs = {}
        for band in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]:
            if band in item.assets:
                hrefs[band] = item.assets[band].href
        for band, href in hrefs.items():
            print(f"  asset {band}: {href}")

        t0 = time.perf_counter()
        try:
            sample = ds[i]
            t1 = time.perf_counter()
            print(f"  read success in {t1-t0:.3f}s | input shape: {sample['input'].shape}")
        except Exception as e:
            t1 = time.perf_counter()
            print(f"  read failed after {t1-t0:.3f}s: {e}")


if __name__ == '__main__':
    main()
