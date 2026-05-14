"""Training entry point for the enhanced DSen2 Sentinel-2 super-resolution model."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from config import MODEL, TRAINING
from data_loader import WaldProtocolDataset, build_dataloader, split_dataset
from model import EnhancedDSen2


# Synthetic dataset for testing without STAC access
class SyntheticDataset(torch.utils.data.Dataset):
    """Random patches for testing without STAC/network access."""
    def __init__(self, num_samples: int = 16, patch_size: int = 64, seed: int = 42):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.seed = seed
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed + idx)
        guide_20m = torch.randn(4, self.patch_size, self.patch_size, generator=gen) * 0.2 + 0.5
        target_20m = torch.randn(6, self.patch_size, self.patch_size, generator=gen) * 0.2 + 0.5
        blurry_20m = torch.randn(6, self.patch_size, self.patch_size, generator=gen) * 0.2 + 0.45
        return {
            "input": torch.cat([torch.clamp(guide_20m, 0, 1), torch.clamp(blurry_20m, 0, 1)], dim=0).float(),
            "target": torch.clamp(target_20m, 0, 1).float(),
            "guide_20m": torch.clamp(guide_20m, 0, 1).float(),
            "blurry_20m": torch.clamp(blurry_20m, 0, 1).float(),
            "metadata": torch.tensor([0, 0, 0], dtype=torch.int64),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return inputs, targets


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    for batch in dataloader:
        inputs, targets = _move_batch_to_device(batch, device)
        predictions = model(inputs)
        squared_error += torch.sum((predictions - targets) ** 2).item()
        pixel_count += targets.numel()
    return math.sqrt(squared_error / max(1, pixel_count))


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    model.train()
    running_squared_error = 0.0
    pixel_count = 0
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs, targets = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        running_squared_error += torch.sum((predictions.detach() - targets) ** 2).item()
        pixel_count += targets.numel()

        if batch_idx % 50 == 0 or batch_idx == total_batches:
            current_rmse = math.sqrt(running_squared_error / max(1, pixel_count))
            print(f"  batch {batch_idx}/{total_batches} | running RMSE: {current_rmse:.6f}")

    return math.sqrt(running_squared_error / max(1, pixel_count))


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_rmse: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the enhanced DSen2 model")
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--num-workers", type=int, default=TRAINING.num_workers)
    parser.add_argument("--learning-rate", type=float, default=TRAINING.learning_rate)
    parser.add_argument("--validation-split", type=float, default=TRAINING.validation_split)
    parser.add_argument("--checkpoint-dir", type=Path, default=TRAINING.checkpoint_dir)
    parser.add_argument("--seed", type=int, default=TRAINING.seed)
    parser.add_argument(
        "--max-patches-per-item",
        type=int,
        default=512,
        help="Maximum number of sampled patches per scene (use <=0 for unlimited)",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Limit number of STAC scenes to use (0 unlimited)")
    parser.add_argument("--max-total-patches", type=int, default=0, help="Limit total patches across scenes (0 unlimited)")
    parser.add_argument("--prefetch-cache", type=str, default=None, help="Directory to prefetch patches into and load from")
    parser.add_argument("--prefetch-max", type=int, default=256, help="Max patches to prefetch into cache")
    parser.add_argument("--dry-run", action="store_true", help="Index dataset, optionally prefetch and estimate epoch time, then exit")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing (no STAC required)")
    parser.add_argument("--synthetic-samples", type=int, default=32, help="Number of synthetic samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset with fallback to synthetic
    if args.synthetic:
        dataset = SyntheticDataset(num_samples=args.synthetic_samples, seed=args.seed)
        print(f"Using synthetic dataset with {args.synthetic_samples} samples")
    else:
        try:
            max_patches = args.max_patches_per_item if args.max_patches_per_item > 0 else None
            max_items_to_use = args.max_items if args.max_items > 0 else None
            max_total_patches = args.max_total_patches if args.max_total_patches > 0 else None
            dataset = WaldProtocolDataset(
                max_patches_per_item=max_patches,
                max_items_to_use=max_items_to_use,
                max_total_patches=max_total_patches,
                seed=args.seed,
                verbose=True,
                cache_dir=args.prefetch_cache,
            )
            print("Successfully loaded STAC dataset")
        except RuntimeError as e:
            print(f"STAC loading failed: {e}")
            print("Falling back to synthetic dataset. To use real data, ensure network connectivity.")
            dataset = SyntheticDataset(num_samples=args.synthetic_samples, seed=args.seed)
    print(f"Dataset size: {len(dataset)} samples")

    # quick read test of first sample to expose remote read errors early
    try:
        _ = dataset[0]
        print("Sample read test: OK")
    except Exception as exc:
        print(f"Warning: reading a sample failed: {exc}")

    # prefetch diagnostic: time reading a few samples to detect slow remote I/O
    try:
        import time
        prefetch_n = min(len(dataset), max(1, args.batch_size), 8)
        print(f"Prefetch diagnostic: timing read of {prefetch_n} samples...")
        times = []
        for i in range(prefetch_n):
            t0 = time.perf_counter()
            _ = dataset[i]
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            print(f"  sample {i} read time: {elapsed:.3f}s")
        avg = sum(times) / len(times)
        print(f"Prefetch diagnostic complete. avg sample read time: {avg:.3f}s")
        if avg > 2.0:
            print("Warning: average sample read time is high — remote COG reads may be slow. Consider reducing caps or using --synthetic.")
    except Exception as exc:
        print(f"Prefetch diagnostic failed: {exc}")

    # optional prefetch to cache (writes .npz files)
    if args.prefetch_cache is not None:
        if hasattr(dataset, "prefetch_to_cache"):
            try:
                print(f"Prefetching up to {args.prefetch_max} patches into {args.prefetch_cache} ...")
                written = dataset.prefetch_to_cache(args.prefetch_cache, max_files=args.prefetch_max)
                print(f"Prefetch complete: wrote {written} files")
            except Exception as exc:
                print(f"Prefetch to cache failed: {exc}")
        else:
            print("Prefetch cache requested but dataset does not support prefetching; skipping.")

    # dry-run mode: estimate epoch time and exit
    if args.dry_run:
        try:
            # use measured avg if available, else re-run a tiny measurement
            measured_avg = locals().get("avg", None)
            if measured_avg is None:
                import time as _time
                t0 = _time.perf_counter()
                _ = dataset[0]
                measured_avg = _time.perf_counter() - t0
            est_sample_proc = 0.01  # rough model processing per sample
            est_epoch_time = len(dataset) * (measured_avg + est_sample_proc)
            print(f"Dry-run: dataset size {len(dataset)} samples")
            print(f"Dry-run: avg sample read time {measured_avg:.3f}s, est epoch time {est_epoch_time/60.0:.2f} minutes")
        except Exception as exc:
            print(f"Dry-run estimation failed: {exc}")
        return
    
    train_dataset, val_dataset = split_dataset(dataset, validation_split=args.validation_split, seed=args.seed)
    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = EnhancedDSen2(
        input_channels=MODEL.input_channels,
        output_channels=MODEL.output_channels,
        base_channels=MODEL.base_channels,
        num_residual_blocks=MODEL.num_residual_blocks,
        se_reduction=MODEL.se_reduction,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True  # type: ignore[call-arg]
        )
    except TypeError as exc:
        if "verbose" not in str(exc):
            raise
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    best_val_rmse = float("inf")
    last_checkpoint = args.checkpoint_dir / "enhanced_dsen2_last.pt"
    best_checkpoint = args.checkpoint_dir / "enhanced_dsen2_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_rmse = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_norm=TRAINING.gradient_clip_norm,
        )
        val_rmse = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        save_checkpoint(last_checkpoint, model, optimizer, epoch, best_val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_val_rmse)

        print(f"Epoch {epoch:03d} | train RMSE: {train_rmse:.6f} | val RMSE: {val_rmse:.6f} | best: {best_val_rmse:.6f}")


if __name__ == "__main__":
    main()
