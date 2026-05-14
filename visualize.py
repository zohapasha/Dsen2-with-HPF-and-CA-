"""Lightweight visualization: run model on a few samples and save images.

Saves per-sample comparison (`guide`, `prediction`, `target`, `error`) and
per-band grayscale images to the `outputs/visuals` folder.

Designed to run quickly on a small indexed subset (default caps set small).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL, TARGET_BANDS_20M, TRAINING
from data_loader import WaldProtocolDataset
from model import EnhancedDSen2


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    mn = float(np.min(image))
    mx = float(np.max(image))
    if mx <= mn:
        return np.zeros_like(image)
    return (image - mn) / (mx - mn)


def _rgb_from_tensor(tensor: torch.Tensor, channels: Tuple[int, int, int]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    image = array[list(channels), :, :]
    image = np.transpose(image, (1, 2, 0))
    return _normalize_for_display(image)


def save_comparison(sample, pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Guide uses actual RGB bands (B04, B03, B02 at indices 2, 1, 0)
    guide_rgb = _rgb_from_tensor(sample["guide_20m"], (2, 1, 0))
    
    # 2. Extract arrays for Target and Prediction
    target_array = sample["target"].detach().cpu().numpy()
    pred_array = pred.detach().cpu().numpy()
    if pred_array.ndim == 4:
        pred_array = pred_array[0]
        
    # 3. Grab the first 3 predicted bands (B05, B06, B07) to use as False-Color RGB
    target_fc = np.transpose(target_array[[0, 1, 2], :, :], (1, 2, 0))
    pred_fc = np.transpose(pred_array[[0, 1, 2], :, :], (1, 2, 0))
    
    # 4. Calculate the visual scale ONLY from the ground truth Target
    t_mn = float(np.min(target_fc))
    t_mx = float(np.max(target_fc))
    
    def scale(img):
        if t_mx <= t_mn: return np.zeros_like(img)
        return np.clip((img - t_mn) / (t_mx - t_mn), 0, 1)

    target_rgb = scale(target_fc)
    pred_rgb = scale(pred_fc)
    
    # 5. Calculate Error
    error_map = np.mean(np.abs(target_array - pred_array), axis=0)

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(guide_rgb)
    axes[0].set_title("Guide (20m True Color)")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction (False Color)")
    axes[2].imshow(target_rgb)
    axes[2].set_title("Target (False Color)")
    axes[3].imshow(_normalize_for_display(error_map), cmap="magma")
    axes[3].set_title("Mean Abs Error")
    
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_band_grays(sample, pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = sample["target"].detach().cpu().numpy()
    pred = pred.squeeze(0).detach().cpu().numpy()
    blurry = sample["blurry_20m"].detach().cpu().numpy()
    for i, band in enumerate(TARGET_BANDS_20M):
        t = _normalize_for_display(target[i])
        p = _normalize_for_display(pred[i])
        b = _normalize_for_display(blurry[i])
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(b, cmap="gray"); axes[0].set_title(f"Blurry {band}"); axes[0].axis('off')
        axes[1].imshow(p, cmap="gray"); axes[1].set_title(f"Pred {band}"); axes[1].axis('off')
        axes[2].imshow(t, cmap="gray"); axes[2].set_title(f"Target {band}"); axes[2].axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f"band_{band}.png", dpi=160)
        plt.close(fig)


def _format_band_scores(scores: np.ndarray, band_names: list[str]) -> str:
    pairs = sorted(zip(band_names, scores.tolist()), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{band}:{score:.4f}" for band, score in pairs)


def _input_band_importance(model: EnhancedDSen2, sample: dict, device: torch.device) -> np.ndarray:
    base_input = sample["input"].unsqueeze(0).to(device)
    with torch.no_grad():
        baseline = model(base_input)

    scores = []
    for band_index in range(base_input.shape[1]):
        perturbed = base_input.clone()
        perturbed[:, band_index : band_index + 1, :, :] = 0.0
        with torch.no_grad():
            output = model(perturbed)
        delta = torch.mean(torch.abs(output - baseline)).item()
        scores.append(delta)
    return np.asarray(scores, dtype=np.float32)


def _block_attention_report(attentions: list[torch.Tensor], band_names: list[str]) -> list[str]:
    report = []
    for block_index, attention in enumerate(attentions):
        scores = attention.detach().mean(dim=0).cpu().numpy()
        report.append(f"block_{block_index}: {_format_band_scores(scores, band_names)}")
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=int, default=4, help="Number of samples to visualize")
    p.add_argument("--max-items", type=int, default=2)
    p.add_argument("--max-patches-per-item", type=int, default=8)
    p.add_argument("--max-total-patches", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/visuals"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WaldProtocolDataset(
        max_patches_per_item=args.max_patches_per_item,
        max_items_to_use=args.max_items,
        max_total_patches=args.max_total_patches,
        verbose=True,
    )

    model = EnhancedDSen2(
        input_channels=MODEL.input_channels,
        output_channels=MODEL.output_channels,
        base_channels=MODEL.base_channels,
        num_residual_blocks=MODEL.num_residual_blocks,
        se_reduction=MODEL.se_reduction,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    n = min(args.samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred, attentions = model.forward_with_attention(inp)
            pred = pred.cpu()

        comp_path = args.output_dir / f"sample_{i}_comparison.png"
        save_comparison(sample, pred, comp_path)
        save_band_grays(sample, pred, args.output_dir / f"sample_{i}_bands")
        band_importance = _input_band_importance(model, sample, device)
        input_band_names = [f"input_{index}" for index in range(band_importance.shape[0])]
        feature_band_names = [f"feat_{index}" for index in range(attentions[0].shape[1])] if attentions else []
        print(f"Sample {i} input-band occlusion: {_format_band_scores(band_importance, input_band_names)}")
        for line in _block_attention_report(attentions, feature_band_names):
            print(f"Sample {i} {line}")
        print(f"Wrote visuals for sample {i} -> {comp_path}")


if __name__ == '__main__':
    main()
