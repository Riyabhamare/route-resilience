"""
PHASE 2 -- Occlusion-Severity Sweep (Controlled)

Evaluates Baseline U-Net and HDDNet across varying occlusion severity
on a FIXED set of validation images with SHARED occlusion masks.

Critical design: for each occlusion ratio, apply_focus_mim is called
ONCE per image to generate a single occluded version. That exact same
occluded tensor is fed to both models. This eliminates random variation
between models and makes the comparison perfectly controlled.

Requirements:
  - Occlusion ratios: [0.0, 0.1, 0.3, 0.5, 0.7]
  - Fixed 60 validation images (indices 300-359)
  - Shared occlusion masks per image per ratio

Usage:
  python scripts/occlusion_sweep.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm

from src.data.dataset import RoadDataset
from src.data.augmentations import get_val_transform, apply_focus_mim
from src.models.baseline_unet import get_baseline_model
from src.models.hddnet import HDDNet

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
BASELINE_CKPT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')

VAL_START = 300
VAL_SIZE = 60
THRESHOLD = 0.5

OCCLUSION_RATIOS = [0.0, 0.1, 0.3, 0.5, 0.7]


def compute_iou_from_tensors(pred_logits, mask):
    """Compute IoU from raw logits and binary mask (both on same device)."""
    pred_bin = (torch.sigmoid(pred_logits) > THRESHOLD).float()
    p = pred_bin.view(-1)
    t = mask.view(-1)
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    return tp / (tp + fp + fn + 1e-8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{'=' * 65}", flush=True)
    print(f"  PHASE 2: Occlusion-Severity Sweep (Controlled)", flush=True)
    print(f"  Both models see IDENTICAL occluded images per ratio.", flush=True)
    print(f"{'=' * 65}", flush=True)

    if not os.path.exists(BASELINE_CKPT):
        print(f"[ERROR] Baseline checkpoint not found: {BASELINE_CKPT}")
        sys.exit(1)
    if not os.path.exists(HDDNET_CKPT):
        print(f"[ERROR] HDDNet checkpoint not found: {HDDNET_CKPT}")
        sys.exit(1)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\nLoading validation data...", flush=True)
    val_dataset = RoadDataset(DATA_DIR, transform=get_val_transform())
    val_indices = list(range(VAL_START, VAL_START + VAL_SIZE))
    val_subset = Subset(val_dataset, val_indices)
    print(f"  Validation images: {len(val_subset)} (fixed set, indices {VAL_START}-{VAL_START + VAL_SIZE - 1})", flush=True)

    # Pre-load all validation images and masks into memory
    # This ensures both models see the exact same data
    print("  Pre-loading all images into memory...", flush=True)
    all_images = []  # List of tensors [C, H, W]
    all_masks = []   # List of tensors [1, H, W]
    for idx in range(len(val_subset)):
        img, mask = val_subset[idx]
        all_images.append(img)
        all_masks.append(mask)
    print(f"  Loaded {len(all_images)} images.", flush=True)

    # ================================================================
    # LOAD MODELS
    # ================================================================
    print("\nLoading models...", flush=True)
    baseline_model = get_baseline_model().to(device)
    baseline_ckpt_data = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt_data['model_state_dict'])
    baseline_model.eval()
    print(f"  Baseline: epoch {baseline_ckpt_data.get('epoch', '?')} (Full Dataset)", flush=True)

    hddnet_model = HDDNet().to(device)
    hddnet_ckpt_data = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(hddnet_ckpt_data['model_state_dict'])
    hddnet_model.eval()
    print(f"  HDDNet:   epoch {hddnet_ckpt_data.get('epoch', '?')}", flush=True)

    # ================================================================
    # SWEEP -- shared occlusion per image
    # ================================================================
    print("\nRunning sweep...", flush=True)
    baseline_ious = []
    hddnet_ious = []

    for ratio in OCCLUSION_RATIOS:
        print(f"\n  Ratio {ratio:.1f}:", flush=True)

        b_ious_this_ratio = []
        h_ious_this_ratio = []

        for img_idx in tqdm(range(len(all_images)), desc=f"    Ratio {ratio:.1f}", leave=False):
            img_tensor = all_images[img_idx]   # [C, H, W]
            mask_tensor = all_masks[img_idx]    # [1, H, W]

            # ---------------------------------------------------------
            # STEP 1: Generate ONE occluded image (shared by both models)
            # ---------------------------------------------------------
            if ratio > 0.0:
                img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                img_occluded_np = apply_focus_mim(img_np, occlusion_ratio=ratio, patch_size=32)
                img_occluded = torch.from_numpy(img_occluded_np).permute(2, 0, 1)  # [C, H, W]
            else:
                img_occluded = img_tensor  # No occlusion

            # Batch dimension: [1, C, H, W]
            x = img_occluded.unsqueeze(0).to(device)
            m = mask_tensor.unsqueeze(0).to(device)

            # ---------------------------------------------------------
            # STEP 2: Feed SAME occluded tensor to both models
            # ---------------------------------------------------------
            with torch.no_grad():
                # Baseline
                baseline_out = baseline_model(x)
                b_iou = compute_iou_from_tensors(baseline_out, m)
                b_ious_this_ratio.append(b_iou)

                # HDDNet (same x, same occlusion pattern)
                hddnet_out, _, _ = hddnet_model(x)
                h_iou = compute_iou_from_tensors(hddnet_out, m)
                h_ious_this_ratio.append(h_iou)

        b_mean = np.mean(b_ious_this_ratio)
        h_mean = np.mean(h_ious_this_ratio)
        baseline_ious.append(b_mean)
        hddnet_ious.append(h_mean)
        delta = h_mean - b_mean
        sign = '+' if delta >= 0 else ''
        print(f"    Baseline IoU: {b_mean:.4f}  |  HDDNet IoU: {h_mean:.4f}  |  Delta: {sign}{delta:.4f}", flush=True)

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESULTS: IoU vs Occlusion Severity (Controlled)", flush=True)
    print(f"  Both models evaluated on IDENTICAL occluded images.", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  {'Ratio':<8} | {'Baseline IoU':>14} | {'HDDNet IoU':>14} | {'Delta':>8}", flush=True)
    print(f"  {'-' * 8}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}", flush=True)
    for i, ratio in enumerate(OCCLUSION_RATIOS):
        b_iou = baseline_ious[i]
        h_iou = hddnet_ious[i]
        delta = h_iou - b_iou
        sign = '+' if delta >= 0 else ''
        print(f"  {ratio:<8.1f} | {b_iou:>14.4f} | {h_iou:>14.4f} | {sign}{delta:>7.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ================================================================
    # CHART (presentation-ready)
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(OCCLUSION_RATIOS, baseline_ious,
            marker='o', linestyle='--', color='#E74C3C', linewidth=2.5,
            markersize=9, label='Baseline U-Net', zorder=5)
    ax.plot(OCCLUSION_RATIOS, hddnet_ious,
            marker='s', linestyle='-', color='#2E86C1', linewidth=2.5,
            markersize=9, label='HDDNet (Ours)', zorder=5)

    # Title + subtitle as suptitle/ax.set_title combo (no overlap with x-axis)
    fig.suptitle('Route Resilience: Occlusion Robustness',
                 fontsize=18, fontweight='bold', y=0.97)
    ax.set_title('Both models evaluated on identical occluded images (controlled comparison)',
                 fontsize=11, fontstyle='italic', color='gray', pad=8)

    ax.set_xlabel('Occlusion Ratio (Focus-MIM)', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean IoU', fontsize=14, labelpad=10)
    ax.set_xticks(OCCLUSION_RATIOS)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0.0, 0.65)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=13, loc='lower left', framealpha=0.9)

    # Data labels with offset to avoid overlap
    for i, ratio in enumerate(OCCLUSION_RATIOS):
        ax.annotate(f"{baseline_ious[i]:.3f}",
                    (ratio, baseline_ious[i]),
                    textcoords="offset points", xytext=(0, -18),
                    ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
        ax.annotate(f"{hddnet_ious[i]:.3f}",
                    (ratio, hddnet_ious[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=10, color='#2E86C1', fontweight='bold')

    plot_path = os.path.join(OUTPUT_DIR, 'occlusion_sweep.png')
    fig.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {plot_path}", flush=True)

    # Save raw data
    data_path = os.path.join(OUTPUT_DIR, 'occlusion_sweep_data.txt')
    with open(data_path, 'w') as f:
        f.write("# Occlusion Sweep Results (Controlled)\n")
        f.write("# Both models evaluated on IDENTICAL occluded images per ratio.\n")
        f.write(f"# Baseline: {BASELINE_CKPT}\n")
        f.write(f"# HDDNet:   {HDDNET_CKPT}\n")
        f.write(f"# Val images: {VAL_SIZE} (indices {VAL_START}-{VAL_START + VAL_SIZE - 1})\n\n")
        f.write("ratio,baseline_iou,hddnet_iou,delta\n")
        for i, ratio in enumerate(OCCLUSION_RATIOS):
            delta = hddnet_ious[i] - baseline_ious[i]
            f.write(f"{ratio:.1f},{baseline_ious[i]:.6f},{hddnet_ious[i]:.6f},{delta:.6f}\n")
    print(f"  Raw data:  {data_path}", flush=True)

    print(f"\n{'=' * 65}", flush=True)
    print(f"  Phase 2 occlusion sweep complete (controlled).", flush=True)
    print(f"{'=' * 65}", flush=True)


if __name__ == '__main__':
    main()
