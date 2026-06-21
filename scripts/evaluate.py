"""
Section 12 -- Evaluation & Baseline Comparison

Evaluates and compares any combination of:
  - Baseline U-Net  (models/baseline/baseline_unet_best.pth)
  - HDDNet          (user-specified checkpoint path)

Usage:
  # Evaluate baseline only
  python scripts/evaluate.py --baseline-only

  # Evaluate both (provide HDDNet checkpoint)
  python scripts/evaluate.py --hddnet-ckpt models/hddnet/hddnet_best.pth

  # Evaluate smoke-test (will be flagged in output)
  python scripts/evaluate.py --hddnet-ckpt models/hddnet/hddnet_epoch3.pth

Output:
  - Console: per-model metrics table
  - debug/evaluation_<tag>.png: side-by-side visual comparison
  - debug/evaluation_<tag>_metrics.txt: machine-readable metrics

The <tag> encodes which HDDNet checkpoint was used so results are
never confused between smoke-test and fully-trained models.

Metrics computed:
  - IoU (Intersection over Union)
  - Dice coefficient
  - Precision
  - Recall
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import RoadDataset
from src.data.augmentations import get_val_transform
from src.models.baseline_unet import get_baseline_model
from src.models.hddnet import HDDNet

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
DEBUG_DIR = os.path.join(PROJECT_ROOT, 'debug')
BASELINE_CKPT_DEFAULT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'best.pth')

# Validation split: use images 300-360 (same indices as training validation)
VAL_START = 300
VAL_SIZE = 60
BATCH_SIZE = 4
NUM_WORKERS = 2
THRESHOLD = 0.5


# ====================================================================
# METRICS
# ====================================================================

def compute_all_metrics(model, loader, device, is_hddnet=False):
    """
    Compute IoU, Dice, Precision, Recall over the entire loader.

    For HDDNet: uses the first return (final_output = max(main, occ)).
    For baseline: uses the single output.

    All metrics are computed per-image, then averaged.
    """
    model.eval()
    all_iou = []
    all_dice = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            if is_hddnet:
                output, _, _ = model(images)
            else:
                output = model(images)

            # Threshold predictions
            pred_prob = torch.sigmoid(output)
            pred_bin = (pred_prob > THRESHOLD).float()

            # Per-image metrics
            for i in range(pred_bin.shape[0]):
                p = pred_bin[i].view(-1)
                t = masks[i].view(-1)

                tp = (p * t).sum().item()
                fp = (p * (1 - t)).sum().item()
                fn = ((1 - p) * t).sum().item()

                # IoU
                iou = tp / (tp + fp + fn + 1e-8)
                all_iou.append(iou)

                # Dice
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                all_dice.append(dice)

                # Precision
                prec = tp / (tp + fp + 1e-8)
                all_precision.append(prec)

                # Recall
                rec = tp / (tp + fn + 1e-8)
                all_recall.append(rec)

    return {
        'IoU': np.mean(all_iou),
        'Dice': np.mean(all_dice),
        'Precision': np.mean(all_precision),
        'Recall': np.mean(all_recall),
        'IoU_std': np.std(all_iou),
        'Dice_std': np.std(all_dice),
        'num_images': len(all_iou),
    }


# ====================================================================
# VISUAL COMPARISON
# ====================================================================

def generate_visual_comparison(baseline_model, hddnet_model, dataset, device,
                               save_path, num_samples=4):
    """
    Generate a visual comparison grid:
    Row per sample: [Satellite | Ground Truth | Baseline Pred | HDDNet Pred]
    If hddnet_model is None, skip the HDDNet column.
    """
    num_cols = 4 if hddnet_model is not None else 3
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Pick evenly-spaced samples
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    col_titles = ['Satellite Image', 'Ground Truth', 'Baseline U-Net']
    if hddnet_model is not None:
        col_titles.append('HDDNet')

    for row, idx in enumerate(indices):
        img_tensor, mask_tensor = dataset[idx]

        # Denormalize image for display (approximate: undo ImageNet normalization)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = np.clip(img_np * std + mean, 0, 1)

        mask_np = mask_tensor.squeeze().numpy()

        # Baseline prediction
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            baseline_pred = torch.sigmoid(baseline_model(x)).cpu().squeeze().numpy()
            baseline_bin = (baseline_pred > THRESHOLD).astype(np.float32)

        # HDDNet prediction
        if hddnet_model is not None:
            with torch.no_grad():
                hddnet_out, _, _ = hddnet_model(x)
                hddnet_pred = torch.sigmoid(hddnet_out).cpu().squeeze().numpy()
                hddnet_bin = (hddnet_pred > THRESHOLD).astype(np.float32)

        # Plot
        axes[row, 0].imshow(img_display)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=13, fontweight='bold')

        axes[row, 1].imshow(mask_np, cmap='gray')
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title(col_titles[1], fontsize=13, fontweight='bold')

        axes[row, 2].imshow(baseline_bin, cmap='gray')
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title(col_titles[2], fontsize=13, fontweight='bold')

        if hddnet_model is not None:
            axes[row, 3].imshow(hddnet_bin, cmap='gray')
            axes[row, 3].axis('off')
            if row == 0:
                axes[row, 3].set_title(col_titles[3], fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visual comparison saved: {save_path}")


# ====================================================================
# CHECKPOINT METADATA
# ====================================================================

def get_checkpoint_tag(ckpt_path):
    """
    Derive a human-readable tag from checkpoint path + metadata.
    Returns (tag, epoch, is_smoke_test) where tag encodes the checkpoint
    identity so output files are never ambiguous.
    """
    if ckpt_path is None:
        return 'baseline_only', -1, False

    basename = os.path.splitext(os.path.basename(ckpt_path))[0]

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', -1)
    except Exception:
        epoch = -1

    # Flag smoke-test: <=5 epochs is definitely not a real training run
    is_smoke = epoch <= 5

    if is_smoke:
        tag = f"{basename}_SMOKETEST_ep{epoch}"
    else:
        tag = f"{basename}_ep{epoch}"

    return tag, epoch, is_smoke


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Section 12: Evaluate models and compare Baseline vs HDDNet')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Evaluate baseline U-Net only (no HDDNet)')
    parser.add_argument('--hddnet-ckpt', type=str, default=None,
                        help='Path to HDDNet checkpoint (.pth). Required unless --baseline-only.')
    parser.add_argument('--val-size', type=int, default=VAL_SIZE,
                        help=f'Number of validation images (default {VAL_SIZE})')
    parser.add_argument('--baseline-ckpt', type=str, default=BASELINE_CKPT_DEFAULT,
                        help=f'Path to baseline checkpoint (default: {BASELINE_CKPT_DEFAULT})')
    parser.add_argument('--num-vis', type=int, default=4,
                        help='Number of samples for visual comparison (default 4)')
    args = parser.parse_args()

    BASELINE_CKPT = args.baseline_ckpt

    # Validate args
    if not args.baseline_only and args.hddnet_ckpt is None:
        parser.error("Either --baseline-only or --hddnet-ckpt is required.")
    if args.hddnet_ckpt and not os.path.exists(args.hddnet_ckpt):
        parser.error(f"HDDNet checkpoint not found: {args.hddnet_ckpt}")
    if not os.path.exists(BASELINE_CKPT):
        parser.error(f"Baseline checkpoint not found: {BASELINE_CKPT}")

    # Get checkpoint tag for output naming
    tag, hddnet_epoch, is_smoke = get_checkpoint_tag(args.hddnet_ckpt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"  Section 12: Evaluation & Baseline Comparison")
    print(f"{'=' * 65}")
    print(f"  Device: {device}")
    print(f"  Baseline ckpt: {BASELINE_CKPT}")
    if args.hddnet_ckpt:
        print(f"  HDDNet ckpt:   {args.hddnet_ckpt}")
        print(f"  HDDNet epoch:  {hddnet_epoch}")
        if is_smoke:
            print(f"  *** WARNING: This is a SMOKE-TEST checkpoint ({hddnet_epoch} epochs). ***")
            print(f"  *** Results are NOT a meaningful baseline-vs-HDDNet comparison. ***")
            print(f"  *** Use a fully-trained HDDNet checkpoint for real evaluation. ***")
    else:
        print(f"  HDDNet: SKIPPED (baseline-only mode)")
    print(f"  Output tag: {tag}")
    print()

    # ================================================================
    # DATA
    # ================================================================
    print("Loading validation data...")
    val_dataset = RoadDataset(DATA_DIR, transform=get_val_transform())
    val_indices = list(range(VAL_START, VAL_START + args.val_size))
    val_subset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(f"  Validation images: {len(val_subset)} (indices {VAL_START}-{VAL_START + args.val_size - 1})")
    print()

    # ================================================================
    # LOAD MODELS
    # ================================================================
    print("Loading baseline U-Net...")
    baseline_model = get_baseline_model().to(device)
    baseline_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    baseline_model.eval()
    print(f"  Loaded (epoch {baseline_ckpt.get('epoch', '?')})")

    hddnet_model = None
    if args.hddnet_ckpt:
        print("Loading HDDNet...")
        hddnet_model = HDDNet().to(device)
        hddnet_ckpt = torch.load(args.hddnet_ckpt, map_location=device, weights_only=False)
        hddnet_model.load_state_dict(hddnet_ckpt['model_state_dict'])
        hddnet_model.eval()
        print(f"  Loaded (epoch {hddnet_ckpt.get('epoch', '?')})")
    print()

    # ================================================================
    # EVALUATE
    # ================================================================
    print("Evaluating baseline U-Net...")
    t0 = time.time()
    baseline_metrics = compute_all_metrics(baseline_model, val_loader, device, is_hddnet=False)
    baseline_time = time.time() - t0

    hddnet_metrics = None
    hddnet_time = 0
    if hddnet_model is not None:
        print("Evaluating HDDNet...")
        t0 = time.time()
        hddnet_metrics = compute_all_metrics(hddnet_model, val_loader, device, is_hddnet=True)
        hddnet_time = time.time() - t0
    print()

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"{'=' * 65}")
    if is_smoke:
        print(f"  RESULTS  [SMOKE-TEST -- NOT a real comparison]")
    else:
        print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<14} | {'Baseline U-Net':>18} ", end='')
    if hddnet_metrics:
        print(f"| {'HDDNet':>18} | {'Delta':>10}")
    else:
        print()
    print(f"  {'-' * 14}-+-{'-' * 18}-", end='')
    if hddnet_metrics:
        print(f"+-{'-' * 18}-+-{'-' * 10}")
    else:
        print()

    metrics_order = ['IoU', 'Dice', 'Precision', 'Recall']
    for m in metrics_order:
        bval = baseline_metrics[m]
        line = f"  {m:<14} | {bval:>18.4f} "
        if hddnet_metrics:
            hval = hddnet_metrics[m]
            delta = hval - bval
            sign = '+' if delta >= 0 else ''
            line += f"| {hval:>18.4f} | {sign}{delta:>9.4f}"
        print(line)

    print()
    print(f"  Evaluated on: {baseline_metrics['num_images']} images")
    print(f"  Baseline eval time: {baseline_time:.1f}s")
    if hddnet_metrics:
        print(f"  HDDNet eval time:   {hddnet_time:.1f}s")
    print()

    if is_smoke:
        print(f"  >>> REMINDER: HDDNet checkpoint is a {hddnet_epoch}-epoch smoke test.")
        print(f"  >>> Run full training (--full flag) before drawing real conclusions.")
        print()

    # ================================================================
    # SAVE METRICS TO FILE
    # ================================================================
    metrics_path = os.path.join(DEBUG_DIR, f'evaluation_{tag}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Tag: {tag}\n")
        f.write(f"Baseline Checkpoint: {BASELINE_CKPT}\n")
        f.write(f"HDDNet Checkpoint: {args.hddnet_ckpt}\n")
        f.write(f"HDDNet Epoch: {hddnet_epoch}\n")
        f.write(f"Is Smoke Test: {is_smoke}\n")
        f.write(f"Num Validation Images: {baseline_metrics['num_images']}\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"\n")
        f.write(f"--- Baseline U-Net ---\n")
        for m in metrics_order:
            f.write(f"{m}: {baseline_metrics[m]:.6f}\n")
        f.write(f"IoU_std: {baseline_metrics['IoU_std']:.6f}\n")
        f.write(f"Dice_std: {baseline_metrics['Dice_std']:.6f}\n")
        if hddnet_metrics:
            f.write(f"\n--- HDDNet ---\n")
            for m in metrics_order:
                f.write(f"{m}: {hddnet_metrics[m]:.6f}\n")
            f.write(f"IoU_std: {hddnet_metrics['IoU_std']:.6f}\n")
            f.write(f"Dice_std: {hddnet_metrics['Dice_std']:.6f}\n")
    print(f"  Metrics saved: {metrics_path}")

    # ================================================================
    # VISUAL COMPARISON
    # ================================================================
    vis_path = os.path.join(DEBUG_DIR, f'evaluation_{tag}.png')
    print(f"  Generating visual comparison ({args.num_vis} samples)...")
    generate_visual_comparison(
        baseline_model, hddnet_model, val_subset, device,
        vis_path, num_samples=args.num_vis
    )

    print(f"\n{'=' * 65}")
    print(f"  Section 12 evaluation complete.")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
