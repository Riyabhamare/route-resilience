"""
test_ensemble.py - Test ensembling baseline U-Net and HDDNet V2.

Compares 5 approaches:
  1. Baseline alone
  2. HDDNet V2 alone
  3. Pixel-wise MAX ensemble
  4. Pixel-wise AVG ensemble
  5. AND-gated MAX (both > 0.3)

Output: numeric results to stdout (pipe to file), PNGs to outputs/ensemble_test/
"""

import os
import sys
import glob

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_unet import get_baseline_model
from src.models.hddnet import HDDNet

# ── Paths ─────────────────────────────────────────────────────────────
BASELINE_CKPT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT   = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
REAL_TEST_DIR  = os.path.join(PROJECT_ROOT, 'real_test_img')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'ensemble_test')

MODEL_SIZE     = 512
LETTERBOX_SIZE = 1024
THRESHOLD      = 0.50

# Images to produce 5-panel visuals for (subset of the 6 "passing" images)
VISUAL_IMAGES = {'506876_sat.jpg', '940563_sat.jpg'}


def letterbox(image, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    lb = cv2.copyMakeBorder(resized, pad_top, target_size - new_h - pad_top,
                            pad_left, target_size - new_w - pad_left,
                            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return lb, scale, pad_top, pad_left, new_h, new_w


def get_prob_maps(baseline_model, hddnet_model, img_rgb, device):
    """
    Run both models on img_rgb. Return probability maps in original image space.
    Returns: (base_prob_orig, hdd_prob_orig, lb_img)
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # Letterbox
    lb_img, scale, pad_top, pad_left, new_h, new_w = letterbox(img_rgb, LETTERBOX_SIZE)

    # Center crop to MODEL_SIZE
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE,
                     crop_start:crop_start + MODEL_SIZE]
    tensor = torch.from_numpy(
        cropped.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        # Baseline: returns (B, 1, H, W) logits
        base_logits = baseline_model(tensor)
        base_prob_crop = torch.sigmoid(base_logits).squeeze().cpu().numpy()

        # HDDNet: returns (final_logits, main_logits, occ_logits)
        final_logits, _, _ = hddnet_model(tensor)
        hdd_prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    # Map each crop back to full letterbox canvas, then to original dims
    def to_original(prob_crop):
        canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
        canvas[crop_start:crop_start + MODEL_SIZE,
               crop_start:crop_start + MODEL_SIZE] = prob_crop
        # Extract the region that maps to the actual image (remove padding)
        img_region = canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        # Resize to original dimensions
        return cv2.resize(img_region, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)

    return to_original(base_prob_crop), to_original(hdd_prob_crop), lb_img


def compute_ensembles(base_prob, hdd_prob):
    """Compute three ensemble variants."""
    max_ens = np.maximum(base_prob, hdd_prob)
    avg_ens = (base_prob + hdd_prob) / 2.0
    and_gated = np.where(
        (base_prob > 0.3) & (hdd_prob > 0.3),
        np.maximum(base_prob, hdd_prob),
        0.0
    ).astype(np.float32)
    return max_ens, avg_ens, and_gated


def region_stats(prob_map):
    """Return dict of stats for a region."""
    return {
        'max':    float(np.max(prob_map)),
        'p99':    float(np.percentile(prob_map, 99)),
        'p95':    float(np.percentile(prob_map, 95)),
        'mean':   float(np.mean(prob_map)),
        'gt050':  int((prob_map > 0.50).sum()),
        'gt035':  int((prob_map > 0.35).sum()),
        'gt020':  int((prob_map > 0.20).sum()),
    }


def print_comparison_table(region_name, stats_dict):
    """
    Print a 5-column comparison table.
    stats_dict: {'Baseline': stats, 'HDDNet': stats, 'Max-Ens': stats, ...}
    """
    names = list(stats_dict.keys())
    metrics = ['max', 'p99', 'p95', 'mean', 'gt050', 'gt035', 'gt020']
    labels = {
        'max': 'Max Probability',
        'p99': '99th Percentile',
        'p95': '95th Percentile',
        'mean': 'Mean Probability',
        'gt050': 'Pixels > 0.50',
        'gt035': 'Pixels > 0.35',
        'gt020': 'Pixels > 0.20',
    }

    print(f"\n  --- {region_name} ---")
    # Header
    header = f"  {'Metric':<20}"
    for n in names:
        header += f" | {n:>12}"
    print(header)
    print("  " + "-" * len(header))

    for m in metrics:
        row = f"  {labels[m]:<20}"
        for n in names:
            v = stats_dict[n][m]
            if m.startswith('gt'):
                row += f" | {v:>12d}"
            else:
                row += f" | {v:>12.6f}"
        print(row)


def overlay_mask(base_rgb, mask_float, color, alpha=0.55):
    """Overlay a binary mask on an RGB image."""
    out = base_rgb.astype(np.float32).copy()
    m = mask_float.astype(np.float32)
    for c_idx, c_val in enumerate(color):
        out[:, :, c_idx] = out[:, :, c_idx] * (1 - alpha * m) + c_val * alpha * m * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def save_5panel(fname_stem, img_rgb, base_prob, hdd_prob,
                max_ens, avg_ens, and_gated):
    """
    5-panel visual: Original | Baseline overlay | HDDNet overlay |
                    Max-Ens overlay | Ensemble heatmaps (3 variants)
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # Row 1: overlays at threshold=0.50
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].axis('off')

    base_mask = (base_prob > THRESHOLD).astype(np.float32)
    axes[0, 1].imshow(overlay_mask(img_rgb, base_mask, (1.0, 0.1, 0.1)))
    bp = float(base_prob.max())
    axes[0, 1].set_title(f'Baseline (maxP={bp:.4f})', fontsize=12)
    axes[0, 1].axis('off')

    hdd_mask = (hdd_prob > THRESHOLD).astype(np.float32)
    axes[0, 2].imshow(overlay_mask(img_rgb, hdd_mask, (0.1, 0.35, 1.0)))
    hp = float(hdd_prob.max())
    axes[0, 2].set_title(f'HDDNet V2 (maxP={hp:.4f})', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: ensemble heatmaps
    im1 = axes[1, 0].imshow(max_ens, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Max-Ensemble (maxP={float(max_ens.max()):.4f})', fontsize=12)
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(avg_ens, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Avg-Ensemble (maxP={float(avg_ens.max()):.4f})', fontsize=12)
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(and_gated, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'AND-Gated (maxP={float(and_gated.max()):.4f})', fontsize=12)
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(f'{fname_stem} — Ensemble Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'{fname_stem}_ensemble.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75)
    print("  ENSEMBLE TEST: Baseline + HDDNet V2")
    print("=" * 75)
    print(f"  Device:   {device}")
    print(f"  Baseline: {BASELINE_CKPT}")
    print(f"  HDDNet:   {HDDNET_CKPT}")

    # ── Load models ───────────────────────────────────────────────────
    print("\n  Loading Baseline U-Net...", flush=True)
    baseline_model = get_baseline_model().to(device)
    b_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(b_ckpt['model_state_dict'])
    baseline_model.eval()
    print(f"  Baseline loaded. Epoch: {b_ckpt.get('epoch', '?')}")

    print("  Loading HDDNet V2...", flush=True)
    hddnet_model = HDDNet().to(device)
    h_ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(h_ckpt['model_state_dict'])
    hddnet_model.eval()
    print(f"  HDDNet V2 loaded. Epoch: {h_ckpt.get('epoch', '?')}, "
          f"Val IoU: {h_ckpt.get('val_iou', '?')}")

    # ── Collect images ────────────────────────────────────────────────
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(REAL_TEST_DIR, ext)))
    all_images = sorted(set(all_images))
    print(f"\n  Found {len(all_images)} images")

    # ══════════════════════════════════════════════════════════════════
    #  PER-IMAGE SUMMARY (all 8)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print("  PER-IMAGE SUMMARY (all 8 images)")
    print(f"{'=' * 75}")

    header = (f"  {'Image':<28} | {'BaseMax':>8} | {'HddMax':>8} | "
              f"{'MaxEns':>8} | {'AvgEns':>8} | {'AndGat':>8} | "
              f"{'B>0.5':>6} | {'H>0.5':>6} | {'MxE>0.5':>7} | "
              f"{'AvE>0.5':>7} | {'AG>0.5':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for img_path in all_images:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  SKIP: {fname}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        base_prob, hdd_prob, lb_img = get_prob_maps(
            baseline_model, hddnet_model, img_rgb, device)
        max_ens, avg_ens, and_gated = compute_ensembles(base_prob, hdd_prob)

        row = (f"  {fname:<28} | "
               f"{float(base_prob.max()):>8.4f} | "
               f"{float(hdd_prob.max()):>8.4f} | "
               f"{float(max_ens.max()):>8.4f} | "
               f"{float(avg_ens.max()):>8.4f} | "
               f"{float(and_gated.max()):>8.4f} | "
               f"{int((base_prob > 0.5).sum()):>6d} | "
               f"{int((hdd_prob > 0.5).sum()):>6d} | "
               f"{int((max_ens > 0.5).sum()):>7d} | "
               f"{int((avg_ens > 0.5).sum()):>7d} | "
               f"{int((and_gated > 0.5).sum()):>7d}")
        print(row)

        # ── cloud_test_3.png: region-level diagnostics ────────────────
        if fname == 'cloud_test_3.png':
            h, w = base_prob.shape[:2]
            regions = {
                'Upper-Left (Dirt Road)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Street Grid)': (slice(h // 2, h), slice(0, w // 2)),
            }
            print(f"\n{'=' * 75}")
            print(f"  CLOUD_TEST_3.PNG — Region-Level Diagnostics")
            print(f"{'=' * 75}")
            for rname, (rslice_h, rslice_w) in regions.items():
                stats = {
                    'Baseline':  region_stats(base_prob[rslice_h, rslice_w]),
                    'HDDNet':    region_stats(hdd_prob[rslice_h, rslice_w]),
                    'Max-Ens':   region_stats(max_ens[rslice_h, rslice_w]),
                    'Avg-Ens':   region_stats(avg_ens[rslice_h, rslice_w]),
                    'AND-Gated': region_stats(and_gated[rslice_h, rslice_w]),
                }
                print_comparison_table(rname, stats)
            print()

        # ── cloud_test_1_coastal.jpg: region-level diagnostics ────────
        if fname == 'cloud_test_1_coastal.jpg':
            h, w = base_prob.shape[:2]
            # Coastal road runs along the left edge; analyze left-half
            # quadrants and also the full left strip
            regions = {
                'Upper-Left (Coast/Town)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Coast/Road)': (slice(h // 2, h), slice(0, w // 2)),
                'Full Left Half':          (slice(0, h), slice(0, w // 2)),
            }
            print(f"\n{'=' * 75}")
            print(f"  CLOUD_TEST_1_COASTAL.JPG — Region-Level Diagnostics")
            print(f"{'=' * 75}")
            for rname, (rslice_h, rslice_w) in regions.items():
                stats = {
                    'Baseline':  region_stats(base_prob[rslice_h, rslice_w]),
                    'HDDNet':    region_stats(hdd_prob[rslice_h, rslice_w]),
                    'Max-Ens':   region_stats(max_ens[rslice_h, rslice_w]),
                    'Avg-Ens':   region_stats(avg_ens[rslice_h, rslice_w]),
                    'AND-Gated': region_stats(and_gated[rslice_h, rslice_w]),
                }
                print_comparison_table(rname, stats)
            print()

        # ── 5-panel visuals for 2 passing images ──────────────────────
        if fname in VISUAL_IMAGES:
            save_5panel(stem, img_rgb, base_prob, hdd_prob,
                        max_ens, avg_ens, and_gated)

    # Also produce ensemble visual for cloud_test_3 and cloud_test_1
    for special_name in ['cloud_test_3.png', 'cloud_test_1_coastal.jpg']:
        sp = os.path.join(REAL_TEST_DIR, special_name)
        if os.path.exists(sp):
            img_bgr = cv2.imread(sp, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            base_prob, hdd_prob, lb_img = get_prob_maps(
                baseline_model, hddnet_model, img_rgb, device)
            max_ens, avg_ens, and_gated = compute_ensembles(base_prob, hdd_prob)
            stem = os.path.splitext(special_name)[0]
            save_5panel(stem, img_rgb, base_prob, hdd_prob,
                        max_ens, avg_ens, and_gated)

    print(f"\n{'=' * 75}")
    print("  ENSEMBLE TEST COMPLETE")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
