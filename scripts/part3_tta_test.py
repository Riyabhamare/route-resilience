"""
Part 3: Test-Time Augmentation (TTA)
Multi-scale (80%/100%/120%) + horizontal flip, merged via pixel-wise max.
Tests on hard images: cloud_test_3.png, cloud_test_1_coastal.jpg, cloud_test_2_village.jpg
Plus 2 standard images for regression check: 506876_sat.jpg, 940563_sat.jpg
Compares region-level diagnostics: No-TTA vs TTA for Max-Ensemble.
"""
import os, sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model

BASELINE_CKPT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT   = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
REAL_TEST_DIR  = os.path.join(PROJECT_ROOT, 'real_test_img')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'tta_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE = 512
LETTERBOX_SIZE = 1024
THRESHOLD = 0.50

# Images to test
TEST_IMAGES = [
    'cloud_test_3.png',
    'cloud_test_1_coastal.jpg',
    'cloud_test_2_village.jpg',
    '506876_sat.jpg',
    '940563_sat.jpg',
]

# TTA scales
TTA_SCALES = [0.8, 1.0, 1.2]


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


def run_single_model(model, img_rgb, device, tta_scale=1.0, flip=False):
    """
    Run a single model on img_rgb with optional scale and flip.
    Returns probability map in ORIGINAL image space.
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # Apply scale: resize the input image before letterboxing
    if tta_scale != 1.0:
        scaled_h = int(round(orig_h * tta_scale))
        scaled_w = int(round(orig_w * tta_scale))
        img_scaled = cv2.resize(img_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_scaled = img_rgb
        scaled_h, scaled_w = orig_h, orig_w

    # Apply flip
    if flip:
        img_scaled = np.flip(img_scaled, axis=1).copy()

    # Letterbox
    lb_img, lb_scale, pad_top, pad_left, new_h, new_w = letterbox(img_scaled, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, tuple):
            logits = out[0]  # HDDNet: (final, main, occ)
        else:
            logits = out     # Baseline: single output
        prob_crop = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Map back to scaled image space
    canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    canvas[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE] = prob_crop
    pred_region = canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    prob_scaled = cv2.resize(pred_region, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # Unflip
    if flip:
        prob_scaled = np.flip(prob_scaled, axis=1).copy()

    # Resize back to original dimensions
    if tta_scale != 1.0:
        prob_orig = cv2.resize(prob_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        prob_orig = prob_scaled

    return prob_orig


def get_tta_ensemble(baseline_model, hddnet_model, img_rgb, device, scales, use_flip=True):
    """
    Run TTA with multiple scales and optional flip on both models.
    Returns pixel-wise MAX across all augmentations and both models.
    """
    orig_h, orig_w = img_rgb.shape[:2]
    combined = np.zeros((orig_h, orig_w), dtype=np.float32)

    for scale in scales:
        for flip in ([False, True] if use_flip else [False]):
            # Run both models at this scale/flip
            hdd_prob = run_single_model(hddnet_model, img_rgb, device, scale, flip)
            base_prob = run_single_model(baseline_model, img_rgb, device, scale, flip)
            # Max-ensemble of both models
            ens = np.maximum(hdd_prob, base_prob)
            # Max across augmentations
            combined = np.maximum(combined, ens)

    return combined


def get_no_tta_ensemble(baseline_model, hddnet_model, img_rgb, device):
    """Standard Max-Ensemble without TTA (scale=1.0, no flip)."""
    hdd_prob = run_single_model(hddnet_model, img_rgb, device, 1.0, False)
    base_prob = run_single_model(baseline_model, img_rgb, device, 1.0, False)
    return np.maximum(hdd_prob, base_prob)


def region_stats(prob_map):
    return {
        'max': float(np.max(prob_map)),
        'p99': float(np.percentile(prob_map, 99)),
        'p95': float(np.percentile(prob_map, 95)),
        'mean': float(np.mean(prob_map)),
        'gt050': int((prob_map > 0.50).sum()),
        'gt035': int((prob_map > 0.35).sum()),
        'gt020': int((prob_map > 0.20).sum()),
    }


def print_comparison(region_name, no_tta_stats, tta_stats):
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
    print(f"  {'Metric':<20} | {'No-TTA':>12} | {'With-TTA':>12} | {'Change':>12}")
    print("  " + "-" * 65)
    for m in metrics:
        v1 = no_tta_stats[m]
        v2 = tta_stats[m]
        if m.startswith('gt'):
            diff = v2 - v1
            print(f"  {labels[m]:<20} | {v1:>12d} | {v2:>12d} | {diff:>+12d}")
        else:
            diff = v2 - v1
            print(f"  {labels[m]:<20} | {v1:>12.6f} | {v2:>12.6f} | {diff:>+12.6f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75)
    print("  PART 3: Test-Time Augmentation (TTA)")
    print("=" * 75)
    print(f"  Device: {device}")
    print(f"  TTA scales: {TTA_SCALES}")
    print(f"  Horizontal flip: Yes")
    print(f"  Merge strategy: pixel-wise MAX across all augmentations")
    print(f"  Total inference passes per image: {len(TTA_SCALES) * 2} (3 scales x 2 flip states)")
    print(f"  x 2 models = {len(TTA_SCALES) * 2 * 2} forward passes per image")

    # Load models
    print("\n  Loading Baseline U-Net...", flush=True)
    baseline_model = get_baseline_model().to(device)
    b_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(b_ckpt['model_state_dict'])
    baseline_model.eval()

    print("  Loading HDDNet V2...", flush=True)
    hddnet_model = HDDNet().to(device)
    h_ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(h_ckpt['model_state_dict'])
    hddnet_model.eval()

    for fname in TEST_IMAGES:
        img_path = os.path.join(REAL_TEST_DIR, fname)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        print(f"\n{'=' * 75}")
        print(f"  IMAGE: {fname} ({w}x{h})")
        print(f"{'=' * 75}")

        # No-TTA Max-Ensemble
        print("  Running No-TTA Max-Ensemble...", flush=True)
        no_tta = get_no_tta_ensemble(baseline_model, hddnet_model, img_rgb, device)

        # TTA Max-Ensemble
        print("  Running TTA Max-Ensemble (3 scales x 2 flips x 2 models)...", flush=True)
        tta = get_tta_ensemble(baseline_model, hddnet_model, img_rgb, device, TTA_SCALES)

        # Full image stats
        print_comparison(
            f"{fname} -- Full Image",
            region_stats(no_tta), region_stats(tta))

        # Region-level diagnostics for specific images
        if fname == 'cloud_test_3.png':
            regions = {
                'Upper-Left (Dirt Road)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Street Grid)': (slice(h // 2, h), slice(0, w // 2)),
            }
            for rname, (rh, rw) in regions.items():
                print_comparison(rname,
                    region_stats(no_tta[rh, rw]),
                    region_stats(tta[rh, rw]))

        elif fname == 'cloud_test_1_coastal.jpg':
            regions = {
                'Upper-Left (Coast/Town)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Coast/Road)': (slice(h // 2, h), slice(0, w // 2)),
                'Full Left Half': (slice(0, h), slice(0, w // 2)),
            }
            for rname, (rh, rw) in regions.items():
                print_comparison(rname,
                    region_stats(no_tta[rh, rw]),
                    region_stats(tta[rh, rw]))

    print(f"\n{'=' * 75}")
    print("  PART 3 COMPLETE")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
