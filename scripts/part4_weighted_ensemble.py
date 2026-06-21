"""
Part 4: Weighted Ensemble
Tests 0.5/0.5, 0.6/0.4, 0.7/0.3 (HDDNet/Baseline) weightings.
Compares against existing max and average ensemble results on region-level
diagnostics for cloud_test_3.png and cloud_test_1_coastal.jpg.
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

MODEL_SIZE = 512
LETTERBOX_SIZE = 1024

WEIGHTINGS = [
    (0.5, 0.5, '0.5H+0.5B'),
    (0.6, 0.4, '0.6H+0.4B'),
    (0.7, 0.3, '0.7H+0.3B'),
]


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
    orig_h, orig_w = img_rgb.shape[:2]
    lb_img, scale, pad_top, pad_left, new_h, new_w = letterbox(img_rgb, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        base_logits = baseline_model(tensor)
        base_prob_crop = torch.sigmoid(base_logits).squeeze().cpu().numpy()
        final_logits, _, _ = hddnet_model(tensor)
        hdd_prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    def to_original(prob_crop):
        canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
        canvas[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE] = prob_crop
        img_region = canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        return cv2.resize(img_region, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return to_original(base_prob_crop), to_original(hdd_prob_crop)


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


def print_table(region_name, all_stats):
    """Print multi-column comparison table."""
    names = list(all_stats.keys())
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
    header = f"  {'Metric':<20}"
    for n in names:
        header += f" | {n:>12}"
    print(header)
    print("  " + "-" * len(header))

    for m in metrics:
        row = f"  {labels[m]:<20}"
        for n in names:
            v = all_stats[n][m]
            if m.startswith('gt'):
                row += f" | {v:>12d}"
            else:
                row += f" | {v:>12.6f}"
        print(row)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75)
    print("  PART 4: Weighted Ensemble Comparison")
    print("=" * 75)
    print(f"  Device: {device}")
    print(f"  Weightings tested: {', '.join(w[2] for w in WEIGHTINGS)}")
    print(f"  Also comparing: Baseline, HDDNet, Max-Ens, Avg-Ens")

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

    test_files = ['cloud_test_3.png', 'cloud_test_1_coastal.jpg']

    for fname in test_files:
        img_path = os.path.join(REAL_TEST_DIR, fname)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        print(f"\n{'=' * 75}")
        print(f"  IMAGE: {fname} ({w}x{h})")
        print(f"{'=' * 75}")

        base_prob, hdd_prob = get_prob_maps(baseline_model, hddnet_model, img_rgb, device)

        # Compute all ensemble variants
        max_ens = np.maximum(base_prob, hdd_prob)
        avg_ens = (base_prob + hdd_prob) / 2.0
        weighted = {}
        for wh, wb, label in WEIGHTINGS:
            weighted[label] = wh * hdd_prob + wb * base_prob

        # Define regions
        if fname == 'cloud_test_3.png':
            regions = {
                'Upper-Left (Dirt Road)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Street Grid)': (slice(h // 2, h), slice(0, w // 2)),
            }
        elif fname == 'cloud_test_1_coastal.jpg':
            regions = {
                'Upper-Left (Coast/Town)': (slice(0, h // 2), slice(0, w // 2)),
                'Lower-Left (Coast/Road)': (slice(h // 2, h), slice(0, w // 2)),
                'Full Left Half': (slice(0, h), slice(0, w // 2)),
            }
        else:
            regions = {'Full Image': (slice(0, h), slice(0, w))}

        for rname, (rh, rw) in regions.items():
            all_stats = {
                'Baseline': region_stats(base_prob[rh, rw]),
                'HDDNet': region_stats(hdd_prob[rh, rw]),
                'Max-Ens': region_stats(max_ens[rh, rw]),
                'Avg-Ens': region_stats(avg_ens[rh, rw]),
            }
            for wh, wb, label in WEIGHTINGS:
                all_stats[label] = region_stats(weighted[label][rh, rw])

            print_table(rname, all_stats)

    print(f"\n{'=' * 75}")
    print("  PART 4 COMPLETE")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
