"""
Part 2: Failure Categorization across all 8 real test images.
For each image, analyze specific regions and categorize failures into:
  Cloud-occlusion, Low-contrast/dirt-road, Coastal/water,
  Fragmented-road, Shadow, Other
Uses Max-Ensemble (Baseline + HDDNet V2, pixel-wise max) as the detection source.
"""
import os, sys, glob
import numpy as np
import cv2
import torch
from scipy import ndimage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model

BASELINE_CKPT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT   = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
REAL_TEST_DIR  = os.path.join(PROJECT_ROOT, 'real_test_img')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'failure_categorization')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE = 512
LETTERBOX_SIZE = 1024
THRESHOLD = 0.50


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

    return to_original(base_prob_crop), to_original(hdd_prob_crop), lb_img


def count_connected_components(binary_mask):
    """Count connected components in a binary mask (excluding background)."""
    labeled, num_features = ndimage.label(binary_mask)
    return num_features


def fragmentation_analysis(prob_map, threshold=0.5):
    """Analyze fragmentation: count components and largest component ratio."""
    binary = (prob_map > threshold).astype(np.uint8)
    total_road_px = int(binary.sum())
    if total_road_px == 0:
        return 0, 0, 0.0
    labeled, num_components = ndimage.label(binary)
    component_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
    largest = int(max(component_sizes)) if len(component_sizes) > 0 else 0
    largest_ratio = largest / total_road_px if total_road_px > 0 else 0.0
    return num_components, largest, largest_ratio


def analyze_image(fname, img_rgb, base_prob, hdd_prob, max_ens):
    """Analyze one image for all failure categories. Returns list of findings."""
    h, w = img_rgb.shape[:2]
    findings = []

    # ── Fragmentation analysis (all images) ───────────────────────────
    n_comp, largest, ratio = fragmentation_analysis(max_ens, THRESHOLD)
    total_road_px = int((max_ens > THRESHOLD).sum())
    findings.append({
        'image': fname,
        'category': 'Fragmented-road',
        'region': 'Full image',
        'detail': (f"Components={n_comp}, largest={largest}px "
                   f"({ratio*100:.1f}% of {total_road_px} road px)"),
        'severity': 'HIGH' if (n_comp > 20 and ratio < 0.5) else
                    'MODERATE' if n_comp > 10 else 'LOW',
    })

    # ── Cloud detection (bright pixel analysis) ───────────────────────
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Clouds are typically very bright (>200) and low-saturation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    bright_mask = (gray > 200) & (hsv[:, :, 1] < 50)  # bright + low saturation
    cloud_fraction = float(bright_mask.mean())
    if cloud_fraction > 0.05:  # more than 5% cloud coverage
        # Check if cloud regions have road gaps
        cloud_road_prob = max_ens[bright_mask].mean() if bright_mask.any() else 0.0
        findings.append({
            'image': fname,
            'category': 'Cloud-occlusion',
            'region': f"Cloud pixels ({cloud_fraction*100:.1f}% of image)",
            'detail': (f"Mean road prob under clouds: {cloud_road_prob:.6f}, "
                       f"cloud px: {int(bright_mask.sum())}"),
            'severity': 'HIGH' if cloud_fraction > 0.20 else 'MODERATE',
        })

    # ── Coastal/water detection ───────────────────────────────────────
    # Water is typically dark blue — check for dark blue pixels
    b, g, r = img_rgb[:, :, 2], img_rgb[:, :, 1], img_rgb[:, :, 0]
    water_mask = (b.astype(int) > g.astype(int) + 10) & (gray < 120)
    water_fraction = float(water_mask.mean())
    if water_fraction > 0.05:
        water_road_prob = max_ens[water_mask].mean() if water_mask.any() else 0.0
        findings.append({
            'image': fname,
            'category': 'Coastal/water',
            'region': f"Water pixels ({water_fraction*100:.1f}% of image)",
            'detail': (f"Mean road prob over water: {water_road_prob:.6f}, "
                       f"maxP over water: {float(max_ens[water_mask].max()) if water_mask.any() else 0:.6f}"),
            'severity': 'HIGH' if water_fraction > 0.20 else 'MODERATE',
        })

    # ── Low-contrast / dirt road detection ────────────────────────────
    # Check local contrast: regions where image is fairly uniform but
    # there should be roads
    local_std = cv2.blur((gray.astype(np.float32) - cv2.blur(gray.astype(np.float32),
                          (31, 31))) ** 2, (31, 31)) ** 0.5
    low_contrast_mask = (local_std < 15) & (~bright_mask) & (~water_mask)
    lc_fraction = float(low_contrast_mask.mean())
    if lc_fraction > 0.10:
        lc_road_prob = max_ens[low_contrast_mask].mean() if low_contrast_mask.any() else 0.0
        lc_px_detected = int((max_ens[low_contrast_mask] > THRESHOLD).sum())
        findings.append({
            'image': fname,
            'category': 'Low-contrast/dirt-road',
            'region': f"Low-contrast pixels ({lc_fraction*100:.1f}% of image)",
            'detail': (f"Mean road prob: {lc_road_prob:.6f}, "
                       f"road px detected in low-contrast area: {lc_px_detected}"),
            'severity': 'MODERATE' if lc_px_detected < 100 else 'LOW',
        })

    # ── Shadow detection ──────────────────────────────────────────────
    shadow_mask = (gray < 60) & (~water_mask)
    shadow_fraction = float(shadow_mask.mean())
    if shadow_fraction > 0.03:
        shadow_road_prob = max_ens[shadow_mask].mean() if shadow_mask.any() else 0.0
        findings.append({
            'image': fname,
            'category': 'Shadow',
            'region': f"Shadow pixels ({shadow_fraction*100:.1f}% of image)",
            'detail': (f"Mean road prob in shadow: {shadow_road_prob:.6f}"),
            'severity': 'MODERATE' if shadow_road_prob < 0.1 else 'LOW',
        })

    return findings


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 75)
    print("  PART 2: Failure Categorization (all 8 real test images)")
    print("=" * 75)
    print(f"  Device: {device}")

    # Load models
    print("  Loading Baseline U-Net...", flush=True)
    baseline_model = get_baseline_model().to(device)
    b_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(b_ckpt['model_state_dict'])
    baseline_model.eval()

    print("  Loading HDDNet V2...", flush=True)
    hddnet_model = HDDNet().to(device)
    h_ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(h_ckpt['model_state_dict'])
    hddnet_model.eval()

    # Collect images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(REAL_TEST_DIR, ext)))
    all_images = sorted(set(all_images))
    print(f"  Found {len(all_images)} images\n")

    all_findings = []

    for img_path in all_images:
        fname = os.path.basename(img_path)
        print(f"  Processing: {fname}", flush=True)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        base_prob, hdd_prob, lb_img = get_prob_maps(baseline_model, hddnet_model, img_rgb, device)
        max_ens = np.maximum(base_prob, hdd_prob)

        findings = analyze_image(fname, img_rgb, base_prob, hdd_prob, max_ens)
        all_findings.extend(findings)

    # ══════════════════════════════════════════════════════════════════
    #  PER-IMAGE FINDINGS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print("  PER-IMAGE FINDINGS")
    print(f"{'=' * 75}")
    current_image = None
    for f in all_findings:
        if f['image'] != current_image:
            current_image = f['image']
            print(f"\n  --- {current_image} ---")
        print(f"    [{f['severity']:>8}] {f['category']:<25} | {f['region']}")
        print(f"             {f['detail']}")

    # ══════════════════════════════════════════════════════════════════
    #  CATEGORY COUNT TABLE
    # ══════════════════════════════════════════════════════════════════
    categories = ['Cloud-occlusion', 'Low-contrast/dirt-road', 'Coastal/water',
                  'Fragmented-road', 'Shadow', 'Other']
    print(f"\n{'=' * 75}")
    print("  FAILURE CATEGORY COUNT TABLE")
    print(f"{'=' * 75}")
    print(f"  {'Category':<25} | {'Count':>5} | {'HIGH':>4} | {'MODERATE':>8} | {'LOW':>4} | Images affected")
    print("  " + "-" * 85)
    for cat in categories:
        cat_findings = [f for f in all_findings if f['category'] == cat]
        n = len(cat_findings)
        high = sum(1 for f in cat_findings if f['severity'] == 'HIGH')
        moderate = sum(1 for f in cat_findings if f['severity'] == 'MODERATE')
        low = sum(1 for f in cat_findings if f['severity'] == 'LOW')
        images = sorted(set(f['image'] for f in cat_findings))
        img_str = ', '.join(images) if images else '-'
        print(f"  {cat:<25} | {n:>5} | {high:>4} | {moderate:>8} | {low:>4} | {img_str}")

    total = len(all_findings)
    print(f"\n  Total findings: {total} across {len(all_images)} images")

    # ── Sample size caveat ────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  SAMPLE SIZE NOTE")
    print(f"{'=' * 75}")
    print(f"  This analysis covers {len(all_images)} images only.")
    print(f"  With 8 images, individual category counts are NOT statistically")
    print(f"  robust enough to rank failure modes by population frequency.")
    print(f"  The counts indicate which categories are PRESENT in this test set,")
    print(f"  not their true prevalence in the full deployment distribution.")

    print(f"\n{'=' * 75}")
    print("  PART 2 COMPLETE")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
