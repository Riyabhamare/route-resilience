"""
test_real_images.py -- Batch inference on all images in real_test_img/.

For each image:
  1. Load + letterbox to 1024x1024 (black padding, aspect-ratio preserved)
  2. Run Baseline U-Net and HDDNet
  3. Produce a 5-panel comparison figure:
       Panel 1: Original image
       Panel 2: Baseline prediction overlay (red, thresholded)
       Panel 3: HDDNet final prediction overlay (blue, thresholded)
       Panel 4: HDDNet decoder_main raw probability heatmap (viridis)
       Panel 5: HDDNet decoder_occlusion raw probability heatmap (hot)
  4. Save to outputs/real_test/<filename>_comparison.png
  5. Print honest decoder_occlusion activation summary for each image.

No IoU/Dice metrics -- these images have NO ground truth.
"""

import os
import sys
import glob

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_unet import get_baseline_model
from src.models.hddnet import HDDNet

# ============================================================
REAL_TEST_DIR  = os.path.join(PROJECT_ROOT, 'real_test_img')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'real_test')
BASELINE_CKPT  = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT    = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')

# Training input size (the models expect 512x512 crops)
MODEL_SIZE     = 512
# We letterbox incoming images to this before tiling/resizing
LETTERBOX_SIZE = 1024
PAD_THRESHOLD  = 0.20    # warn if either dimension is padded more than 20%
THRESHOLD      = 0.50    # probability threshold for mask binarisation
# ============================================================

OBLIQUE_FILENAMES = {'78954.jpg'}  # oblique aerial -- distribution-shift flag


# ------------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------------

def letterbox(image: np.ndarray, target_size: int = 1024) -> tuple[np.ndarray, float, int, int]:
    """
    Resize image to fit inside (target_size x target_size) while preserving
    aspect ratio, then pad the shorter dimension with black to reach exactly
    (target_size x target_size).

    Returns:
        lb_img  : letterboxed RGB image (target_size x target_size x 3)
        scale   : scale factor applied to original
        pad_top : pixels of top padding added
        pad_left: pixels of left padding added
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top  = (target_size - new_h) // 2
    pad_bot  = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    lb_img = cv2.copyMakeBorder(
        resized, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return lb_img, scale, pad_top, pad_left


def preprocess_for_model(lb_img: np.ndarray) -> tuple[torch.Tensor, int]:
    """
    Take a letterboxed LETTERBOX_SIZE image, centre-crop MODEL_SIZE x MODEL_SIZE,
    normalize to [0,1] and return a batch tensor.  The crop offset is also
    returned so we can put the prediction back in the right position.
    """
    h, w = lb_img.shape[:2]
    assert h == LETTERBOX_SIZE and w == LETTERBOX_SIZE, f"Expected {LETTERBOX_SIZE}x{LETTERBOX_SIZE}, got {h}x{w}"

    # For 1024 -> 512: take the centre 512x512 crop
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE]

    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor, crop_start


def build_full_pred(raw_prob: np.ndarray, crop_start: int) -> np.ndarray:
    """
    Place a MODEL_SIZE x MODEL_SIZE prediction back into a
    LETTERBOX_SIZE x LETTERBOX_SIZE canvas (zeros elsewhere).
    """
    canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    canvas[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE] = raw_prob
    return canvas


# ------------------------------------------------------------------
# Overlay helpers
# ------------------------------------------------------------------

def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray,
                 color: tuple, alpha: float = 0.55) -> np.ndarray:
    """Blend a binary/float mask of shape (H,W) over a (H,W,3) RGB image."""
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(np.float32)
    for c_idx, c_val in enumerate(color):
        out[:, :, c_idx] = out[:, :, c_idx] * (1 - alpha * m) + c_val * alpha * m * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Activation analysis
# ------------------------------------------------------------------

def analyse_branch(prob_map: np.ndarray, branch_name: str) -> dict:
    """
    Compute statistics on a decoder branch's probability map.
    Works for both decoder_main and decoder_occlusion.
    """
    cs = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    active = prob_map[cs:cs + MODEL_SIZE, cs:cs + MODEL_SIZE]

    mean_prob  = float(active.mean())
    max_prob   = float(active.max())
    std_prob   = float(active.std())
    frac_above = float((active > 0.20).mean())
    road_pixels = float((active > THRESHOLD).mean())

    contribution = "ACTIVE"
    notes = []
    if max_prob < 0.35:
        contribution = "COLLAPSED"
        notes.append(f"{branch_name}: max prob < 0.35 -- branch never confident")
    elif frac_above < 0.01:
        contribution = "SPARSE"
        notes.append(f"{branch_name}: < 1% of pixels with p > 0.20")
    elif road_pixels < 0.002:
        contribution = "NEAR-BLANK"
        notes.append(f"{branch_name}: < 0.2% of pixels thresholded as road")

    return {
        'branch_name'  : branch_name,
        'mean_prob'    : mean_prob,
        'max_prob'     : max_prob,
        'std_prob'     : std_prob,
        'frac_above_02': frac_above,
        'road_frac'    : road_pixels,
        'contribution' : contribution,
        'notes'        : notes,
    }


# ------------------------------------------------------------------
# Per-image pipeline
# ------------------------------------------------------------------

def run_image(path: str, baseline_model, hddnet_model, device: str) -> dict:
    fname = os.path.basename(path)
    stem  = os.path.splitext(fname)[0]
    is_oblique = fname in OBLIQUE_FILENAMES

    print(f"\n{'='*70}", flush=True)
    print(f"  Image: {fname}", flush=True)
    if is_oblique:
        print(f"  *** DISTRIBUTION-SHIFT TEST: oblique/foggy aerial photo ***", flush=True)

    # 1. Load
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  ERROR: could not read {path}", flush=True)
        return {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f"  Original size: {orig_w}x{orig_h}", flush=True)

    # 2. Letterbox
    lb_img, scale, pad_top, pad_left = letterbox(img_rgb, LETTERBOX_SIZE)
    scaled_h = int(round(orig_h * scale))
    scaled_w = int(round(orig_w * scale))
    pad_h_frac = (LETTERBOX_SIZE - scaled_h) / LETTERBOX_SIZE
    pad_w_frac = (LETTERBOX_SIZE - scaled_w) / LETTERBOX_SIZE

    if pad_h_frac > PAD_THRESHOLD or pad_w_frac > PAD_THRESHOLD:
        print(f"  WARNING: significant padding — "
              f"h_pad={pad_h_frac*100:.1f}%, w_pad={pad_w_frac*100:.1f}%. "
              f"Aspect ratio differs substantially from training distribution.", flush=True)

    # 3. Preprocess for model (centre crop to MODEL_SIZE)
    tensor, crop_start = preprocess_for_model(lb_img)
    tensor = tensor.to(device)
    print(f"  Letterboxed -> {LETTERBOX_SIZE}x{LETTERBOX_SIZE}, "
          f"model crop offset = {crop_start}px", flush=True)

    # 4. Inference
    print("  Running inference...", flush=True)
    with torch.no_grad():
        # Baseline
        base_logits = baseline_model(tensor)
        base_prob_crop = torch.sigmoid(base_logits).squeeze().cpu().numpy()

        # HDDNet
        final_logits, main_logits, occ_logits = hddnet_model(tensor)
        hdd_prob_crop  = torch.sigmoid(final_logits).squeeze().cpu().numpy()
        main_prob_crop = torch.sigmoid(main_logits).squeeze().cpu().numpy()
        occ_prob_crop  = torch.sigmoid(occ_logits).squeeze().cpu().numpy()

    # Place predictions back into full letterbox canvas
    base_prob = build_full_pred(base_prob_crop, crop_start)
    hdd_prob  = build_full_pred(hdd_prob_crop,  crop_start)
    main_prob = build_full_pred(main_prob_crop, crop_start)
    occ_prob  = build_full_pred(occ_prob_crop,  crop_start)

    # Threshold
    base_mask = (base_prob > THRESHOLD).astype(np.float32)
    hdd_mask  = (hdd_prob  > THRESHOLD).astype(np.float32)

    # 5. Per-branch analysis (both main AND occlusion)
    main_stats = analyse_branch(main_prob, 'main')
    occ_stats  = analyse_branch(occ_prob, 'occ')
    print(f"  decoder_main:      maxP={main_stats['max_prob']:.4f}  "
          f"meanP={main_stats['mean_prob']:.4f}  "
          f"std={main_stats['std_prob']:.4f}  "
          f"road%={main_stats['road_frac']*100:.2f}%  [{main_stats['contribution']}]", flush=True)
    print(f"  decoder_occlusion: maxP={occ_stats['max_prob']:.4f}  "
          f"meanP={occ_stats['mean_prob']:.4f}  "
          f"std={occ_stats['std_prob']:.4f}  "
          f"road%={occ_stats['road_frac']*100:.2f}%  [{occ_stats['contribution']}]", flush=True)
    # Mean ratio: should be near 1.0 if branches are balanced
    if main_stats['mean_prob'] > 1e-6:
        ratio = occ_stats['mean_prob'] / main_stats['mean_prob']
        print(f"  Occ/Main mean ratio: {ratio:.2f} (1.0 = perfectly balanced)", flush=True)
    for note in main_stats['notes'] + occ_stats['notes']:
        print(f"    -> {note}", flush=True)

    # 6. 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    dist_note = " [DISTRIBUTION SHIFT: oblique/foggy - not satellite]" if is_oblique else ""
    fig.suptitle(
        f"No ground truth - visual inspection only{dist_note}\n{fname}  "
        f"(original {orig_w}x{orig_h}, model input centre-{MODEL_SIZE}px crop of {LETTERBOX_SIZE}px letterbox)",
        fontsize=9, y=1.01, style='italic', color='gray'
    )

    # Panel 1: original letterboxed image
    axes[0].imshow(lb_img)
    axes[0].set_title('Original\n(letterboxed)', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Baseline overlay (red)
    base_overlay = overlay_mask(lb_img, base_mask, color=(1.0, 0.1, 0.1))
    axes[1].imshow(base_overlay)
    axes[1].set_title('Baseline U-Net\n(thresholded, red)', fontsize=12)
    axes[1].axis('off')

    # Panel 3: HDDNet final overlay (blue)
    hdd_overlay = overlay_mask(lb_img, hdd_mask, color=(0.1, 0.35, 1.0))
    axes[2].imshow(hdd_overlay)
    axes[2].set_title('HDDNet - Final\n(thresholded, blue)', fontsize=12)
    axes[2].axis('off')

    # Panel 4: decoder_main raw heatmap
    im4 = axes[3].imshow(main_prob, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[3].set_title('HDDNet decoder_main\n(raw prob, viridis)', fontsize=12)
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: decoder_occlusion raw heatmap
    im5 = axes[4].imshow(occ_prob, cmap='hot', vmin=0.0, vmax=1.0)
    contrib_label = occ_stats['contribution']
    axes[4].set_title(f'HDDNet decoder_occlusion\n(raw prob, hot) [{contrib_label}]', fontsize=12)
    axes[4].axis('off')
    fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

    return {
        'fname'     : fname,
        'is_oblique': is_oblique,
        'main_stats': main_stats,
        'occ_stats' : occ_stats,
        'out_path'  : out_path,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Collect images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(REAL_TEST_DIR, ext)))
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"No images found in {REAL_TEST_DIR}", flush=True)
        sys.exit(1)

    print(f"Found {len(all_images)} image(s) in {REAL_TEST_DIR}", flush=True)

    # Load models once
    print("\nLoading Baseline U-Net...", flush=True)
    baseline_model = get_baseline_model().to(device)
    ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_model.eval()

    print("Loading HDDNet...", flush=True)
    hddnet_model = HDDNet().to(device)
    ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(ckpt['model_state_dict'])
    hddnet_model.eval()

    # Process each image
    results = []
    for img_path in all_images:
        r = run_image(img_path, baseline_model, hddnet_model, device)
        if r:
            results.append(r)

    # -------------------------------------------------------
    # Final summary: BOTH branches
    # -------------------------------------------------------
    print(f"\n{'='*90}", flush=True)
    print("  DUAL-DECODER BRANCH HEALTH -- REAL IMAGE SUMMARY", flush=True)
    print(f"  Compare against v1 (collapsed): Main max ~0.13, Main mean ~0.003-0.011", flush=True)
    print(f"  Compare against v2 training:    Main mean ~0.045, Occ mean ~0.045 (balanced)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  {'Image':<25} | {'MnMaxP':>7} | {'MnMeanP':>8} | {'OcMaxP':>7} | {'OcMeanP':>8} | {'Ratio':>5} | {'MainSt':<8} | {'OccSt':<8}", flush=True)
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}", flush=True)

    main_active = 0
    occ_active = 0
    for r in results:
        ms = r['main_stats']
        os_ = r['occ_stats']
        flag = " *" if r['is_oblique'] else ""
        ratio = os_['mean_prob'] / ms['mean_prob'] if ms['mean_prob'] > 1e-6 else float('inf')
        print(
            f"  {r['fname']:<25} | {ms['max_prob']:>7.4f} | {ms['mean_prob']:>8.4f} | "
            f"{os_['max_prob']:>7.4f} | {os_['mean_prob']:>8.4f} | {ratio:>5.2f} | "
            f"{ms['contribution']:<8} | {os_['contribution']:<8}{flag}",
            flush=True
        )
        if ms['contribution'] == 'ACTIVE':
            main_active += 1
        if os_['contribution'] == 'ACTIVE':
            occ_active += 1

    print(f"\n  Main ACTIVE: {main_active}/{len(results)}  |  Occ ACTIVE: {occ_active}/{len(results)}", flush=True)
    if main_active == len(results) and occ_active == len(results):
        print("  -> Both decoders ACTIVE on all images. Branch collapse is cured.", flush=True)
    elif main_active == 0:
        print("  -> decoder_main COLLAPSED on all images. Same failure as v1.", flush=True)
    else:
        print(f"  -> Mixed results. Investigate images where branches are not both ACTIVE.", flush=True)
    print(f"{'='*90}", flush=True)


if __name__ == '__main__':
    main()
