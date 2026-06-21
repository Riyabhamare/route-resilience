"""
verify_real_images.py -- Clean verification run for HDDNet v2 branch health.

Outputs everything to outputs/real_test_v2_verification/:
  - 7 individual 5-panel comparison PNGs
  - SUMMARY.txt with per-image stats and PASS/FAIL verdicts

PASS criteria:
  - Both branches ACTIVE (maxP >= 0.35)
  - Balance ratio (Occ meanP / Main meanP) between 0.8 and 1.2

FAIL criteria:
  - Either branch COLLAPSED (maxP < 0.35)
  - OR ratio outside 0.2-5.0 range (one branch dominating)
"""

import os
import sys
import glob
import datetime

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
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'real_test_v2_verification')
BASELINE_CKPT  = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT    = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')

MODEL_SIZE     = 512
LETTERBOX_SIZE = 1024
THRESHOLD      = 0.50
# ============================================================

OBLIQUE_FILENAMES = {'78954.jpg'}


def letterbox(image, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_bot = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    lb_img = cv2.copyMakeBorder(
        resized, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return lb_img, scale, pad_top, pad_left


def preprocess_for_model(lb_img):
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor, crop_start


def build_full_pred(raw_prob, crop_start):
    canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    canvas[crop_start:crop_start + MODEL_SIZE, crop_start:crop_start + MODEL_SIZE] = raw_prob
    return canvas


def analyse_branch(prob_map, branch_name):
    cs = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    active = prob_map[cs:cs + MODEL_SIZE, cs:cs + MODEL_SIZE]
    mean_prob = float(active.mean())
    max_prob = float(active.max())
    std_prob = float(active.std())
    road_frac = float((active > THRESHOLD).mean())

    if max_prob < 0.35:
        status = "COLLAPSED"
    elif road_frac < 0.002 and max_prob < 0.5:
        status = "NEAR-BLANK"
    else:
        status = "ACTIVE"

    return {
        'branch': branch_name,
        'max_prob': max_prob,
        'mean_prob': mean_prob,
        'std_prob': std_prob,
        'road_frac': road_frac,
        'status': status,
    }


def overlay_mask(base_rgb, mask, color, alpha=0.55):
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(np.float32)
    for c_idx, c_val in enumerate(color):
        out[:, :, c_idx] = out[:, :, c_idx] * (1 - alpha * m) + c_val * alpha * m * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def judge_verdict(main_stats, occ_stats):
    """Return (verdict, reason) tuple."""
    if main_stats['status'] == 'COLLAPSED' and occ_stats['status'] == 'COLLAPSED':
        return 'FAIL', 'Both branches COLLAPSED (maxP < 0.35)'
    if main_stats['status'] == 'COLLAPSED':
        return 'FAIL', f"decoder_main COLLAPSED (maxP={main_stats['max_prob']:.4f})"
    if occ_stats['status'] == 'COLLAPSED':
        return 'FAIL', f"decoder_occlusion COLLAPSED (maxP={occ_stats['max_prob']:.4f})"

    # Check balance ratio
    if main_stats['mean_prob'] > 1e-6:
        ratio = occ_stats['mean_prob'] / main_stats['mean_prob']
    else:
        return 'FAIL', 'decoder_main mean probability is near-zero'

    if ratio < 0.2 or ratio > 5.0:
        return 'FAIL', f"Extreme imbalance: Occ/Main ratio = {ratio:.2f} (outside 0.2-5.0)"
    if ratio < 0.8 or ratio > 1.2:
        return 'PASS*', f"Minor imbalance: Occ/Main ratio = {ratio:.2f} (outside 0.8-1.2 but within 0.2-5.0)"

    return 'PASS', f"Both ACTIVE, balanced (ratio={ratio:.2f})"


def run_one_image(path, baseline_model, hddnet_model, device):
    fname = os.path.basename(path)
    stem = os.path.splitext(fname)[0]
    is_oblique = fname in OBLIQUE_FILENAMES

    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    lb_img, scale, pad_top, pad_left = letterbox(img_rgb, LETTERBOX_SIZE)
    tensor, crop_start = preprocess_for_model(lb_img)
    tensor = tensor.to(device)

    with torch.no_grad():
        base_logits = baseline_model(tensor)
        base_prob_crop = torch.sigmoid(base_logits).squeeze().cpu().numpy()

        final_logits, main_logits, occ_logits = hddnet_model(tensor)
        hdd_prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()
        main_prob_crop = torch.sigmoid(main_logits).squeeze().cpu().numpy()
        occ_prob_crop = torch.sigmoid(occ_logits).squeeze().cpu().numpy()

    base_prob = build_full_pred(base_prob_crop, crop_start)
    hdd_prob = build_full_pred(hdd_prob_crop, crop_start)
    main_prob = build_full_pred(main_prob_crop, crop_start)
    occ_prob = build_full_pred(occ_prob_crop, crop_start)

    base_mask = (base_prob > THRESHOLD).astype(np.float32)
    hdd_mask = (hdd_prob > THRESHOLD).astype(np.float32)

    main_stats = analyse_branch(main_prob, 'main')
    occ_stats = analyse_branch(occ_prob, 'occ')

    verdict, reason = judge_verdict(main_stats, occ_stats)

    if main_stats['mean_prob'] > 1e-6:
        ratio = occ_stats['mean_prob'] / main_stats['mean_prob']
    else:
        ratio = float('inf')

    # Generate 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    dist_note = " [OBLIQUE/DIST-SHIFT]" if is_oblique else ""
    fig.suptitle(
        f"{fname} ({orig_w}x{orig_h}){dist_note}  |  Verdict: {verdict}",
        fontsize=11, y=1.01, fontweight='bold'
    )

    axes[0].imshow(lb_img)
    axes[0].set_title('Original\n(letterboxed)', fontsize=12)
    axes[0].axis('off')

    base_overlay = overlay_mask(lb_img, base_mask, color=(1.0, 0.1, 0.1))
    axes[1].imshow(base_overlay)
    axes[1].set_title('Baseline U-Net\n(thresholded, red)', fontsize=12)
    axes[1].axis('off')

    hdd_overlay = overlay_mask(lb_img, hdd_mask, color=(0.1, 0.35, 1.0))
    axes[2].imshow(hdd_overlay)
    axes[2].set_title('HDDNet - Final\n(thresholded, blue)', fontsize=12)
    axes[2].axis('off')

    im4 = axes[3].imshow(main_prob, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[3].set_title(f'decoder_main\nmaxP={main_stats["max_prob"]:.4f} meanP={main_stats["mean_prob"]:.4f}', fontsize=11)
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    im5 = axes[4].imshow(occ_prob, cmap='hot', vmin=0.0, vmax=1.0)
    axes[4].set_title(f'decoder_occlusion\nmaxP={occ_stats["max_prob"]:.4f} meanP={occ_stats["mean_prob"]:.4f}', fontsize=11)
    axes[4].axis('off')
    fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return {
        'fname': fname,
        'orig_size': f"{orig_w}x{orig_h}",
        'is_oblique': is_oblique,
        'main_stats': main_stats,
        'occ_stats': occ_stats,
        'ratio': ratio,
        'verdict': verdict,
        'reason': reason,
        'out_path': out_path,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"Baseline: {BASELINE_CKPT}", flush=True)
    print(f"HDDNet:   {HDDNET_CKPT}", flush=True)

    # Collect images
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(REAL_TEST_DIR, ext)))
    all_images = sorted(set(all_images))

    print(f"Found {len(all_images)} images", flush=True)

    # Load models
    print("Loading Baseline U-Net...", flush=True)
    baseline_model = get_baseline_model().to(device)
    ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_model.eval()
    baseline_epoch = ckpt.get('epoch', '?')

    print("Loading HDDNet...", flush=True)
    hddnet_model = HDDNet().to(device)
    ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(ckpt['model_state_dict'])
    hddnet_model.eval()
    hddnet_epoch = ckpt.get('epoch', '?')

    print(f"Baseline epoch: {baseline_epoch}, HDDNet epoch: {hddnet_epoch}", flush=True)

    # Process each image
    results = []
    for img_path in all_images:
        fname = os.path.basename(img_path)
        print(f"\nProcessing: {fname}...", flush=True)
        r = run_one_image(img_path, baseline_model, hddnet_model, device)
        if r:
            print(f"  Main:  maxP={r['main_stats']['max_prob']:.4f}  meanP={r['main_stats']['mean_prob']:.4f}  [{r['main_stats']['status']}]", flush=True)
            print(f"  Occ:   maxP={r['occ_stats']['max_prob']:.4f}  meanP={r['occ_stats']['mean_prob']:.4f}  [{r['occ_stats']['status']}]", flush=True)
            print(f"  Ratio: {r['ratio']:.2f}  Verdict: {r['verdict']}  ({r['reason']})", flush=True)
            results.append(r)

    # ============================================================
    # Write SUMMARY.txt
    # ============================================================
    summary_path = os.path.join(OUTPUT_DIR, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 95 + "\n")
        f.write("  HDDNet v2 REAL-IMAGE VERIFICATION SUMMARY\n")
        f.write("=" * 95 + "\n")
        f.write(f"  Date:             {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Baseline ckpt:    {BASELINE_CKPT} (epoch {baseline_epoch})\n")
        f.write(f"  HDDNet ckpt:      {HDDNET_CKPT} (epoch {hddnet_epoch})\n")
        f.write(f"  Device:           {device}\n")
        f.write(f"  Images tested:    {len(results)}\n")
        f.write(f"  Threshold:        {THRESHOLD}\n")
        f.write("\n")
        f.write("  REFERENCE (v1 collapsed model, for comparison):\n")
        f.write("    Main maxP:  0.11 - 0.14   (decoder_main was effectively dead)\n")
        f.write("    Main meanP: 0.003 - 0.011 (near-zero contribution)\n")
        f.write("    Occ maxP:   0.999+        (decoder_occlusion did all the work)\n")
        f.write("    Occ meanP:  0.08 - 0.15\n")
        f.write("\n")
        f.write("  REFERENCE (v2 training validation, healthy):\n")
        f.write("    Main meanP: ~0.045, Occ meanP: ~0.045 (balanced)\n")
        f.write("    Ratio:      ~1.0\n")
        f.write("\n")
        f.write("  PASS criteria:  Both branches ACTIVE (maxP >= 0.35),\n")
        f.write("                  balance ratio 0.8-1.2\n")
        f.write("  PASS* criteria: Both ACTIVE but ratio slightly outside 0.8-1.2\n")
        f.write("  FAIL criteria:  Either branch COLLAPSED (maxP < 0.35),\n")
        f.write("                  OR extreme imbalance (ratio outside 0.2-5.0)\n")
        f.write("\n")
        f.write("=" * 95 + "\n")
        f.write("  PER-IMAGE RESULTS\n")
        f.write("=" * 95 + "\n\n")

        header = (f"  {'Image':<28} | {'MnMaxP':>7} | {'MnMeanP':>8} | "
                  f"{'OcMaxP':>7} | {'OcMeanP':>8} | {'Ratio':>5} | "
                  f"{'MainSt':<9} | {'OccSt':<9} | {'Verdict':<6}")
        f.write(header + "\n")
        f.write("  " + "-" * 28 + "-+-" + "-" * 7 + "-+-" + "-" * 8 + "-+-"
                + "-" * 7 + "-+-" + "-" * 8 + "-+-" + "-" * 5 + "-+-"
                + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 6 + "\n")

        pass_count = 0
        fail_count = 0
        for r in results:
            ms = r['main_stats']
            os_ = r['occ_stats']
            oblique_flag = " *" if r['is_oblique'] else ""
            line = (f"  {r['fname']:<28} | {ms['max_prob']:>7.4f} | {ms['mean_prob']:>8.4f} | "
                    f"{os_['max_prob']:>7.4f} | {os_['mean_prob']:>8.4f} | {r['ratio']:>5.2f} | "
                    f"{ms['status']:<9} | {os_['status']:<9} | {r['verdict']:<6}{oblique_flag}")
            f.write(line + "\n")
            if r['verdict'] == 'PASS':
                pass_count += 1
            elif r['verdict'] == 'FAIL':
                fail_count += 1
            else:
                pass_count += 1  # PASS* counts as pass

        f.write("\n")
        f.write(f"  TOTAL: {pass_count} PASS, {fail_count} FAIL out of {len(results)} images\n")
        f.write("\n")

        # Per-image detail block
        f.write("=" * 95 + "\n")
        f.write("  PER-IMAGE DETAIL & VERDICT REASONING\n")
        f.write("=" * 95 + "\n\n")

        for i, r in enumerate(results, 1):
            ms = r['main_stats']
            os_ = r['occ_stats']
            f.write(f"  [{i}/{len(results)}] {r['fname']}\n")
            f.write(f"    Original size:   {r['orig_size']}\n")
            if r['is_oblique']:
                f.write(f"    NOTE:            Oblique/foggy aerial photo (distribution shift)\n")
            f.write(f"    decoder_main:    maxP={ms['max_prob']:.4f}  meanP={ms['mean_prob']:.4f}  "
                    f"std={ms['std_prob']:.4f}  road%={ms['road_frac']*100:.2f}%  [{ms['status']}]\n")
            f.write(f"    decoder_occ:     maxP={os_['max_prob']:.4f}  meanP={os_['mean_prob']:.4f}  "
                    f"std={os_['std_prob']:.4f}  road%={os_['road_frac']*100:.2f}%  [{os_['status']}]\n")
            f.write(f"    Occ/Main ratio:  {r['ratio']:.2f}\n")
            f.write(f"    VERDICT:         {r['verdict']} -- {r['reason']}\n")
            f.write(f"    Comparison PNG:  {os.path.basename(r['out_path'])}\n")
            f.write("\n")

        f.write("=" * 95 + "\n")
        f.write("  END OF SUMMARY\n")
        f.write("=" * 95 + "\n")

    print(f"\nSUMMARY.txt written: {summary_path}", flush=True)
    print(f"All comparison PNGs in: {OUTPUT_DIR}", flush=True)
    print(f"\nFinal: {pass_count} PASS, {fail_count} FAIL out of {len(results)}", flush=True)


if __name__ == '__main__':
    main()
