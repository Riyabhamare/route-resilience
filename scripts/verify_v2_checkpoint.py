"""
V2 Checkpoint Integrity Verification
Runs Part 1 (branch health on all 8 images) and Part 2 (cloud_test_3 region analysis)
using ONLY the archived V2 checkpoint: models/hddnet_archive_v2/hddnet_best.pth
"""
import os, sys, glob, numpy as np, cv2, torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet

V2_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
REAL_TEST_DIR = os.path.join(PROJECT_ROOT, 'real_test_img')
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
                            borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return lb, scale, pad_top, pad_left

def run_inference(model, img_rgb, device):
    orig_h, orig_w = img_rgb.shape[:2]
    lb_img, scale, pad_top, pad_left = letterbox(img_rgb, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        final_logits, main_logits, occ_logits = model(tensor)
        final_prob = torch.sigmoid(final_logits).squeeze().cpu().numpy()
        main_prob = torch.sigmoid(main_logits).squeeze().cpu().numpy()
        occ_prob = torch.sigmoid(occ_logits).squeeze().cpu().numpy()

    # Map back to original image space
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    full_pred = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    full_pred[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE] = final_prob
    pred_orig = cv2.resize(full_pred[pad_top:pad_top+new_h, pad_left:pad_left+new_w],
                           (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return final_prob, main_prob, occ_prob, pred_orig, crop_start

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load V2 checkpoint
    print("=" * 70)
    print("  V2 CHECKPOINT INTEGRITY VERIFICATION")
    print("=" * 70)
    print(f"  Checkpoint: {V2_CKPT}")
    print(f"  File exists: {os.path.exists(V2_CKPT)}")
    file_size_mb = os.path.getsize(V2_CKPT) / (1024*1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    model = HDDNet().to(device)
    ckpt = torch.load(V2_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Loaded successfully. Epoch: {ckpt.get('epoch')}")
    print(f"  Stored Val IoU: {ckpt.get('val_iou')}")
    print(f"  Stored Best IoU: {ckpt.get('best_iou')}")

    # ========== PART 1: Branch Health on All Images ==========
    print(f"\n{'=' * 70}")
    print("  PART 1: Branch Health Verification (all 8 images)")
    print(f"{'=' * 70}")

    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(REAL_TEST_DIR, ext)))
    all_images = sorted(set(all_images))
    print(f"  Found {len(all_images)} images\n")

    header = f"  {'Image':<28} | {'MnMaxP':>7} | {'MnMeanP':>8} | {'OcMaxP':>7} | {'OcMeanP':>8} | {'Ratio':>5} | {'Verdict':<6}"
    print(header)
    print("  " + "-" * 90)

    for img_path in all_images:
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  SKIP: {fname} (could not read)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        final_prob, main_prob, occ_prob, pred_orig, crop_start = run_inference(model, img_rgb, device)

        # Branch stats on the 512x512 crop
        mn_max = float(main_prob.max())
        mn_mean = float(main_prob.mean())
        oc_max = float(occ_prob.max())
        oc_mean = float(occ_prob.mean())
        ratio = oc_mean / mn_mean if mn_mean > 1e-6 else float('inf')

        # Verdict
        if mn_max < 0.35 or oc_max < 0.35:
            verdict = "FAIL"
        elif 0.8 <= ratio <= 1.2:
            verdict = "PASS"
        elif 0.2 <= ratio <= 5.0:
            verdict = "PASS*"
        else:
            verdict = "FAIL"

        print(f"  {fname:<28} | {mn_max:>7.4f} | {mn_mean:>8.4f} | {oc_max:>7.4f} | {oc_mean:>8.4f} | {ratio:>5.2f} | {verdict:<6}")

    # ========== PART 2: cloud_test_3.png Region Analysis ==========
    print(f"\n{'=' * 70}")
    print("  PART 2: cloud_test_3.png Region Analysis (V2 checkpoint)")
    print(f"{'=' * 70}")

    ct3_path = os.path.join(REAL_TEST_DIR, 'cloud_test_3.png')
    img_bgr = cv2.imread(ct3_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    final_prob, main_prob, occ_prob, pred_orig, crop_start = run_inference(model, img_rgb, device)

    print(f"  Image size: {orig_w}x{orig_h}")
    print(f"  Prediction map size: {pred_orig.shape[1]}x{pred_orig.shape[0]}")

    ul_region = pred_orig[0:orig_h//2, 0:orig_w//2]
    ll_region = pred_orig[orig_h//2:orig_h, 0:orig_w//2]

    for name, region in [("Upper-Left (Dirt Road)", ul_region), ("Lower-Left (Street Grid)", ll_region)]:
        print(f"\n  --- {name} ---")
        print(f"  Max Probability:   {np.max(region):.4f}")
        print(f"  99th Percentile:   {np.percentile(region, 99):.4f}")
        print(f"  95th Percentile:   {np.percentile(region, 95):.4f}")
        print(f"  Mean Probability:  {np.mean(region):.4f}")
        print(f"  Pixels > 0.50:     {(region > 0.50).sum()}")
        print(f"  Pixels > 0.35:     {(region > 0.35).sum()}")
        print(f"  Pixels > 0.20:     {(region > 0.20).sum()}")

    print(f"\n{'=' * 70}")
    print("  VERIFICATION COMPLETE")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
