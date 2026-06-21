"""
Run diagnose_low_contrast region analysis using the V2 archived checkpoint
to establish the actual V2 baseline numbers on cloud_test_3.png.
"""
import os, sys, numpy as np, cv2, torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.models.hddnet import HDDNet

IMAGE_PATH = os.path.join(PROJECT_ROOT, 'real_test_img', 'cloud_test_3.png')
V2_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
V3_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024

def letterbox(image, target_size=1024):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    lb_img = cv2.copyMakeBorder(resized, pad_top, target_size - new_h - pad_top,
                                 pad_left, target_size - new_w - pad_left,
                                 borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return lb_img, scale, pad_top, pad_left

def run_diagnosis(ckpt_path, label):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HDDNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    lb_img, scale, pad_top, pad_left = letterbox(img_rgb, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        final_logits, _, _ = model(tensor)
        prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    full_pred = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    full_pred[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE] = prob_crop
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    pred_scaled = full_pred[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    pred_orig = cv2.resize(pred_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    ul_region = pred_orig[0:orig_h//2, 0:orig_w//2]
    ll_region = pred_orig[orig_h//2:orig_h, 0:orig_w//2]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch')}, Val IoU: {ckpt.get('val_iou')}")
    print(f"{'='*60}")

    for name, region in [("Upper-Left (Dirt Road)", ul_region), ("Lower-Left (Street Grid)", ll_region)]:
        print(f"\n  --- {name} ---")
        print(f"  Max Probability:   {np.max(region):.4f}")
        print(f"  99th Percentile:   {np.percentile(region, 99):.4f}")
        print(f"  95th Percentile:   {np.percentile(region, 95):.4f}")
        print(f"  Mean Probability:  {np.mean(region):.4f}")
        print(f"  Pixels > 0.50:     {(region > 0.50).sum()}")
        print(f"  Pixels > 0.35:     {(region > 0.35).sum()}")
        print(f"  Pixels > 0.20:     {(region > 0.20).sum()}")

print("Running cloud_test_3.png diagnosis with BOTH checkpoints...\n")
run_diagnosis(V2_CKPT, "V2 (Pre-Augmentation, Archived)")
run_diagnosis(V3_CKPT, "V3 (Post-Augmentation, Current)")
print("\n\nDone.")
