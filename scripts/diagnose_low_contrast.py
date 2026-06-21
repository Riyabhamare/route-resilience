"""
diagnose_low_contrast.py

Analyzes the raw unthresholded probability predictions for cloud_test_3.png,
specifically targeting the upper-left (curved dirt road) and lower-left 
(residential street grid) regions.

Also generates visualizations thresholded at 0.35 vs 0.50.
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet

IMAGE_PATH = os.path.join(PROJECT_ROOT, 'real_test_img', 'cloud_test_3.png')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'low_contrast_diagnosis')
HDDNET_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')

MODEL_SIZE = 512
LETTERBOX_SIZE = 1024

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

def overlay_mask(base_rgb, mask, color, alpha=0.55):
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(np.float32)
    for c_idx, c_val in enumerate(color):
        out[:, :, c_idx] = out[:, :, c_idx] * (1 - alpha * m) + c_val * alpha * m * 255
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Model
    print("Loading HDDNet...", flush=True)
    hddnet_model = HDDNet().to(device)
    ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(ckpt['model_state_dict'])
    hddnet_model.eval()

    # 2. Load and preprocess image
    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Could not read {IMAGE_PATH}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    lb_img, scale, pad_top, pad_left = letterbox(img_rgb, LETTERBOX_SIZE)
    tensor, crop_start = preprocess_for_model(lb_img)
    tensor = tensor.to(device)

    # 3. Inference
    with torch.no_grad():
        final_logits, _, _ = hddnet_model(tensor)
        final_prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    # We want to analyze the original image space, not the letterboxed canvas.
    # The image is scaled and placed at (pad_top, pad_left) in the 1024x1024 canvas.
    # The model sees a 512x512 crop starting at crop_start (256, 256).
    # First, let's map the 512x512 prediction back to the 1024x1024 canvas.
    full_pred = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    full_pred[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE] = final_prob_crop

    # Now extract just the scaled image region from the canvas
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pred_scaled = full_pred[pad_top:pad_top+new_h, pad_left:pad_left+new_w]

    # Resize back to original image dimensions for precise analysis
    pred_orig = cv2.resize(pred_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    print(f"\nOriginal Image Size: {orig_w}x{orig_h}")
    print(f"Prediction Map Size: {pred_orig.shape[1]}x{pred_orig.shape[0]}")

    # 4. Region Analysis
    # Region 1: Upper-left (Curved dirt road)
    # Define bounding box: x in [0, w//2], y in [0, h//2]
    ul_region = pred_orig[0:orig_h//2, 0:orig_w//2]
    
    # Region 2: Lower-left (Dense residential grid)
    # Define bounding box: x in [0, w//2], y in [h//2, orig_h]
    ll_region = pred_orig[orig_h//2:orig_h, 0:orig_w//2]

    def print_stats(name, region):
        print(f"\n--- {name} Region Stats ---")
        print(f"Max Probability:       {np.max(region):.4f}")
        print(f"99th Percentile:       {np.percentile(region, 99):.4f}")
        print(f"95th Percentile:       {np.percentile(region, 95):.4f}")
        print(f"Mean Probability:      {np.mean(region):.4f}")
        print(f"Pixels > 0.50 (Road):  {(region > 0.50).sum()}")
        print(f"Pixels > 0.35 (Mod.):  {(region > 0.35).sum()}")
        print(f"Pixels > 0.20 (Weak):  {(region > 0.20).sum()}")

    print_stats("Upper-Left (Dirt Road)", ul_region)
    print_stats("Lower-Left (Street Grid)", ll_region)

    # 5. Visual Comparison (0.5 vs 0.35 vs 0.2)
    mask_050 = (pred_orig > 0.50).astype(np.float32)
    mask_035 = (pred_orig > 0.35).astype(np.float32)
    mask_020 = (pred_orig > 0.20).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    overlay_050 = overlay_mask(img_rgb, mask_050, color=(1.0, 0.1, 0.1))
    axes[1].imshow(overlay_050)
    axes[1].set_title('Threshold = 0.50 (Current)\nRed Mask')
    axes[1].axis('off')

    overlay_035 = overlay_mask(img_rgb, mask_035, color=(1.0, 0.5, 0.1))
    axes[2].imshow(overlay_035)
    axes[2].set_title('Threshold = 0.35\nOrange Mask')
    axes[2].axis('off')

    overlay_020 = overlay_mask(img_rgb, mask_020, color=(1.0, 0.8, 0.1))
    axes[3].imshow(overlay_020)
    axes[3].set_title('Threshold = 0.20\nYellow Mask')
    axes[3].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'threshold_comparison.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\nSaved threshold comparison to: {out_path}")

    # Save raw heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_orig, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Raw Probability Heatmap")
    plt.axis('off')
    heatmap_path = os.path.join(OUTPUT_DIR, 'raw_heatmap.png')
    plt.savefig(heatmap_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved raw heatmap to: {heatmap_path}")

if __name__ == '__main__':
    main()
