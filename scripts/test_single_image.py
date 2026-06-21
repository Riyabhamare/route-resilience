"""
Test a single image using both Baseline U-Net and HDDNet.
Outputs a side-by-side comparison plot.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_unet import get_baseline_model
from src.models.hddnet import HDDNet
from src.data.augmentations import get_val_transform

BASELINE_CKPT = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
HDDNET_CKPT = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'test_single_image_result.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load image
    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    print(f"Loading image: {img_path}")
    image = np.array(Image.open(img_path).convert("RGB"))
    
    # Store original for plotting
    orig_image = image.copy()

    # Preprocess
    transform = get_val_transform()
    augmented = transform(image=image)
    cropped_img = augmented['image']
    
    # Store cropped image for plotting so it aligns with the 512x512 prediction
    orig_image = cropped_img.copy()
    
    # Normalize exactly as dataset.py does
    tensor_img = cropped_img.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Load Baseline
    print("Loading Baseline U-Net...")
    baseline_model = get_baseline_model().to(device)
    baseline_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    baseline_model.eval()

    # Load HDDNet
    print("Loading HDDNet...")
    hddnet_model = HDDNet().to(device)
    hddnet_ckpt = torch.load(HDDNET_CKPT, map_location=device, weights_only=False)
    hddnet_model.load_state_dict(hddnet_ckpt['model_state_dict'])
    hddnet_model.eval()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        base_out = baseline_model(tensor_img)
        hdd_out, _, _ = hddnet_model(tensor_img)

        base_pred = (torch.sigmoid(base_out) > 0.5).float().squeeze().cpu().numpy()
        hdd_pred = (torch.sigmoid(hdd_out) > 0.5).float().squeeze().cpu().numpy()

    # Plotting
    print("Generating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(orig_image)
    axes[1].imshow(base_pred, cmap='Reds', alpha=0.5)
    axes[1].set_title('Baseline Prediction', fontsize=16)
    axes[1].axis('off')

    axes[2].imshow(orig_image)
    axes[2].imshow(hdd_pred, cmap='Blues', alpha=0.5)
    axes[2].set_title('HDDNet Prediction', fontsize=16)
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved result to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
