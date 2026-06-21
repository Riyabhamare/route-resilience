"""
test_synthetic_contrast.py

Tests a synthetic contrast-reduction augmentation function designed to
simulate low-contrast roads by alpha-blending road pixels with an 
inpainted background estimation.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

def reduce_road_contrast(img_bgr, mask_gray, alpha=0.7):
    """
    Blends road pixels towards their local surrounding background colors.
    Uses downscaled cv2.inpaint to quickly estimate the background color 
    beneath the road, then alpha-blends it with the original road.
    """
    h, w = img_bgr.shape[:2]
    
    # Fast inpainting on downscaled image (1/4 size)
    scale = 0.25
    small_w, small_h = int(w * scale), int(h * scale)
    
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # Inpaint: fill the mask area with surrounding pixels
    small_bg = cv2.inpaint(small_img, small_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Upscale back to original size
    bg_only = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Alpha blend only on the road pixels
    road_idx = mask_gray > 127
    
    out_img = img_bgr.copy().astype(np.float32)
    bg_only = bg_only.astype(np.float32)
    
    out_img[road_idx] = out_img[road_idx] * (1.0 - alpha) + bg_only[road_idx] * alpha
    
    return np.clip(out_img, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get 3 random high-quality images (we'll just take the first 3)
    img_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, '*_sat.jpg')))[:3]
    
    alphas = [0.6, 0.75, 0.9]
    
    fig, axes = plt.subplots(len(img_paths), len(alphas) + 1, figsize=(20, 15))
    fig.suptitle('Synthetic Low-Contrast Augmentation Test', fontsize=18)
    
    for row, img_path in enumerate(img_paths):
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot original
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Original\n{os.path.basename(img_path)}")
        axes[row, 0].axis('off')
        
        # Plot augmentations
        for col, alpha in enumerate(alphas):
            aug_bgr = reduce_road_contrast(img, mask, alpha=alpha)
            aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
            
            axes[row, col + 1].imshow(aug_rgb)
            axes[row, col + 1].set_title(f"Contrast Reduction (Alpha={alpha})")
            axes[row, col + 1].axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, 'synthetic_contrast_test.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved synthetic contrast test grid to: {out_path}")

if __name__ == '__main__':
    main()
