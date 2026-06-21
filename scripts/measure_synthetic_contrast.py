"""
measure_synthetic_contrast.py

Applies the synthetic contrast reduction to 3 sample images and computes
the exact LAB contrast score (same metric as full dataset analysis) at
different alpha levels.
"""

import os
import glob
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')

def compute_contrast(img_bgr, mask_gray):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    road_mask = mask_gray > 127
    bg_mask = ~road_mask
    
    kernel = np.ones((5,5), np.uint8)
    dilated_road = cv2.dilate(mask_gray, kernel, iterations=2) > 127
    surrounding_bg_mask = dilated_road & bg_mask
    
    if not road_mask.any() or not surrounding_bg_mask.any():
        return None
        
    road_mean = np.mean(img_lab[road_mask], axis=0)
    surround_mean = np.mean(img_lab[surrounding_bg_mask], axis=0)
    
    return float(np.linalg.norm(road_mean - surround_mean))

def reduce_road_contrast(img_bgr, mask_gray, alpha):
    h, w = img_bgr.shape[:2]
    scale = 0.25
    small_w, small_h = int(w * scale), int(h * scale)
    
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    small_bg = cv2.inpaint(small_img, small_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    bg_only = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    
    road_idx = mask_gray > 127
    out_img = img_bgr.copy().astype(np.float32)
    bg_only = bg_only.astype(np.float32)
    
    out_img[road_idx] = out_img[road_idx] * (1.0 - alpha) + bg_only[road_idx] * alpha
    return np.clip(out_img, 0, 255).astype(np.uint8)

def main():
    img_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, '*_sat.jpg')))[:3]
    
    # Specific alphas for each image
    alphas_dict = {
        img_paths[0]: [0.60, 0.75, 0.90], # Image 1
        img_paths[1]: [0.60, 0.70, 0.75, 0.80, 0.85, 0.90], # Image 2 (Finer grid)
        img_paths[2]: [0.60, 0.75, 0.90]  # Image 3
    }
    
    print("--- Synthetic Contrast Reduction LAB Score Check ---")
    for img_path in img_paths:
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        img_id = os.path.basename(img_path)
        alphas = alphas_dict[img_path]
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        orig_contrast = compute_contrast(img, mask)
        print(f"\nImage: {img_id}")
        print(f"  Original Contrast: {orig_contrast:.4f}")
        
        for alpha in alphas:
            aug_img = reduce_road_contrast(img, mask, alpha)
            aug_contrast = compute_contrast(aug_img, mask)
            print(f"  Alpha={alpha:.2f} Contrast:  {aug_contrast:.4f}")

if __name__ == '__main__':
    main()
