"""
measure_7_images.py

Tests the alpha=0.85 synthetic contrast reduction on 7 images evenly 
sampled across the entire contrast distribution to verify that it 
naturally regulates its strength based on terrain texture without 
requiring a classifier.
"""

import os
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
CSV_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'full_dataset_contrast.csv')

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
    df = pd.read_csv(CSV_PATH)
    
    # Sample 7 images evenly across the distribution indices
    total_imgs = len(df)
    indices = np.linspace(0, total_imgs - 1, 7, dtype=int)
    
    print("--- Broader 7-Image Contrast Reduction Check (Alpha=0.85) ---")
    for idx in indices:
        row = df.iloc[idx]
        img_id = str(row['image_id']).zfill(6)
        orig_csv_contrast = row['contrast_score']
        
        # Some IDs in the CSV might be missing leading zeros depending on pandas parsing,
        # but the original files don't have leading zeros if they are just numbers.
        # Actually, let's just use the exact string from the CSV.
        img_id_str = str(int(row['image_id'])) if isinstance(row['image_id'], (int, float)) else str(row['image_id'])
        
        img_path = os.path.join(TRAIN_DIR, f"{img_id_str}_sat.jpg")
        mask_path = os.path.join(TRAIN_DIR, f"{img_id_str}_mask.png")
        
        if not os.path.exists(img_path):
            # Fallback in case of leading zeros issue
            img_path = os.path.join(TRAIN_DIR, f"{img_id}_sat.jpg")
            mask_path = os.path.join(TRAIN_DIR, f"{img_id}_mask.png")
            
        if not os.path.exists(img_path):
            print(f"Skipping {img_id_str}, file not found.")
            continue
            
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        orig_contrast = compute_contrast(img, mask)
        aug_img = reduce_road_contrast(img, mask, alpha=0.85)
        aug_contrast = compute_contrast(aug_img, mask)
        
        print(f"\nImage: {img_id_str}_sat.jpg (Percentile: ~{(idx/total_imgs)*100:.1f}%)")
        print(f"  Original Contrast: {orig_contrast:.4f}")
        print(f"  Alpha=0.85:        {aug_contrast:.4f}")

if __name__ == '__main__':
    main()
