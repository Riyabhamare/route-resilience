"""
compute_full_dataset_contrast.py

Computes the LAB color distance between road pixels and immediately 
surrounding background pixels for the ENTIRE DeepGlobe training dataset.
Saves the results to a CSV, outputs summary statistics, and generates 
a list of the lowest-contrast 30% of images for oversampling.
"""

import os
import glob
import cv2
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

def process_image(img_path):
    try:
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        if not os.path.exists(mask_path):
            return None
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            return None
            
        # Convert to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        road_mask = mask > 127
        bg_mask = ~road_mask
        
        if not road_mask.any() or not bg_mask.any():
            return None
            
        # Dilate road to find surrounding background boundary
        kernel = np.ones((5,5), np.uint8)
        dilated_road = cv2.dilate(mask, kernel, iterations=2) > 127
        surrounding_bg_mask = dilated_road & bg_mask
        
        if not surrounding_bg_mask.any():
            return None
            
        road_pixels = img_lab[road_mask]
        surround_pixels = img_lab[surrounding_bg_mask]
        
        road_mean = np.mean(road_pixels, axis=0)
        surround_mean = np.mean(surround_pixels, axis=0)
        
        contrast = float(np.linalg.norm(road_mean - surround_mean))
        
        image_id = os.path.basename(img_path).replace('_sat.jpg', '')
        return (image_id, contrast)
    except Exception as e:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, '*_sat.jpg')))
    if not img_files:
        print("No training images found in Dataset/deep/train")
        return
        
    print(f"Found {len(img_files)} images. Starting full dataset analysis...", flush=True)
    
    results = []
    # Use multiprocessing to speed up processing of 6k+ 1024x1024 images
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image, path): path for path in img_files}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            count += 1
            if count % 500 == 0:
                print(f"Processed {count}/{len(img_files)} images...", flush=True)
                
    if not results:
        print("Failed to process any images.")
        return
        
    # Sort from lowest to highest contrast
    results.sort(key=lambda x: x[1])
    
    # 1. Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'full_dataset_contrast.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'contrast_score'])
        for r in results:
            writer.writerow([r[0], f"{r[1]:.4f}"])
            
    # 2. Compute Distribution Stats
    scores = [r[1] for r in results]
    min_c = np.min(scores)
    max_c = np.max(scores)
    mean_c = np.mean(scores)
    median_c = np.median(scores)
    below_20_count = sum(1 for s in scores if s < 20.0)
    
    print("\n=======================================================", flush=True)
    print("  FULL DATASET CONTRAST DISTRIBUTION", flush=True)
    print("=======================================================", flush=True)
    print(f"Total Valid Images Processed: {len(results)}", flush=True)
    print(f"Min Contrast Score:           {min_c:.4f}", flush=True)
    print(f"Max Contrast Score:           {max_c:.4f}", flush=True)
    print(f"Mean Contrast Score:          {mean_c:.4f}", flush=True)
    print(f"Median Contrast Score:        {median_c:.4f}", flush=True)
    print(f"Images < 20 Contrast:         {below_20_count} ({below_20_count/len(results)*100:.1f}%)", flush=True)
    print("=======================================================", flush=True)
    print(f"Saved full sorted list to: {csv_path}", flush=True)
    
    # 3. Save bottom 30% to txt
    bottom_30_idx = int(0.30 * len(results))
    bottom_30_results = results[:bottom_30_idx]
    
    txt_path = os.path.join(OUTPUT_DIR, 'low_contrast_image_ids.txt')
    with open(txt_path, 'w') as f:
        for r in bottom_30_results:
            f.write(f"{r[0]}\n")
            
    print(f"Saved bottom {bottom_30_idx} image IDs (lowest 30%) to: {txt_path}", flush=True)
    print("=======================================================", flush=True)

if __name__ == '__main__':
    main()
