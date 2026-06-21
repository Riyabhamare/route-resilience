"""
analyze_cloud_test_3_contrast.py

Computes the approximate LAB contrast score for the two failure regions
in cloud_test_3.png and ranks them against the full 6,226-image distribution.
"""

import os
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG_PATH = os.path.join(PROJECT_ROOT, 'real_test_img', 'cloud_test_3.png')
CSV_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'full_dataset_contrast.csv')

def get_region_contrast(region_bgr):
    # Convert to LAB
    region_lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Reshape for clustering
    pixels = region_lab.reshape(-1, 3)
    
    # Use cv2.kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    

    # The contrast is the Euclidean distance between the two dominant cluster centers in LAB space
    contrast = float(np.linalg.norm(centers[0] - centers[1]))
    return contrast

def main():
    if not os.path.exists(IMG_PATH):
        print(f"Error: Could not find {IMG_PATH}")
        return
        
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    # 1. Compute contrast for the failure regions
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    
    ul_region = img_bgr[0:h//2, 0:w//2]         # Upper-Left (Dirt road)
    ll_region = img_bgr[h//2:h, 0:w//2]         # Lower-Left (Street grid)
    
    ul_contrast = get_region_contrast(ul_region)
    ll_contrast = get_region_contrast(ll_region)
    
    print(f"--- cloud_test_3.png Failure Regions ---")
    print(f"Upper-Left (Dirt Road) Contrast:   {ul_contrast:.4f}")
    print(f"Lower-Left (Street Grid) Contrast: {ll_contrast:.4f}")
    
    # 2. Compare against full distribution
    df = pd.read_csv(CSV_PATH)
    total_images = len(df)
    
    # Percentile function
    def get_percentile(score):
        # How many images have a contrast score LOWER than this?
        lower_count = (df['contrast_score'] < score).sum()
        percentile = (lower_count / total_images) * 100
        return percentile, lower_count
        
    ul_pct, ul_rank = get_percentile(ul_contrast)
    ll_pct, ll_rank = get_percentile(ll_contrast)
    
    print(f"\n--- Distribution Ranking (Out of {total_images} images) ---")
    print(f"Upper-Left Contrast ({ul_contrast:.2f}):")
    print(f"  Rank: {ul_rank} out of {total_images} (Lowest to Highest)")
    print(f"  Percentile: Bottom {ul_pct:.2f}% of the dataset")
    
    print(f"\nLower-Left Contrast ({ll_contrast:.2f}):")
    print(f"  Rank: {ll_rank} out of {total_images} (Lowest to Highest)")
    print(f"  Percentile: Bottom {ll_pct:.2f}% of the dataset")
    
    avg_failure_contrast = (ul_contrast + ll_contrast) / 2
    avg_pct, _ = get_percentile(avg_failure_contrast)
    
    print(f"\nConclusion Check:")
    if avg_pct < 5.0:
        print(f"These regions are extreme outliers (Bottom {avg_pct:.1f}%). Even the 'low contrast' pool (Bottom 30%) may not have enough similar examples.")
    elif avg_pct < 30.0:
        print(f"These regions fall well within the Bottom 30% oversampling pool (Bottom {avg_pct:.1f}%). Oversampling should naturally target these.")
    else:
        print(f"These regions are unexpectedly NOT extreme outliers (Bottom {avg_pct:.1f}%). Contrast alone does not explain the failure.")

if __name__ == '__main__':
    main()
