"""
check_deepglobe_contrast.py

Analyzes the DeepGlobe training dataset to determine if it is dominated
by high-contrast roads. It calculates the average RGB standard deviation
and the color distance between road pixels and immediately adjacent 
background pixels.
"""

import os
import glob
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')

def main():
    img_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, '*_sat.jpg')))
    if not img_files:
        print("No training images found!")
        return

    # Sample up to 100 images to keep it fast
    sample_files = img_files[:100]
    
    contrast_scores = []
    
    for img_path in sample_files:
        mask_path = img_path.replace('_sat.jpg', '_mask.png')
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
            
        # Convert to LAB color space for better perceptual distance
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Binary mask
        road_mask = mask > 127
        bg_mask = ~road_mask
        
        if not road_mask.any() or not bg_mask.any():
            continue
            
        # To get the contrast *at the boundary*, we dilate the road mask 
        # to find the immediately surrounding background
        kernel = np.ones((5,5), np.uint8)
        dilated_road = cv2.dilate(mask, kernel, iterations=2) > 127
        surrounding_bg_mask = dilated_road & bg_mask
        
        if not surrounding_bg_mask.any():
            continue
            
        road_pixels = img_lab[road_mask]
        surround_pixels = img_lab[surrounding_bg_mask]
        
        road_mean = np.mean(road_pixels, axis=0)
        surround_mean = np.mean(surround_pixels, axis=0)
        
        # Euclidean distance in LAB space is a proxy for perceptual contrast
        contrast = np.linalg.norm(road_mean - surround_mean)
        contrast_scores.append(contrast)

    if contrast_scores:
        avg_contrast = np.mean(contrast_scores)
        median_contrast = np.median(contrast_scores)
        low_contrast_count = sum(1 for c in contrast_scores if c < 20.0) # Arbitrary low threshold in LAB
        
        print("\n--- DeepGlobe Training Data Contrast Analysis ---")
        print(f"Images Sampled: {len(contrast_scores)}")
        print(f"Average LAB Distance (Road vs Surroundings): {avg_contrast:.2f}")
        print(f"Median LAB Distance:  {median_contrast:.2f}")
        print(f"Low-Contrast Roads (<20 LAB distance): {low_contrast_count} out of {len(contrast_scores)} ({(low_contrast_count/len(contrast_scores))*100:.1f}%)")
        if avg_contrast > 30:
            print("Conclusion: The dataset is DOMINATED by high-contrast roads (e.g., dark asphalt vs bright fields).")
        else:
            print("Conclusion: The dataset contains a significant mix of contrasts.")
    else:
        print("Could not process any images.")

if __name__ == '__main__':
    main()
