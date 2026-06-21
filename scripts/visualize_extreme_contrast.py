"""
visualize_extreme_contrast.py

Creates a visual comparison grid between the failure regions in cloud_test_3.png
and 6 of the lowest-contrast training images (contrast scores 4.0 - 4.01).
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG_PATH = os.path.join(PROJECT_ROOT, 'real_test_img', 'cloud_test_3.png')
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Selected extreme low-contrast image IDs from the prior diagnostic
TRAIN_IDS = ['653446', '695052', '327075', '662044', '267645', '919589']

def overlay_mask(base_bgr, mask_gray, color=(0, 0, 255), alpha=0.5):
    # Overlay a red mask on the BGR image
    overlay = base_bgr.copy()
    m = mask_gray > 127
    overlay[m] = overlay[m] * (1 - alpha) + np.array(color) * alpha
    return overlay

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load cloud_test_3.png crops
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    ul_region = img_bgr[0:h//2, 0:w//2]         # Upper-Left (Dirt road)
    ll_region = img_bgr[h//2:h, 0:w//2]         # Lower-Left (Street grid)
    
    # Convert BGR to RGB for matplotlib
    ul_rgb = cv2.cvtColor(ul_region, cv2.COLOR_BGR2RGB)
    ll_rgb = cv2.cvtColor(ll_region, cv2.COLOR_BGR2RGB)
    
    # 2. Setup matplotlib figure
    # 2 rows, 4 columns: 
    # Row 1: UL Failure | Train 1 | Train 2 | Train 3
    # Row 2: LL Failure | Train 4 | Train 5 | Train 6
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Visual Comparison: cloud_test_3.png Failures vs. Extreme Low-Contrast Training Data (LAB 4.0)', fontsize=16)
    
    # Plot Failure Regions
    axes[0, 0].imshow(ul_rgb)
    axes[0, 0].set_title('FAILURE: Upper-Left (Dirt Road)', color='red', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(ll_rgb)
    axes[1, 0].set_title('FAILURE: Lower-Left (Street Grid)', color='red', fontweight='bold')
    axes[1, 0].axis('off')
    
    # 3. Load and plot Training Images
    for i, t_id in enumerate(TRAIN_IDS):
        row = i // 3
        col = (i % 3) + 1
        
        t_img_path = os.path.join(TRAIN_DIR, f"{t_id}_sat.jpg")
        t_mask_path = os.path.join(TRAIN_DIR, f"{t_id}_mask.png")
        
        if os.path.exists(t_img_path) and os.path.exists(t_mask_path):
            t_img = cv2.imread(t_img_path)
            t_mask = cv2.imread(t_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Overlay mask so the user can see where the road actually is
            t_over = overlay_mask(t_img, t_mask, color=(0, 0, 255), alpha=0.5)
            t_rgb = cv2.cvtColor(t_over, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(t_rgb)
            axes[row, col].set_title(f'Train ID: {t_id}\n(Road shown in red)')
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, f"Missing {t_id}", ha='center', va='center')
            axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, 'low_contrast_visual_check.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visual comparison generated at: {out_path}")

if __name__ == '__main__':
    main()
