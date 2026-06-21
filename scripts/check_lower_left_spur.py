import os, sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.test_new_real_images import load_model, load_baseline_model, predict_roads

def overlay_mask(img_rgb, mask, color=(255, 0, 0), alpha=0.5):
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading models...")
    hddnet = load_model(device)
    baseline = load_baseline_model(device)

    img_name = '17193_sat.jpg'
    img_path = os.path.join(PROJECT_ROOT, 'real_test_img', img_name)
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    configs = [
        ("M1: HDDNet", False, False),
        ("M2: Max-Ens", True, False),
        ("M3: TTA", False, True),
        ("M4: Full", True, True)
    ]
    
    # Define crop region for lower-left spur
    # Visual estimation: lower left quadrant, maybe y: 700-1000, x: 0-350
    y1, y2 = 700, 1000
    x1, x2 = 0, 350
    
    img_crop = img_rgb[y1:y2, x1:x2]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(img_crop)
    axes[0].set_title("Original Crop")
    axes[0].axis('off')
    
    print("\n--- Lower-Left Crop Analysis ---")
    print(f"Crop bounds: y={y1}:{y2}, x={x1}:{x2}")
    
    for i, (name, ens, tta) in enumerate(configs):
        pred, _ = predict_roads(img_path, hddnet, device, 0.5, baseline, ens, tta)
        pred_crop = pred[y1:y2, x1:x2]
        
        px_count = int((pred_crop == 255).sum())
        print(f"{name:<15} | Crop Road Px: {px_count:>5}")
        
        overlaid = overlay_mask(img_crop, pred_crop, color=(255, 0, 0), alpha=0.6)
        axes[i+1].imshow(overlaid)
        axes[i+1].set_title(f"{name}\n{px_count} px")
        axes[i+1].axis('off')
        
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'new_real_test')
    save_path = os.path.join(out_dir, 'spur_zoom.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved {save_path}")
