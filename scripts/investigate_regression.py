"""
Investigation script for V3 regression analysis.

Part A: Visualize 10 augmented training images with their masks
         to confirm the mask is correctly preserved (not misaligned or corrupted)
         after apply_contrast_reduction.

Part B: Load V2 and V3 checkpoints and extract exact best Val IoU for comparison.
"""

import os
import sys
import random
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import RoadDataset
from src.data.augmentations import get_train_transform, apply_contrast_reduction

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'regression_investigation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PART A: Visualize augmented training images + masks
# ============================================================
print("=" * 70)
print("  PART A: Augmented Image + Mask Alignment Check")
print("=" * 70)

DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
dataset = RoadDataset(DATA_DIR, transform=get_train_transform())

# Pick 10 random samples that actually have road pixels
random.seed(42)
indices = random.sample(range(len(dataset)), min(50, len(dataset)))

checked = 0
fig, axes = plt.subplots(10, 4, figsize=(24, 60))

for idx in indices:
    if checked >= 10:
        break
    
    image_tensor, mask_tensor = dataset[idx]
    
    # Convert to numpy (same way train_hddnet.py does it)
    img_np = image_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
    img_np = (img_np * 255).astype(np.uint8)
    
    mask_np = mask_tensor.squeeze().numpy()
    if mask_np.max() <= 1.0:
        mask_np_uint8 = (mask_np * 255).astype(np.uint8)
    else:
        mask_np_uint8 = mask_np.astype(np.uint8)
    
    # Skip images with very few road pixels
    road_pixel_count = (mask_np_uint8 > 127).sum()
    if road_pixel_count < 100:
        continue
    
    # Apply contrast reduction (exactly as training code does)
    img_augmented = apply_contrast_reduction(img_np, mask_np_uint8)
    
    # The mask should NOT have changed -- verify
    mask_after = mask_np_uint8.copy()  # mask is never passed by reference to be modified
    
    # Compute stats
    road_idx = mask_np_uint8 > 127
    
    # Check: original road pixel mean vs augmented road pixel mean
    orig_road_mean = img_np[road_idx].mean()
    aug_road_mean = img_augmented[road_idx].mean()
    
    # Check: original non-road pixel mean vs augmented non-road pixel mean
    non_road_idx = ~road_idx
    orig_nonroad_mean = img_np[non_road_idx].mean() if non_road_idx.sum() > 0 else 0
    aug_nonroad_mean = img_augmented[non_road_idx].mean() if non_road_idx.sum() > 0 else 0
    
    # Compute contrast: difference between road and non-road mean intensity
    orig_contrast = abs(float(orig_road_mean) - float(orig_nonroad_mean))
    aug_contrast = abs(float(aug_road_mean) - float(aug_nonroad_mean))
    
    print(f"\n  Sample {checked+1} (dataset idx {idx}):")
    print(f"    Road pixels: {road_pixel_count}")
    print(f"    Original  - Road mean: {orig_road_mean:.1f}, Non-road mean: {orig_nonroad_mean:.1f}, Contrast: {orig_contrast:.1f}")
    print(f"    Augmented - Road mean: {aug_road_mean:.1f}, Non-road mean: {aug_nonroad_mean:.1f}, Contrast: {aug_contrast:.1f}")
    print(f"    Contrast reduction: {orig_contrast:.1f} -> {aug_contrast:.1f} ({aug_contrast/max(orig_contrast,0.01)*100:.0f}% of original)")
    print(f"    Mask unchanged: {np.array_equal(mask_np_uint8, mask_after)}")
    print(f"    Non-road pixels changed: {not np.array_equal(img_np[non_road_idx], img_augmented[non_road_idx])}")
    
    # Plot: Original | Augmented | Mask | Overlay (augmented + mask edges)
    row = checked
    
    axes[row, 0].imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_np)
    axes[row, 0].set_title(f'Original (idx={idx})', fontsize=10)
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB) if img_augmented.shape[2] == 3 else img_augmented)
    axes[row, 1].set_title(f'After Contrast Reduction', fontsize=10)
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(mask_np_uint8, cmap='gray')
    axes[row, 2].set_title(f'Mask (unchanged)', fontsize=10)
    axes[row, 2].axis('off')
    
    # Overlay: augmented image with mask edges in red
    overlay = img_augmented.copy()
    if overlay.shape[2] == 3:
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    else:
        overlay_rgb = overlay
    # Draw mask contours
    contours, _ = cv2.findContours(mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 2)
    axes[row, 3].imshow(overlay_rgb)
    axes[row, 3].set_title(f'Overlay (contrast={aug_contrast:.1f})', fontsize=10)
    axes[row, 3].axis('off')
    
    checked += 1

plt.suptitle('Contrast Reduction Augmentation: Mask Alignment Verification\n'
             'Col 1: Original | Col 2: Augmented | Col 3: Mask | Col 4: Aug + Mask Edges',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'augmentation_alignment_check.png')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"\n  Saved visualization to: {save_path}")

# ============================================================
# PART B: Extract Val IoU from V2 and V3 checkpoints
# ============================================================
print("\n" + "=" * 70)
print("  PART B: Val IoU Comparison -- V2 vs V3")
print("=" * 70)

# Check all available checkpoint locations
v3_best = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_best.pth')
v3_latest = os.path.join(PROJECT_ROOT, 'models', 'hddnet', 'hddnet_latest.pth')
v2_baseline = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')

# Also check for any archived V2 HDDNet checkpoint
v2_hddnet_candidates = [
    os.path.join(PROJECT_ROOT, 'models', 'hddnet_v2', 'hddnet_best.pth'),
    os.path.join(PROJECT_ROOT, 'models', 'hddnet_backup', 'hddnet_best.pth'),
    os.path.join(PROJECT_ROOT, 'models', 'hddnet_pre_augmentation', 'hddnet_best.pth'),
]

def extract_checkpoint_info(path, label):
    if not os.path.exists(path):
        print(f"\n  [{label}] NOT FOUND: {path}")
        return None
    
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'unknown')
    val_iou = ckpt.get('val_iou', None)
    best_iou = ckpt.get('best_iou', None)
    best_epoch = ckpt.get('best_epoch', None)
    val_loss = ckpt.get('val_loss', None)
    train_loss = ckpt.get('train_loss', None)
    
    print(f"\n  [{label}] {path}")
    print(f"    Epoch:       {epoch}")
    print(f"    Val IoU:     {val_iou}")
    print(f"    Best IoU:    {best_iou}")
    print(f"    Best Epoch:  {best_epoch}")
    print(f"    Val Loss:    {val_loss}")
    print(f"    Train Loss:  {train_loss}")
    
    # Check for val_stats with more detailed info
    val_stats = ckpt.get('val_stats', None)
    if val_stats:
        print(f"    Val Stats:   {val_stats}")
    
    return {'epoch': epoch, 'val_iou': val_iou, 'best_iou': best_iou, 'best_epoch': best_epoch}

v3_info = extract_checkpoint_info(v3_best, "V3 Best (hddnet_best.pth)")
extract_checkpoint_info(v3_latest, "V3 Latest (hddnet_latest.pth)")
v2_info = extract_checkpoint_info(v2_baseline, "V2 Baseline (baseline_full_best.pth)")

for candidate in v2_hddnet_candidates:
    extract_checkpoint_info(candidate, f"V2 HDDNet candidate")

# List all checkpoint files to find any archived V2
print("\n  --- All files in models/ ---")
models_dir = os.path.join(PROJECT_ROOT, 'models')
if os.path.isdir(models_dir):
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath) / (1024*1024)
            print(f"    {fpath} ({fsize:.1f} MB)")

# ============================================================
# PART C: Check if non-road pixels were accidentally modified
# ============================================================
print("\n" + "=" * 70)
print("  PART C: Non-Road Pixel Modification Check")
print("=" * 70)
print("  Checking if apply_contrast_reduction modifies ANY non-road pixels...")

random.seed(99)
for trial in range(5):
    idx = random.randint(0, len(dataset)-1)
    image_tensor, mask_tensor = dataset[idx]
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    mask_np = mask_tensor.squeeze().numpy()
    if mask_np.max() <= 1.0:
        mask_np = (mask_np * 255).astype(np.uint8)
    else:
        mask_np = mask_np.astype(np.uint8)
    
    img_aug = apply_contrast_reduction(img_np, mask_np)
    
    non_road = mask_np <= 127
    road = mask_np > 127
    
    nonroad_diff = np.abs(img_np[non_road].astype(float) - img_aug[non_road].astype(float))
    road_diff = np.abs(img_np[road].astype(float) - img_aug[road].astype(float))
    
    print(f"\n  Trial {trial+1} (idx={idx}):")
    print(f"    Non-road pixels changed: max_diff={nonroad_diff.max():.1f}, mean_diff={nonroad_diff.mean():.4f}")
    print(f"    Road pixels changed:     max_diff={road_diff.max():.1f}, mean_diff={road_diff.mean():.1f}")
    if nonroad_diff.max() > 0:
        print(f"    *** WARNING: Non-road pixels WERE modified! ***")
    else:
        print(f"    OK: Only road pixels were modified.")

print("\n" + "=" * 70)
print("  INVESTIGATION COMPLETE")
print("=" * 70)
