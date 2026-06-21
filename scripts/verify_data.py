"""

Section 2 -- Dataset Verification Script
Verifies the DeepGlobe Road Extraction dataset structure before any training.

Expected structure:
  Dataset/deep/train/  -> <id>_sat.jpg and <id>_mask.png pairs
  Dataset/deep/valid/  -> <id>_sat.jpg and <id>_mask.png pairs
  Dataset/deep/test/   -> <id>_sat.jpg only (no masks)

Outputs (ASCII-safe for Windows console):
  debug/sample_pair_check.png — side-by-side visualization of one sat/mask pair
"""

import os
import sys
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
VALID_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'valid')
TEST_DIR  = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'test')
DEBUG_DIR = os.path.join(PROJECT_ROOT, 'debug')

def verify_folder(folder_path, folder_name, expect_masks=True):
    """Verify a dataset folder and return counts."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Expected dataset folder not found: {folder_path}"
        )

    sat_files = sorted(glob.glob(os.path.join(folder_path, '*_sat.jpg')))
    mask_files = sorted(glob.glob(os.path.join(folder_path, '*_mask.png')))

    print(f"\n{'='*60}")
    print(f"  {folder_name}: {folder_path}")
    print(f"{'='*60}")
    print(f"  Satellite images (*_sat.jpg): {len(sat_files)}")
    print(f"  Mask images (*_mask.png):     {len(mask_files)}")

    if expect_masks:
        if len(sat_files) != len(mask_files):
            print(f"  [WARN] Unequal count! {len(sat_files)} sat vs {len(mask_files)} mask")
        else:
            print(f"  [OK] Equal count of sat and mask files")

        # Verify each sat file has a matching mask
        missing_masks = []
        for sat_path in sat_files:
            basename = os.path.basename(sat_path)
            mask_name = basename.replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(folder_path, mask_name)
            if not os.path.exists(mask_path):
                missing_masks.append(mask_name)

        if missing_masks:
            print(f"  [WARN] {len(missing_masks)} sat files have no matching mask:")
            for m in missing_masks[:5]:
                print(f"      Missing: {m}")
            if len(missing_masks) > 5:
                print(f"      ... and {len(missing_masks) - 5} more")
        else:
            print(f"  [OK] All satellite images have matching masks")
    else:
        if len(mask_files) > 0:
            print(f"  [i] Note: {len(mask_files)} mask files found (unexpected for test set)")
        else:
            print(f"  [OK] No mask files (expected for test set)")

    return sat_files, mask_files


def inspect_sample_pair(sat_path, mask_path):
    """Open one sat/mask pair and print detailed info."""
    print(f"\n{'='*60}")
    print(f"  Sample Pair Inspection")
    print(f"{'='*60}")
    print(f"  Sat file:  {os.path.basename(sat_path)}")
    print(f"  Mask file: {os.path.basename(mask_path)}")

    sat_img = cv2.imread(sat_path, cv2.IMREAD_COLOR)
    if sat_img is None:
        raise IOError(f"Failed to read satellite image: {sat_path}")
    sat_rgb = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise IOError(f"Failed to read mask image: {mask_path}")

    print(f"\n  Satellite image:")
    print(f"    Shape: {sat_rgb.shape}")
    print(f"    Dtype: {sat_rgb.dtype}")
    print(f"    Value range: [{sat_rgb.min()}, {sat_rgb.max()}]")

    print(f"\n  Mask image:")
    print(f"    Shape: {mask_img.shape}")
    print(f"    Dtype: {mask_img.dtype}")
    unique_vals = np.unique(mask_img)
    print(f"    Unique pixel values: {unique_vals}")
    print(f"    Number of unique values: {len(unique_vals)}")
    road_pixels = np.sum(mask_img > 127)
    total_pixels = mask_img.size
    print(f"    Road pixels (>127): {road_pixels} ({100*road_pixels/total_pixels:.2f}%)")
    print(f"    Background pixels:  {total_pixels - road_pixels} ({100*(total_pixels-road_pixels)/total_pixels:.2f}%)")

    return sat_rgb, mask_img


def save_visualization(sat_rgb, mask_img, save_path):
    """Save side-by-side sat + mask visualization."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(sat_rgb)
    axes[0].set_title('Satellite Image (_sat.jpg)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title('Road Mask (_mask.png)', fontsize=14)
    axes[1].axis('off')

    plt.suptitle(f'DeepGlobe Road Extraction — Sample Pair', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Visualization saved to: {save_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  DeepGlobe Road Dataset — Verification Script")
    print("=" * 60)

    # Step 1: Verify all three folders
    train_sats, train_masks = verify_folder(TRAIN_DIR, "TRAIN", expect_masks=True)
    valid_sats, valid_masks = verify_folder(VALID_DIR, "VALID", expect_masks=True)
    test_sats, _            = verify_folder(TEST_DIR,  "TEST",  expect_masks=False)

    # Step 2: Inspect one sample pair from train/
    if len(train_sats) == 0:
        print("  [FAIL] No satellite images found in train folder!")
        sys.exit(1)

    sample_sat = train_sats[0]
    sample_mask_name = os.path.basename(sample_sat).replace('_sat.jpg', '_mask.png')
    sample_mask = os.path.join(TRAIN_DIR, sample_mask_name)

    if not os.path.exists(sample_mask):
        raise FileNotFoundError(
            f"Expected mask not found: {sample_mask}\n"
            f"Looked for mask matching satellite image: {sample_sat}"
        )

    sat_rgb, mask_img = inspect_sample_pair(sample_sat, sample_mask)

    # Step 3: Save visualization
    save_path = os.path.join(DEBUG_DIR, 'sample_pair_check.png')
    save_visualization(sat_rgb, mask_img, save_path)

    print(f"\n{'='*60}")
    print("  [OK] Dataset verification PASSED")
    print(f"{'='*60}")
    print(f"  Train pairs: {len(train_sats)}")
    print(f"  Valid pairs: {len(valid_sats)}")
    print(f"  Test images: {len(test_sats)} (no masks)")
    print(f"\n  Review {save_path}")
    print(f"  and confirm it looks correct before proceeding.\n")
