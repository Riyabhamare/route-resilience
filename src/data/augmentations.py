"""
Augmentations for DeepGlobe Road Segmentation.

Provides:
1. get_train_transform() -- standard spatial + color augmentations for training
2. apply_focus_mim(image, ...) -- simulates tree canopy / shadow occlusion
   by zeroing random patches on the IMAGE ONLY (never the mask)
"""

import random
import numpy as np
import albumentations as A
import cv2


def get_train_transform():
    """
    Returns an albumentations.Compose with training augmentations.
    Applied jointly to image and mask to keep spatial transforms aligned.
    """
    return A.Compose([
        A.RandomCrop(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
    ])


def get_val_transform():
    """
    Returns an albumentations.Compose for validation.
    Only center-crops to 512x512 -- no random augmentations.
    """
    return A.Compose([
        A.CenterCrop(512, 512),
    ])


def apply_focus_mim(image, occlusion_ratio=0.3, patch_size=32):
    """
    Simulates tree canopy / shadow occlusion by zeroing out random
    square patches in the IMAGE ONLY.

    This function deliberately has NO mask parameter -- it is
    structurally impossible to accidentally occlude the label.

    Args:
        image (np.ndarray): Input image, shape (H, W, C) or (H, W).
            Modified IN-PLACE and also returned.
        occlusion_ratio (float): Fraction of image area to occlude.
            Default 0.3 = 30% of the image is masked out.
        patch_size (int): Side length of each square patch. Default 32.

    Returns:
        np.ndarray: The image with random patches zeroed out.
    """
    image = image.copy()  # Don't modify the original
    height, width = image.shape[:2]

    # Calculate number of patches to zero out
    num_patches = int((height * width * occlusion_ratio) / (patch_size ** 2))

    for _ in range(num_patches):
        # Random top-left corner, ensuring patch fits entirely in bounds
        top = random.randint(0, max(0, height - patch_size))
        left = random.randint(0, max(0, width - patch_size))
        image[top:top + patch_size, left:left + patch_size] = 0

    return image


def apply_contrast_reduction(img_bgr, mask_gray, alpha_range=(0.80, 0.90)):
    """
    Blends road pixels towards their local surrounding background colors.
    Uses downscaled cv2.inpaint to quickly estimate the background color 
    beneath the road, then alpha-blends it with the original road.
    """
    h, w = img_bgr.shape[:2]
    
    # Randomly select an alpha value in the specified range
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    
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


if __name__ == '__main__':
    """Visual test: load one sample, apply occlusion, save comparison."""
    import os
    import sys
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

    from src.data.dataset import RoadDataset

    train_dir = os.path.join(project_root, 'Dataset', 'deep', 'train')
    ds = RoadDataset(train_dir)

    # Get raw image (before tensor conversion) -- load directly
    sat_path = ds.sat_files[0]
    image = cv2.imread(sat_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply occlusion
    occluded = apply_focus_mim(image, occlusion_ratio=0.3, patch_size=32)

    # Save comparison
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'occlusion_test.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].set_title('Original Satellite Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(occluded)
    axes[1].set_title('After Focus-MIM Occlusion (30%)', fontsize=14)
    axes[1].axis('off')

    plt.suptitle('Occlusion Simulation (apply_focus_mim)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Original image shape: {image.shape}")
    print(f"Occluded image shape: {occluded.shape}")
    print(f"Patches zeroed: {int((image.shape[0] * image.shape[1] * 0.3) / (32**2))}")
    print(f"Saved to: {save_path}")
