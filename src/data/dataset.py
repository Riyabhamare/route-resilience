"""
RoadDataset -- PyTorch Dataset for DeepGlobe Road Extraction.

Expects a single data_dir containing both satellite images (*_sat.jpg)
and road masks (*_mask.png) in the SAME folder. Pairs are matched by
replacing '_sat.jpg' with '_mask.png' in the filename.

Masks are binarized: pixel > 127 -> 1.0, else 0.0.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    """
    Dataset for DeepGlobe road segmentation.

    Args:
        data_dir (str): Path to directory containing *_sat.jpg and *_mask.png files.
        transform (albumentations.Compose, optional): Albumentations transform applied
            to both image and mask jointly (spatial augmentations stay aligned).
        subset_size (int, optional): If provided, only use the first N image-mask pairs.
            Useful for quick debugging / smoke tests.
    """

    def __init__(self, data_dir, transform=None, subset_size=None):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"RoadDataset: data directory not found: {data_dir}"
            )

        self.data_dir = data_dir
        self.transform = transform

        # Build index from satellite images only
        sat_pattern = os.path.join(data_dir, '*_sat.jpg')
        self.sat_files = sorted(glob.glob(sat_pattern))

        if len(self.sat_files) == 0:
            raise FileNotFoundError(
                f"RoadDataset: no *_sat.jpg files found in {data_dir}"
            )

        # Verify every sat file has a matching mask
        self.mask_files = []
        for sat_path in self.sat_files:
            mask_name = os.path.basename(sat_path).replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(data_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"RoadDataset: no matching mask found for satellite image.\n"
                    f"  Satellite file: {sat_path}\n"
                    f"  Expected mask:  {mask_path}"
                )
            self.mask_files.append(mask_path)

        # Optional subset
        if subset_size is not None:
            self.sat_files = self.sat_files[:subset_size]
            self.mask_files = self.mask_files[:subset_size]

    def __len__(self):
        return len(self.sat_files)

    def __getitem__(self, idx):
        # 1. Load satellite image (BGR -> RGB)
        sat_path = self.sat_files[idx]
        image = cv2.imread(sat_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"RoadDataset: failed to read image: {sat_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load mask as grayscale
        mask_path = self.mask_files[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"RoadDataset: failed to read mask: {mask_path}")

        # 3. Binarize mask: > 127 -> 1.0, else 0.0
        mask = (mask > 127).astype(np.float32)

        # 4. Apply albumentations transform (joint spatial augmentation)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 5. Convert image to float tensor, channel-first, normalized to [0,1]
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # 6. Convert mask to float tensor with shape [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW

        return image, mask


if __name__ == '__main__':
    """Quick self-test: load one sample from train/ and print shapes."""
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_dir = os.path.join(project_root, 'Dataset', 'deep', 'train')

    print(f"Loading RoadDataset from: {train_dir}")
    ds = RoadDataset(train_dir)
    print(f"Dataset size: {len(ds)} image-mask pairs")

    image, mask = ds[0]
    print(f"\nSample 0:")
    print(f"  Image tensor: shape={image.shape}, dtype={image.dtype}, range=[{image.min():.3f}, {image.max():.3f}]")
    print(f"  Mask tensor:  shape={mask.shape}, dtype={mask.dtype}, unique={torch.unique(mask).tolist()}")

    # Verify shapes match expected format
    assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
    assert mask.shape[0] == 1, f"Expected 1 channel mask, got {mask.shape[0]}"
    assert image.shape[1] == mask.shape[1] and image.shape[2] == mask.shape[2], \
        f"Spatial dims mismatch: image {image.shape[1:]} vs mask {mask.shape[1:]}"
    print("\n  [OK] All shape assertions passed.")
