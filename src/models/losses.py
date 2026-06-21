"""
Loss functions for road segmentation.

Contains:
- dice_loss: Dice loss operating on raw logits (applies sigmoid internally)
- bce_dice_loss: Combined BCE + Dice loss for baseline model (0.5/0.5 weighting)
- soft_skel: Differentiable soft skeletonization (iterative min-pool/max-pool)
- soft_cldice_loss: Topology-preserving clDice loss (connectivity-aware)
- combined_loss: Advanced loss for HDDNet (0.4 Dice + 0.4 clDice + 0.2 BCE)

The soft_cldice implementation follows the paper:
  "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure
   Segmentation" (Shit et al., CVPR 2021)
Implemented directly here because the jocpae/clDice GitHub repo is not
pip-installable (no setup.py/pyproject.toml at repo root).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss from raw logits.

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W], values in {0, 1}
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        Tensor: Scalar loss = 1 - dice_coefficient
    """
    pred_sigmoid = torch.sigmoid(pred)

    # Flatten spatial dimensions
    pred_flat = pred_sigmoid.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def bce_dice_loss(pred, target):
    """
    Combined BCE + Dice loss with equal 0.5/0.5 weighting.
    Used for the baseline U-Net model.

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W]

    Returns:
        Tensor: Scalar combined loss
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


# =========================================================================
# Soft clDice (topology-preserving loss for tubular structures)
# =========================================================================

def soft_erode(img):
    """Soft morphological erosion via min-pooling with 3x3 kernel."""
    if len(img.shape) == 4:
        # Use negative max-pool on negated image = min-pool
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        return torch.min(p1, p2)
    else:
        raise ValueError(f"soft_erode expects 4D tensor, got shape {img.shape}")


def soft_dilate(img):
    """Soft morphological dilation via max-pooling with 3x3 kernel.

    Uses a full 3x3 kernel (all 8 neighbors + center) matching the original
    clDice reference implementation (jocpae/clDice).
    """
    if len(img.shape) == 4:
        return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    else:
        raise ValueError(f"soft_dilate expects 4D tensor, got shape {img.shape}")


def soft_open(img):
    """Soft morphological opening = erode then dilate."""
    return soft_dilate(soft_erode(img))


def soft_skel(img, num_iter=10):
    """
    Differentiable soft skeletonization.

    Iteratively applies morphological opening and subtracts from the
    eroded image to extract skeleton-like features while maintaining
    differentiability for backpropagation.

    Args:
        img (Tensor): Soft binary image, shape [B, 1, H, W], values in [0, 1]
        num_iter (int): Number of skeletonization iterations. Default 10.

    Returns:
        Tensor: Soft skeleton, same shape as input
    """
    img_eroded = img.clone()
    skel = F.relu(img_eroded - soft_open(img_eroded))

    for _ in range(num_iter):
        img_eroded = soft_erode(img_eroded)
        delta = F.relu(img_eroded - soft_open(img_eroded))
        skel = skel + F.relu(delta - skel * delta)

    return skel


def soft_cldice_loss(pred, target, num_iter=10, smooth=1e-6):
    """
    Soft clDice loss -- topology-preserving loss for tubular structures.

    Computes the clDice metric using differentiable soft skeletonization,
    which penalizes broken connectivity in predicted segmentation masks.

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W]
        num_iter (int): Skeletonization iterations. Default 10.
        smooth (float): Smoothing constant. Default 1e-6.

    Returns:
        Tensor: Scalar loss = 1 - clDice
    """
    pred_prob = torch.sigmoid(pred)

    # Compute soft skeletons
    skel_pred = soft_skel(pred_prob, num_iter)
    skel_target = soft_skel(target, num_iter)

    # Topology precision: skeleton of prediction covered by target
    tprec_num = (skel_pred * target).sum()
    tprec_den = skel_pred.sum() + smooth

    # Topology sensitivity: skeleton of target covered by prediction
    tsens_num = (skel_target * pred_prob).sum()
    tsens_den = skel_target.sum() + smooth

    tprec = tprec_num / tprec_den
    tsens = tsens_num / tsens_den

    # clDice = harmonic mean of tprec and tsens
    cl_dice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)

    return 1.0 - cl_dice


def combined_loss(pred, target):
    """
    Combined loss for HDDNet training.

    Weights:
        0.4 * dice_loss
        0.4 * soft_cldice_loss
        0.2 * BCEWithLogitsLoss

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W]

    Returns:
        Tensor: Scalar combined loss
    """
    d_loss = dice_loss(pred, target)
    cl_loss = soft_cldice_loss(pred, target)
    bce = nn.BCEWithLogitsLoss()(pred, target)

    return 0.4 * d_loss + 0.4 * cl_loss + 0.2 * bce


def aux_branch_loss(pred, target):
    """
    Lightweight auxiliary loss for per-branch supervision (Fix A).

    Provides gradient signal + basic topology awareness without the full
    10-iteration soft_skel cost. Uses num_iter=3 for clDice -- coarser
    skeletonization that still captures "roads should be connected" signal.

    Weights: 0.4 BCE + 0.3 Dice + 0.3 clDice(num_iter=3)

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W]

    Returns:
        Tensor: Scalar combined loss
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    cl_loss = soft_cldice_loss(pred, target, num_iter=3)

    return 0.4 * bce + 0.3 * d_loss + 0.3 * cl_loss


if __name__ == '__main__':
    """Standalone test with dummy tensors -- tests ALL loss functions."""
    print("Testing loss functions...")
    print()

    pred = torch.randn(2, 1, 512, 512)
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()

    # Test dice_loss
    dl = dice_loss(pred, target)
    print(f"dice_loss:        {dl.item():.6f} (finite={torch.isfinite(dl).item()})")

    # Test bce_dice_loss
    bdl = bce_dice_loss(pred, target)
    print(f"bce_dice_loss:    {bdl.item():.6f} (finite={torch.isfinite(bdl).item()})")

    # Test soft_cldice_loss
    cld = soft_cldice_loss(pred, target)
    print(f"soft_cldice_loss: {cld.item():.6f} (finite={torch.isfinite(cld).item()})")

    # Test combined_loss with individual term breakdown
    d_term = dice_loss(pred, target)
    cl_term = soft_cldice_loss(pred, target)
    bce_term = nn.BCEWithLogitsLoss()(pred, target)
    cl = combined_loss(pred, target)

    print()
    print("combined_loss breakdown:")
    print(f"  0.4 * dice_loss        = 0.4 * {d_term.item():.6f} = {0.4 * d_term.item():.6f}")
    print(f"  0.4 * soft_cldice_loss = 0.4 * {cl_term.item():.6f} = {0.4 * cl_term.item():.6f}")
    print(f"  0.2 * BCE              = 0.2 * {bce_term.item():.6f} = {0.2 * bce_term.item():.6f}")
    print(f"  combined_loss          = {cl.item():.6f}")

    # Check magnitudes are comparable (no term >10x another)
    terms = {
        'dice': 0.4 * d_term.item(),
        'cldice': 0.4 * cl_term.item(),
        'bce': 0.2 * bce_term.item(),
    }
    max_term = max(terms.values())
    min_term = min(terms.values())
    if min_term > 0 and max_term / min_term > 10:
        print(f"\n  [WARN] Magnitude imbalance! Max/min ratio = {max_term/min_term:.1f}x")
        print(f"  Largest:  {max(terms, key=terms.get)} = {max_term:.6f}")
        print(f"  Smallest: {min(terms, key=terms.get)} = {min_term:.6f}")
    else:
        print(f"\n  [OK] Term magnitudes are comparable (ratio = {max_term/max(min_term,1e-10):.1f}x)")

    assert torch.isfinite(dl), "dice_loss returned non-finite value!"
    assert torch.isfinite(bdl), "bce_dice_loss returned non-finite value!"
    assert torch.isfinite(cld), "soft_cldice_loss returned non-finite value!"
    assert torch.isfinite(cl), "combined_loss returned non-finite value!"

    print("\n[OK] All loss function tests passed.")
