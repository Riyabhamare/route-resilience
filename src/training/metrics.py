"""
Evaluation metrics for road segmentation.

Provides:
- compute_iou: Intersection-over-Union from raw logits + binary target
"""

import torch


def compute_iou(pred, target, threshold=0.5):
    """
    Compute Intersection-over-Union (IoU / Jaccard Index).

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W], values in {0, 1}
        threshold (float): Threshold applied after sigmoid to binarize prediction

    Returns:
        float: IoU score as a Python float
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()

        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum().item()
        union = pred_flat.sum().item() + target_flat.sum().item() - intersection

        iou = intersection / (union + 1e-6)

    return iou


def compute_iou_per_image(pred, target, threshold=0.5):
    """
    Compute IoU per image in a batch.

    Args:
        pred (Tensor): Raw logits, shape [B, 1, H, W]
        target (Tensor): Binary targets, shape [B, 1, H, W]
        threshold (float): Binarization threshold

    Returns:
        list[float]: IoU score for each image in the batch
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        batch_size = pred.shape[0]
        ious = []
        for i in range(batch_size):
            p = pred_binary[i].view(-1)
            t = target[i].view(-1)
            intersection = (p * t).sum().item()
            union = p.sum().item() + t.sum().item() - intersection
            ious.append(intersection / (union + 1e-6))
    return ious


if __name__ == '__main__':
    """Self-test with dummy tensors."""
    pred = torch.randn(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()

    iou = compute_iou(pred, target)
    print(f"Batch IoU: {iou:.4f}")

    per_image = compute_iou_per_image(pred, target)
    print(f"Per-image IoUs: {[f'{v:.4f}' for v in per_image]}")

    assert 0.0 <= iou <= 1.0, f"IoU out of range: {iou}"
    print("[OK] Metrics test passed.")
