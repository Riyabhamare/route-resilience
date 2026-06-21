"""
Test the UPDATED predict_roads function from HANDOVER.md (with ensemble support).
Tests THREE call patterns:
  (a) Old signature: predict_roads(image_path, model) — no ensemble
  (b) ensemble=True but baseline_model=None — expect clear error
  (c) ensemble=True with baseline_model — the intended Max-Ensemble use
"""
import os, sys, traceback
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model

# ── Exact code from HANDOVER.md Section 2 (updated) ─────────────────
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth'
)
BASELINE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth'
)
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024


def load_model(checkpoint_path=CHECKPOINT_PATH,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Loads the V2 HDDNet checkpoint."""
    model = HDDNet().to(device)
    # weights_only=False is required: checkpoint stores optimizer state, epoch,
    # val_iou, and val_stats alongside model weights — not just a state_dict.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_baseline_model(checkpoint_path=BASELINE_CHECKPOINT_PATH,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Loads the Baseline U-Net checkpoint for ensemble inference."""
    model = get_baseline_model().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _letterbox(image, target_size=LETTERBOX_SIZE):
    """Pad image to square with black borders, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    lb = cv2.copyMakeBorder(
        resized,
        pad_top, target_size - new_h - pad_top,
        pad_left, target_size - new_w - pad_left,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return lb, scale, pad_top, pad_left


def predict_roads(image_path, model,
                  device='cuda' if torch.cuda.is_available() else 'cpu',
                  threshold=0.5,
                  baseline_model=None, ensemble=False):
    """
    Run road extraction on a single satellite image.

    Args:
        image_path: Path to the input image (any format cv2 can read).
        model: HDDNet model returned by load_model().
        device: 'cuda' or 'cpu'.
        threshold: Probability threshold for binary mask (default 0.5).
        baseline_model: Optional Baseline U-Net from load_baseline_model().
                        Required when ensemble=True.
        ensemble: If True, returns the pixel-wise MAXIMUM of HDDNet and
                  Baseline probability maps (Max-Ensemble). Requires
                  baseline_model to be provided.

    Returns:
        pred_mask: Binary mask (uint8, 0 or 255) at original image resolution.
        prob_map:  Raw probability map (float32, 0.0-1.0) at original resolution.
                   When ensemble=True, this is the Max-Ensemble map.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Letterbox to 1024x1024, then center-crop to 512x512
    lb_img, scale, pad_top, pad_left = _letterbox(image, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE,
                     crop_start:crop_start + MODEL_SIZE]

    # To tensor
    tensor = torch.from_numpy(
        cropped.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference — HDDNet
    with torch.no_grad():
        final_logits, _, _ = model(tensor)
        hdd_prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    # Helper: map a 512x512 crop back to original image dimensions
    def _to_original(prob_crop):
        full_pred = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
        full_pred[crop_start:crop_start + MODEL_SIZE,
                  crop_start:crop_start + MODEL_SIZE] = prob_crop
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        pred_scaled = full_pred[pad_top:pad_top + new_h,
                                pad_left:pad_left + new_w]
        return cv2.resize(pred_scaled, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)

    prob_map = _to_original(hdd_prob_crop)

    # Max-Ensemble: pixel-wise maximum of HDDNet and Baseline
    if ensemble:
        if baseline_model is None:
            raise ValueError("ensemble=True requires baseline_model")
        with torch.no_grad():
            base_logits = baseline_model(tensor)
            base_prob_crop = torch.sigmoid(base_logits).squeeze().cpu().numpy()
        base_prob_map = _to_original(base_prob_crop)
        prob_map = np.maximum(prob_map, base_prob_map)

    # Threshold
    pred_mask = (prob_map > threshold).astype(np.uint8) * 255
    return pred_mask, prob_map
# ── End of HANDOVER.md code ─────────────────────────────────────────


def print_result(pred_mask, prob_map, orig_h, orig_w):
    """Print literal output details."""
    print(f"  pred_mask shape:     {pred_mask.shape}")
    print(f"  pred_mask dtype:     {pred_mask.dtype}")
    print(f"  pred_mask values:    min={pred_mask.min()}, max={pred_mask.max()}, "
          f"unique={np.unique(pred_mask).tolist()}")
    print(f"  pred_mask road px:   {(pred_mask == 255).sum()}")
    print(f"  prob_map shape:      {prob_map.shape}")
    print(f"  prob_map dtype:      {prob_map.dtype}")
    print(f"  prob_map range:      [{prob_map.min():.6f}, {prob_map.max():.6f}]")
    print(f"  prob_map mean:       {prob_map.mean():.6f}")
    # Verify shapes match original
    assert pred_mask.shape == (orig_h, orig_w), \
        f"Shape mismatch: {pred_mask.shape} vs ({orig_h}, {orig_w})"
    assert prob_map.shape == (orig_h, orig_w), \
        f"Shape mismatch: {prob_map.shape} vs ({orig_h}, {orig_w})"
    print(f"  Shape check:         PASSED (matches {orig_w}x{orig_h})")
    # Value range check
    if prob_map.max() <= 1.0 and prob_map.min() >= 0.0:
        print(f"  Value range check:   PASSED (prob_map in [0.0, 1.0])")
    else:
        print(f"  Value range check:   FAILED (prob_map outside [0.0, 1.0]!)")
    mask_vals = set(np.unique(pred_mask).tolist())
    if mask_vals <= {0, 255}:
        print(f"  Mask values check:   PASSED (only 0 and 255)")
    else:
        print(f"  Mask values check:   FAILED (unexpected values: {mask_vals})")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_image = os.path.join(PROJECT_ROOT, 'real_test_img', '506876_sat.jpg')
    img = cv2.imread(test_image)
    orig_h, orig_w = img.shape[:2]

    print("=" * 70)
    print("  HANDOVER FUNCTION TEST — Three Call Patterns")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  Test image: 506876_sat.jpg ({orig_w}x{orig_h})")
    print(f"  HDDNet:     {CHECKPOINT_PATH}")
    print(f"  Baseline:   {BASELINE_CHECKPOINT_PATH}")

    # Load models
    print("\n  Loading HDDNet V2...", flush=True)
    hddnet = load_model(device=device)
    print("  Loading Baseline U-Net...", flush=True)
    baseline = load_baseline_model(device=device)

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (a): Old signature — no ensemble params
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (a): predict_roads(image_path, model)")
    print("  Expected: HDDNet-only output, backward compatible")
    print(f"{'=' * 70}")
    pred_mask_a, prob_map_a = predict_roads(test_image, hddnet, device)
    print_result(pred_mask_a, prob_map_a, orig_h, orig_w)

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (b): ensemble=True but baseline_model=None
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (b): predict_roads(image_path, model, ensemble=True)")
    print("  baseline_model=None (omitted)")
    print("  Expected: clear error")
    print(f"{'=' * 70}")
    try:
        pred_mask_b, prob_map_b = predict_roads(
            test_image, hddnet, device, ensemble=True)
        print("  UNEXPECTED: No error raised! Got output:")
        print_result(pred_mask_b, prob_map_b, orig_h, orig_w)
    except Exception as e:
        print(f"  Exception type:  {type(e).__name__}")
        print(f"  Exception msg:   {e}")
        print(f"  Behavior:        RAISES clear error as expected")

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (c): ensemble=True with baseline_model provided
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (c): predict_roads(image_path, model,")
    print("               baseline_model=baseline, ensemble=True)")
    print("  Expected: Max-Ensemble output")
    print(f"{'=' * 70}")
    pred_mask_c, prob_map_c = predict_roads(
        test_image, hddnet, device,
        baseline_model=baseline, ensemble=True)
    print_result(pred_mask_c, prob_map_c, orig_h, orig_w)

    # ══════════════════════════════════════════════════════════════════
    #  CROSS-CHECK: ensemble should have >= pixels of either alone
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  CROSS-CHECK: Ensemble vs HDDNet-only pixel counts")
    print(f"{'=' * 70}")
    hdd_px = int((pred_mask_a == 255).sum())
    ens_px = int((pred_mask_c == 255).sum())
    print(f"  HDDNet-only road px:    {hdd_px}")
    print(f"  Max-Ensemble road px:   {ens_px}")
    print(f"  Ensemble gain:          +{ens_px - hdd_px} pixels")
    if ens_px >= hdd_px:
        print(f"  Consistency:            PASSED (ensemble >= HDDNet)")
    else:
        print(f"  Consistency:            FAILED (ensemble < HDDNet — unexpected)")

    print(f"\n{'=' * 70}")
    print("  ALL TESTS COMPLETE")
    print(f"{'=' * 70}")
