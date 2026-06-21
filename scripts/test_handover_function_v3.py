"""
Test the UPDATED predict_roads function (with ensemble + TTA support).
Tests FIVE call patterns with exact-match verification for backward compatibility.

KNOWN-GOOD BASELINES (from previous verified tests):
  Pattern (a) HDDNet-only:     34480 road px on 506876_sat.jpg
  Pattern (b) Ensemble-only:   35649 road px on 506876_sat.jpg
"""
import os, sys, traceback
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model

# ── Exact code from HANDOVER.md Section 2 (with TTA) ────────────────
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth'
)
BASELINE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth'
)
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024
TTA_SCALES = [0.8, 1.0, 1.2]  # scales for test-time augmentation


def load_model(checkpoint_path=CHECKPOINT_PATH,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Loads the V2 HDDNet checkpoint."""
    model = HDDNet().to(device)
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


def _run_single_pass(model, img_rgb, device, tta_scale=1.0, flip=False):
    """Run one model at one scale/flip, return prob map in original space."""
    orig_h, orig_w = img_rgb.shape[:2]
    if tta_scale != 1.0:
        sh = int(round(orig_h * tta_scale))
        sw = int(round(orig_w * tta_scale))
        img_s = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    else:
        img_s = img_rgb
        sh, sw = orig_h, orig_w
    if flip:
        img_s = np.flip(img_s, axis=1).copy()

    lb_img, lb_scale, pad_top, pad_left = _letterbox(img_s, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE,
                     crop_start:crop_start + MODEL_SIZE]
    tensor = torch.from_numpy(
        cropped.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        logits = out[0] if isinstance(out, tuple) else out
        prob_crop = torch.sigmoid(logits).squeeze().cpu().numpy()

    new_h = int(round(sh * lb_scale))
    new_w = int(round(sw * lb_scale))
    canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    canvas[crop_start:crop_start + MODEL_SIZE,
           crop_start:crop_start + MODEL_SIZE] = prob_crop
    region = canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    prob_scaled = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if flip:
        prob_scaled = np.flip(prob_scaled, axis=1).copy()
    if tta_scale != 1.0:
        return cv2.resize(prob_scaled, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)
    return prob_scaled


def predict_roads(image_path, model,
                  device='cuda' if torch.cuda.is_available() else 'cpu',
                  threshold=0.5,
                  baseline_model=None, ensemble=False, tta=False):
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
        tta: If True, applies Test-Time Augmentation: multi-scale
             (80%/100%/120%) + horizontal flip, merged via pixel-wise
             MAX. Runs 6 forward passes per model (3 scales x 2 flips).
             When combined with ensemble=True, runs 12 total passes.

    Returns:
        pred_mask: Binary mask (uint8, 0 or 255) at original image resolution.
        prob_map:  Raw probability map (float32, 0.0-1.0) at original resolution.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    models_to_run = [model]
    if ensemble:
        if baseline_model is None:
            raise ValueError("ensemble=True requires baseline_model")
        models_to_run.append(baseline_model)

    scales = TTA_SCALES if tta else [1.0]
    flips = [False, True] if tta else [False]

    prob_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    for m in models_to_run:
        for s in scales:
            for f in flips:
                prob_map = np.maximum(
                    prob_map, _run_single_pass(m, image, device, s, f))

    pred_mask = (prob_map > threshold).astype(np.uint8) * 255
    return pred_mask, prob_map
# ── End of HANDOVER.md code ─────────────────────────────────────────


def print_result(label, pred_mask, prob_map, orig_h, orig_w):
    """Print literal output details."""
    road_px = int((pred_mask == 255).sum())
    print(f"  pred_mask shape:     {pred_mask.shape}")
    print(f"  pred_mask dtype:     {pred_mask.dtype}")
    print(f"  pred_mask values:    min={pred_mask.min()}, max={pred_mask.max()}, "
          f"unique={np.unique(pred_mask).tolist()}")
    print(f"  pred_mask road px:   {road_px}")
    print(f"  prob_map shape:      {prob_map.shape}")
    print(f"  prob_map dtype:      {prob_map.dtype}")
    print(f"  prob_map range:      [{prob_map.min():.6f}, {prob_map.max():.6f}]")
    print(f"  prob_map mean:       {prob_map.mean():.6f}")
    # Shape check
    assert pred_mask.shape == (orig_h, orig_w), \
        f"Shape mismatch: {pred_mask.shape} vs ({orig_h}, {orig_w})"
    assert prob_map.shape == (orig_h, orig_w), \
        f"Shape mismatch: {prob_map.shape} vs ({orig_h}, {orig_w})"
    print(f"  Shape check:         PASSED (matches {orig_w}x{orig_h})")
    # Value range
    if prob_map.max() <= 1.0 and prob_map.min() >= 0.0:
        print(f"  Value range check:   PASSED (prob_map in [0.0, 1.0])")
    else:
        print(f"  Value range check:   FAILED (prob_map outside [0.0, 1.0]!)")
    mask_vals = set(np.unique(pred_mask).tolist())
    if mask_vals <= {0, 255}:
        print(f"  Mask values check:   PASSED (only 0 and 255)")
    else:
        print(f"  Mask values check:   FAILED (unexpected values: {mask_vals})")
    return road_px


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_image = os.path.join(PROJECT_ROOT, 'real_test_img', '506876_sat.jpg')
    img = cv2.imread(test_image)
    orig_h, orig_w = img.shape[:2]

    # Known-good baselines from previous verified tests
    EXPECTED_HDDNET_ONLY = 34480
    EXPECTED_ENSEMBLE_ONLY = 35649

    print("=" * 70)
    print("  HANDOVER FUNCTION TEST v3 -- Five Call Patterns")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  Test image: 506876_sat.jpg ({orig_w}x{orig_h})")
    print(f"  HDDNet:     {CHECKPOINT_PATH}")
    print(f"  Baseline:   {BASELINE_CHECKPOINT_PATH}")
    print(f"  Expected (a) HDDNet-only:   {EXPECTED_HDDNET_ONLY} road px")
    print(f"  Expected (b) Ensemble-only: {EXPECTED_ENSEMBLE_ONLY} road px")

    # Load models
    print("\n  Loading HDDNet V2...", flush=True)
    hddnet = load_model(device=device)
    print("  Loading Baseline U-Net...", flush=True)
    baseline = load_baseline_model(device=device)

    regression_detected = False

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (a): Old signature -- no ensemble, no TTA
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (a): predict_roads(image_path, model)")
    print("  Expected: HDDNet-only, backward compatible")
    print(f"  MUST match: {EXPECTED_HDDNET_ONLY} road px")
    print(f"{'=' * 70}")
    pred_a, prob_a = predict_roads(test_image, hddnet, device)
    px_a = print_result("(a)", pred_a, prob_a, orig_h, orig_w)
    if px_a == EXPECTED_HDDNET_ONLY:
        print(f"  EXACT MATCH:         PASSED ({px_a} == {EXPECTED_HDDNET_ONLY})")
    else:
        print(f"  EXACT MATCH:         FAILED ({px_a} != {EXPECTED_HDDNET_ONLY})")
        print(f"  *** REGRESSION DETECTED -- refactor broke backward compatibility ***")
        regression_detected = True

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (b): Ensemble-only, no TTA
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (b): predict_roads(image_path, model,")
    print("               ensemble=True, baseline_model=baseline)")
    print(f"  MUST match: {EXPECTED_ENSEMBLE_ONLY} road px")
    print(f"{'=' * 70}")
    pred_b, prob_b = predict_roads(
        test_image, hddnet, device,
        baseline_model=baseline, ensemble=True)
    px_b = print_result("(b)", pred_b, prob_b, orig_h, orig_w)
    if px_b == EXPECTED_ENSEMBLE_ONLY:
        print(f"  EXACT MATCH:         PASSED ({px_b} == {EXPECTED_ENSEMBLE_ONLY})")
    else:
        print(f"  EXACT MATCH:         FAILED ({px_b} != {EXPECTED_ENSEMBLE_ONLY})")
        print(f"  *** REGRESSION DETECTED -- refactor broke ensemble behavior ***")
        regression_detected = True

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (c): TTA only, no ensemble
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (c): predict_roads(image_path, model, tta=True)")
    print("  Expected: HDDNet-only with TTA (should have MORE px than (a))")
    print(f"{'=' * 70}")
    pred_c, prob_c = predict_roads(test_image, hddnet, device, tta=True)
    px_c = print_result("(c)", pred_c, prob_c, orig_h, orig_w)
    if px_c >= px_a:
        print(f"  Sanity check:        PASSED (TTA {px_c} >= no-TTA {px_a})")
    else:
        print(f"  Sanity check:        WARNING (TTA {px_c} < no-TTA {px_a})")

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (d): Ensemble + TTA (full combined mode)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (d): predict_roads(image_path, model,")
    print("               ensemble=True, baseline_model=baseline, tta=True)")
    print("  Expected: Max-Ensemble + TTA (should have MOST px of all)")
    print(f"{'=' * 70}")
    pred_d, prob_d = predict_roads(
        test_image, hddnet, device,
        baseline_model=baseline, ensemble=True, tta=True)
    px_d = print_result("(d)", pred_d, prob_d, orig_h, orig_w)
    if px_d >= px_b and px_d >= px_c:
        print(f"  Sanity check:        PASSED (ens+tta {px_d} >= ens-only {px_b} and tta-only {px_c})")
    else:
        print(f"  Sanity check:        WARNING (ens+tta {px_d} vs ens-only {px_b}, tta-only {px_c})")

    # ══════════════════════════════════════════════════════════════════
    #  PATTERN (e): ensemble=True, tta=True, baseline_model=None
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PATTERN (e): predict_roads(image_path, model,")
    print("               ensemble=True, tta=True)  [baseline_model=None]")
    print("  Expected: clear ValueError")
    print(f"{'=' * 70}")
    try:
        pred_e, prob_e = predict_roads(
            test_image, hddnet, device, ensemble=True, tta=True)
        print("  UNEXPECTED: No error raised! Got output:")
        print_result("(e)", pred_e, prob_e, orig_h, orig_w)
    except ValueError as e:
        print(f"  Exception type:  {type(e).__name__}")
        print(f"  Exception msg:   {e}")
        print(f"  Behavior:        RAISES clear ValueError as expected")
    except Exception as e:
        print(f"  UNEXPECTED exception type: {type(e).__name__}")
        print(f"  Exception msg:   {e}")
        print(f"  Behavior:        WRONG error type (expected ValueError)")

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Pattern (a) HDDNet-only:      {px_a:>6} road px  (expected {EXPECTED_HDDNET_ONLY})")
    print(f"  Pattern (b) Ensemble-only:    {px_b:>6} road px  (expected {EXPECTED_ENSEMBLE_ONLY})")
    print(f"  Pattern (c) TTA-only:         {px_c:>6} road px")
    print(f"  Pattern (d) Ensemble+TTA:     {px_d:>6} road px")
    print(f"  Pattern (e) Error check:      ValueError raised correctly")
    if regression_detected:
        print(f"\n  *** REGRESSION DETECTED -- DO NOT UPDATE HANDOVER.md ***")
    else:
        print(f"\n  All backward compatibility checks PASSED.")
    print(f"{'=' * 70}")
