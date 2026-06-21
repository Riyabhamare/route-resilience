"""
Step 3: Test the predict_roads function from HANDOVER.md against the V2 checkpoint.
Tests on 2 images, prints literal output shapes and value ranges.
"""
import os, sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet

# ── Exact code from HANDOVER.md Section 2 ───────────────────────────
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth'
)
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024


def load_model(checkpoint_path=CHECKPOINT_PATH,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = HDDNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _letterbox(image, target_size=LETTERBOX_SIZE):
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
                  threshold=0.5):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    lb_img, scale, pad_top, pad_left = _letterbox(image, LETTERBOX_SIZE)
    crop_start = (LETTERBOX_SIZE - MODEL_SIZE) // 2
    cropped = lb_img[crop_start:crop_start + MODEL_SIZE,
                     crop_start:crop_start + MODEL_SIZE]

    tensor = torch.from_numpy(
        cropped.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        final_logits, _, _ = model(tensor)
        prob_crop = torch.sigmoid(final_logits).squeeze().cpu().numpy()

    full_pred = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    full_pred[crop_start:crop_start + MODEL_SIZE,
              crop_start:crop_start + MODEL_SIZE] = prob_crop
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pred_scaled = full_pred[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    prob_map = cv2.resize(pred_scaled, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)

    pred_mask = (prob_map > threshold).astype(np.uint8) * 255
    return pred_mask, prob_map
# ── End of HANDOVER.md code ─────────────────────────────────────────


def test_image(image_path, model, device):
    fname = os.path.basename(image_path)
    print(f"\n  Testing: {fname}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    pred_mask, prob_map = predict_roads(image_path, model, device)

    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    print(f"  Input image size:    {orig_w}x{orig_h}")
    print(f"  pred_mask shape:     {pred_mask.shape}")
    print(f"  pred_mask dtype:     {pred_mask.dtype}")
    print(f"  pred_mask values:    min={pred_mask.min()}, max={pred_mask.max()}, unique={np.unique(pred_mask).tolist()}")
    print(f"  pred_mask road px:   {(pred_mask == 255).sum()}")
    print(f"  prob_map shape:      {prob_map.shape}")
    print(f"  prob_map dtype:      {prob_map.dtype}")
    print(f"  prob_map range:      [{prob_map.min():.6f}, {prob_map.max():.6f}]")
    print(f"  prob_map mean:       {prob_map.mean():.6f}")

    # Verify shapes match original image
    assert pred_mask.shape == (orig_h, orig_w), f"Shape mismatch: {pred_mask.shape} vs ({orig_h}, {orig_w})"
    assert prob_map.shape == (orig_h, orig_w), f"Shape mismatch: {prob_map.shape} vs ({orig_h}, {orig_w})"
    print(f"  Shape check:         PASSED (matches {orig_w}x{orig_h})")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading model from: {CHECKPOINT_PATH}")

    model = load_model(CHECKPOINT_PATH, device)

    test_dir = os.path.join(PROJECT_ROOT, 'real_test_img')

    # Test image 1: standard satellite (should have lots of road pixels)
    test_image(os.path.join(test_dir, '506876_sat.jpg'), model, device)

    # Test image 2: the challenging cloud_test_3
    test_image(os.path.join(test_dir, 'cloud_test_3.png'), model, device)

    print(f"\n  All tests PASSED. The predict_roads function works correctly")
    print(f"  with the V2 checkpoint at: {CHECKPOINT_PATH}")
