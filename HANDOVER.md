# HDDNet Handover Package for Member 3

## 1. Selected Checkpoint

> **IMPORTANT**: The checkpoint being delivered is:
> ```
> models/hddnet_archive_v2/hddnet_best.pth
> ```
> This is the V2 HDDNet model (epoch 32, Val IoU 0.5808). Do **NOT** use
> `models/hddnet/hddnet_best.pth` — that directory contains V3, a later
> training run that introduced a contrast-reduction augmentation experiment
> which caused a measurable regression (see Section 5 below).
>
> The folder name `hddnet_archive_v2` is misleading — it sounds like "the
> old, superseded one," but it is in fact the best-performing checkpoint
> we have. The V3 experiment in `models/hddnet/` was the failed attempt
> that came after it.

### Checkpoint Verification (performed 2026-06-21)

| Metric | Value |
|--------|-------|
| Epoch | 32 |
| Best Val IoU | 0.5808 |
| Val Loss | 0.1956 |
| File Size | 316.1 MB |
| Branch Health | Both ACTIVE, balanced (ratio ~0.98) on 7/8 test images |
| Known FAIL | `cloud_test_1_coastal.jpg` (open-water — expected, see Limitations) |

---

## Checkpoint Download

The model checkpoint files are too large for GitHub and are hosted separately:

**Google Drive folder:** https://drive.google.com/drive/folders/1gzSVD44quln2ZHEjVFXVnLBUSYe_hQIM?usp=drive_link

Download both files and place them in your local project at:
- `models/hddnet_archive_v2/hddnet_best.pth`
- `models/baseline_full/baseline_full_best.pth`

These exact paths are required for `predict_roads()` to load correctly.

## 2. Inference Function

```python
import torch
import cv2
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model


# ── Configuration ────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth'
)
BASELINE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth'
)
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024
TTA_SCALES = [0.8, 1.0, 1.2]  # scales for test-time augmentation
# ─────────────────────────────────────────────────────────────────────


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
```

### Verified Output Examples

The `predict_roads` function was tested on 2026-06-21 against the V2 checkpoint
(`models/hddnet_archive_v2/hddnet_best.pth`) on two images with different
resolutions. Literal output:

**Image 1: `506876_sat.jpg` (1024×1024, standard satellite tile)**
```
pred_mask shape:     (1024, 1024)
pred_mask dtype:     uint8
pred_mask values:    min=0, max=255, unique=[0, 255]
pred_mask road px:   34480
prob_map shape:      (1024, 1024)
prob_map dtype:      float32
prob_map range:      [0.000000, 0.999993]
prob_map mean:       0.032649
Shape check:         PASSED (matches 1024x1024)
```

**Image 2: `cloud_test_3.png` (442×441, non-square low-contrast image)**
```
pred_mask shape:     (441, 442)
pred_mask dtype:     uint8
pred_mask values:    min=0, max=255, unique=[0, 255]
pred_mask road px:   5053
prob_map shape:      (441, 442)
prob_map dtype:      float32
prob_map range:      [0.000000, 0.999962]
prob_map mean:       0.026038
Shape check:         PASSED (matches 442x441)
```

Both outputs match the input image dimensions exactly, confirming the
letterbox/un-letterbox pipeline correctly maps predictions back to original
image space regardless of input aspect ratio.

### Max-Ensemble Improvement (tested 2026-06-21)

The Max-Ensemble approach (pixel-wise maximum of Baseline U-Net and HDDNet V2)
measurably increases detected road pixel count over either model alone across
all 6 passing images, at zero retraining cost. Literal pixel counts at
threshold > 0.50:

| Image | Baseline | HDDNet V2 | Max-Ensemble | Gain over best single |
|-------|----------|-----------|--------------|----------------------|
| 506876_sat.jpg | 30216 | 34480 | 35649 | +1169 over HDDNet |
| 55062_sat.jpg | 23289 | 26546 | 27687 | +1141 over HDDNet |
| 696659_sat.jpg | 40314 | 37431 | 44855 | +4541 over Baseline |
| 78954.jpg | 20075 | 25304 | 25533 | +229 over HDDNet |
| 940563_sat.jpg | 29507 | 40403 | 41238 | +835 over HDDNet |
| cloud_test_2_village.jpg | 5950 | 9460 | 9474 | +14 over HDDNet |

**Recommendation**: Use Max-Ensemble + TTA (`ensemble=True, tta=True` in
`predict_roads`) as the final pixel-level output to hand to Member 3's
graph-based post-processing. Both are free improvements — no retraining,
just inference with both existing checkpoints.

### TTA Improvement (tested 2026-06-21)

Test-Time Augmentation (3 scales: 80/100/120% + horizontal flip, merged via
pixel-wise MAX) produces consistent gains on every image where the model has
any signal, at the cost of 6x inference time per model (12x total with
ensemble). Literal before/after numbers:

| Image | No-TTA px>0.50 | With-TTA px>0.50 | Gain |
|-------|---------------|-----------------|------|
| cloud_test_3.png (full) | 5185 | 5630 | +445 (+8.6%) |
| cloud_test_3.png upper-left (dirt road) | 1149 | 1235 | +86 (+7.5%) |
| cloud_test_3.png lower-left (street grid) | 1703 | 1885 | +182 (+10.7%) |
| cloud_test_2_village.jpg | 9474 | 10417 | +943 (+10.0%) |
| 506876_sat.jpg | 35649 | 39495 | +3846 (+10.8%) |
| 940563_sat.jpg | 41238 | 47791 | +6553 (+15.9%) |
| cloud_test_1_coastal.jpg | 0 | 0 | +0 (no signal) |

TTA does NOT help `cloud_test_1_coastal.jpg` — this confirms the coastal
failure is a true training data gap, not a confidence issue.

---

## 3. Known Limitations

1. **Coastal / Open-Water Hallucination**: The model produces near-zero
   predictions over open water and coastal regions. On `cloud_test_1_coastal.jpg`,
   both decoder branches produce maxP < 0.01. An external land/water mask
   should be applied before finalizing road network output in coastal tiles.

2. **Heavy Cloud Cover**: Dense, bright clouds completely block ground truth.
   On `cloud_test_2_village.jpg`, the model correctly detects roads in
   cloud-free areas but cannot infer roads hidden under solid cloud cover,
   resulting in fragmented networks.

3. **Low-Contrast Dirt Roads and Street Grids (UNSOLVED)**: On imagery where
   road surfaces closely match surrounding terrain color (e.g., `cloud_test_3.png`),
   the model detects roads with high confidence in some pixels (99th percentile
   ~0.99 in both target regions) but with very sparse coverage. The overall
   mean probability remains low (0.02–0.03), and many actual road pixels are
   missed. This is a limitation of the current pixel-level segmentation
   approach and remains the primary unsolved failure mode.

4. **Fragmented Road Predictions**: Both models frequently produce FRAGMENTED
   road predictions — correctly locating a road but breaking it into
   disconnected segments rather than one continuous line. This is a structural
   limitation of pixel-level segmentation and is the primary motivation for
   the graph-based gap-healing step recommended in Section 4 — Member 3's
   post-processing should specifically expect and handle this pattern, not
   treat it as a rare edge case.

---

## 4. Suggestion for Member 3: Graph-Based Gap Healing

Since pixel-level classification struggles with fragmented networks (under
clouds, in low-contrast terrain, or at tile boundaries), we recommend a
**graph-based post-processing** step:

1. **Vectorize** the binary mask into a topological graph (nodes at
   intersections, edges for road segments).
2. **Identify dangling nodes** (dead ends) that are geographically close
   and aligned in heading/orientation.
3. **Bridge gaps** using A* pathfinding over the raw `prob_map` as a cost
   surface (low probability = high cost), or apply a simple distance/angle
   heuristic to connect plausible road segments.
4. This post-processing connects fragments naturally without requiring the
   segmentation model to "see through" opaque clouds or detect invisible
   roads.

---

## 5. Failed Experiment Log: V3 Contrast-Reduction Augmentation

### What was tried
A 32-epoch fresh training run ("V3") with a synthetic contrast-reduction
augmentation applied to 30% of training images. The augmentation used
`cv2.inpaint` to estimate the background color beneath road pixels, then
alpha-blended road pixels toward that background at alpha=0.80–0.90. The
intent was to teach the model to detect roads even when they closely match
surrounding terrain.

### Why it failed
The augmentation was mechanically correct (masks were verified to remain
perfectly aligned; only road pixels in the image were modified; non-road
pixels were untouched). However, the effect was too aggressive — at
alpha 0.85, road/non-road contrast was reduced to as little as 7% of
the original, making roads genuinely invisible in many augmented samples.

The model responded by learning to **suppress responses to ambiguous
low-contrast features entirely**, rather than developing sensitivity to
faint roads. The result was a severe regression on exactly the regions
the augmentation was targeting:

| Region (cloud_test_3.png) | Metric | V2 | V3 | Change |
|---------------------------|--------|-----|-----|--------|
| Upper-Left (Dirt Road) | 99th Percentile | 0.9938 | 0.0133 | −98.7% |
| | Pixels > 0.50 | 1,133 | 0 | −100% |
| Lower-Left (Street Grid) | 99th Percentile | 0.9872 | 0.0077 | −99.2% |
| | Pixels > 0.50 | 1,633 | 0 | −100% |

Overall validation accuracy also regressed: V3 best Val IoU was 0.5707
vs V2's 0.5808 (a 1.7% drop).

### Lesson for future attempts
If revisiting contrast augmentation, use a much gentler alpha range
(e.g., 0.3–0.5 instead of 0.8–0.9) so roads remain faintly visible
rather than becoming completely indistinguishable from background.
Alternatively, consider curriculum learning: start training with no
contrast reduction and introduce it gradually in later epochs, so the
model develops strong road features first before being challenged with
harder examples.

---

## 6. Future Work: Capacity Analysis Summary

A full training capacity analysis was performed (see
`outputs/capacity_check/PART1_RESULTS.txt` and
`outputs/capacity_check/training_curves_v2.png` for raw data).

### Training was NOT finished, but gains are diminishing

The V2 model's best Val IoU was set at the very last epoch (32: 0.5808),
and the best-epoch counter advanced 4 times in the final 8 epochs
(25, 26, 29, 31, 32). Val loss continued declining (0.2016 at epoch 25
to 0.1956 at epoch 32). However, epoch-over-epoch IoU gains are clearly
shrinking — epoch 32's new best was only +0.000138 over epoch 31,
which is noise-level improvement.

### Recommended next step: Resume training 15-20 more epochs

Resuming from `hddnet_epoch32.pth` for 15-20 more epochs on the SAME
architecture is cheap and may yield a small further gain (likely
0.585-0.590 IoU, not a breakthrough). This should be tried before
considering any architecture change.

### Architecture changes are NOT warranted yet

The dominant failure modes found across 8 test images are:
- **Shadow** (8/8 images, all MODERATE) and **Low-contrast/dirt-road**
  (7/8 images) — these likely overlap substantially and may be the
  same underlying phenomenon, not two independent failure modes.
- **Fragmented-road** (8/8 images, mostly LOW severity).
- **Cloud-occlusion** (2/8) and **Coastal/water** (1/8) — rare.

These are structural limitations of pixel-level segmentation in general,
not HDDNet-specific. A different architecture (D-LinkNet, SegFormer)
would face the same issues without addressing the training data gap
(no coastal examples) that causes the coastal failure.

Only consider architecture changes if resumed training plateaus AND
the shadow/low-contrast gaps remain after more epochs.
