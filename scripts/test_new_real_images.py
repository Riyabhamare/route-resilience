import os, sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.hddnet import HDDNet
from src.models.baseline_unet import get_baseline_model

# ── Copied from HANDOVER.md ────────────────────────
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2', 'hddnet_best.pth')
BASELINE_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'models', 'baseline_full', 'baseline_full_best.pth')
MODEL_SIZE = 512
LETTERBOX_SIZE = 1024
TTA_SCALES = [0.8, 1.0, 1.2]

def load_model(device):
    model = HDDNet().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def load_baseline_model(device):
    model = get_baseline_model().to(device)
    ckpt = torch.load(BASELINE_CHECKPOINT_PATH, map_location=device, weights_only=False)
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
    lb = cv2.copyMakeBorder(resized, pad_top, target_size - new_h - pad_top,
                            pad_left, target_size - new_w - pad_left,
                            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return lb, scale, pad_top, pad_left

def _run_single_pass(model, img_rgb, device, tta_scale=1.0, flip=False):
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
    cropped = lb_img[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        logits = out[0] if isinstance(out, tuple) else out
        prob_crop = torch.sigmoid(logits).squeeze().cpu().numpy()
    new_h = int(round(sh * lb_scale))
    new_w = int(round(sw * lb_scale))
    canvas = np.zeros((LETTERBOX_SIZE, LETTERBOX_SIZE), dtype=np.float32)
    canvas[crop_start:crop_start+MODEL_SIZE, crop_start:crop_start+MODEL_SIZE] = prob_crop
    region = canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    prob_scaled = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if flip:
        prob_scaled = np.flip(prob_scaled, axis=1).copy()
    if tta_scale != 1.0:
        return cv2.resize(prob_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_scaled

def predict_roads(image_path, model, device, threshold=0.5, baseline_model=None, ensemble=False, tta=False):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    models_to_run = [model]
    if ensemble:
        models_to_run.append(baseline_model)
    scales = TTA_SCALES if tta else [1.0]
    flips = [False, True] if tta else [False]
    prob_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    for m in models_to_run:
        for s in scales:
            for f in flips:
                prob_map = np.maximum(prob_map, _run_single_pass(m, image, device, s, f))
    pred_mask = (prob_map > threshold).astype(np.uint8) * 255
    return pred_mask, prob_map

# ── Evaluation Script ─────────────────────────────
def overlay_mask(img_rgb, mask, color=(255, 0, 0), alpha=0.5):
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

if __name__ == '__main__':
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'new_real_test')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'RESULTS.txt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading models...")
    hddnet = load_model(device)
    baseline = load_baseline_model(device)

    images = ['315111_sat.jpg', '17193_sat.jpg']
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for img_name in images:
            img_path = os.path.join(PROJECT_ROOT, 'real_test_img', img_name)
            if not os.path.exists(img_path):
                print(f"Skipping {img_name}, file not found at {img_path}")
                continue
                
            print(f"\nProcessing {img_name}...")
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            f.write(f"======================================================================\n")
            f.write(f"IMAGE: {img_name}\n")
            f.write(f"======================================================================\n")
            
            results = []
            configs = [
                ("Mode 1: HDDNet-only", False, False),
                ("Mode 2: Max-Ensemble", True, False),
                ("Mode 3: TTA-only", False, True),
                ("Mode 4: Full Combined", True, True)
            ]
            
            f.write(f"{'Mode':<25} | {'Road Px':>8} | {'Max Prob':>8} | {'Mean Prob':>9}\n")
            f.write("-" * 58 + "\n")
            
            for mode_name, ens, tta in configs:
                print(f"  Running {mode_name}...")
                pred, prob = predict_roads(img_path, hddnet, device, 0.5, baseline, ens, tta)
                px_count = int((pred == 255).sum())
                max_p = float(prob.max())
                mean_p = float(prob.mean())
                results.append((mode_name, pred, px_count))
                f.write(f"{mode_name:<25} | {px_count:>8} | {max_p:>8.6f} | {mean_p:>9.6f}\n")
            
            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"Original\n{img_name}")
            axes[0].axis('off')
            
            for i, (mode_name, pred, px) in enumerate(results):
                ax = axes[i+1]
                overlaid = overlay_mask(img_rgb, pred, color=(255, 0, 0), alpha=0.6)
                ax.imshow(overlaid)
                # Shorten title to fit
                short_mode = mode_name.replace("Mode ", "M").replace(": ", "\n")
                ax.set_title(f"{short_mode}\n({px} px)")
                ax.axis('off')
                
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{img_name.split('.')[0]}_comparison.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {save_path}")
            f.write("\n")
