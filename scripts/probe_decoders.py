"""
Probe HDDNet decoder branches numerically.
Prints raw sigmoid probability stats for decoder_main vs decoder_occlusion.
"""
import os, sys, glob
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import cv2
import numpy as np
from src.models.hddnet import HDDNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hddnet = HDDNet().to(device)
ckpt = torch.load('models/hddnet/hddnet_best.pth', map_location=device, weights_only=False)
hddnet.load_state_dict(ckpt['model_state_dict'])
hddnet.eval()

img_paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'real_test_img', '*.jpg')))
print("=" * 100)
print("  DECODER BRANCH NUMERIC COMPARISON (raw sigmoid probabilities)")
print("=" * 100)
print()

header = "  {:22s} | {:>10s} | {:>10s} | {:>10s} | {:>10s} | {:>12s}".format(
    "Image", "MAIN max", "MAIN mean", "OCC max", "OCC mean", "mean ratio"
)
sep = "  " + "-"*22 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*12
print(header)
print(sep)

for path in img_paths:
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))
    x = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        final, main_logits, occ_logits = hddnet(x)
        main_prob = torch.sigmoid(main_logits).squeeze().cpu().numpy()
        occ_prob = torch.sigmoid(occ_logits).squeeze().cpu().numpy()

    ratio = main_prob.mean() / max(occ_prob.mean(), 1e-10)
    name = os.path.basename(path)
    row = "  {:22s} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>12.4f}".format(
        name, main_prob.max(), main_prob.mean(), occ_prob.max(), occ_prob.mean(), ratio
    )
    print(row)

print()
print("=" * 100)
print("  INTERPRETATION:")
print("  - If mean_ratio << 1: decoder_main is near-dead, occlusion branch dominates")
print("  - If mean_ratio ~= 1: both branches contribute roughly equally")
print("  - If mean_ratio >> 1: decoder_main dominates (occlusion branch is weak)")
print("=" * 100)
