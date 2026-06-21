"""
Section 3 -- Environment Verification Script
Verifies CUDA availability, GPU details, and all required packages.
"""

import sys
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# --- CUDA / PyTorch verification ---
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM (GB): {props.total_memory / 1e9:.2f}")
        print(f"CUDA version (runtime): {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("\n[FAIL] CUDA is NOT available!")
        print("Action: Uninstall torch/torchvision/torchaudio and reinstall with cu121/cu124 index-url.")
        sys.exit(1)
except ImportError:
    print("[FAIL] PyTorch not installed!")
    sys.exit(1)

print()

# --- Verify all required packages ---
packages = [
    ("segmentation_models_pytorch", "smp"),
    ("albumentations", None),
    ("cv2", "opencv-python"),
    ("numpy", None),
    ("matplotlib", None),
    ("skimage", "scikit-image"),
    ("rasterio", None),
    ("tqdm", None),
    ("tensorboard", None),
    ("torchvision", None),
    ("torchaudio", None),
]

print("Package verification:")
all_ok = True
for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        name = display_name or import_name
        print(f"  [OK] {name}: {version}")
    except ImportError:
        name = display_name or import_name
        print(f"  [FAIL] {name}: NOT INSTALLED")
        all_ok = False

print()
if all_ok:
    print("[OK] All required packages are installed.")
else:
    print("[FAIL] Some packages are missing. Install them before proceeding.")
    sys.exit(1)

# --- Quick tensor test on GPU ---
print()
print("GPU tensor test:")
x = torch.randn(2, 3, 512, 512, device='cuda')
print(f"  Created tensor on GPU: shape={x.shape}, device={x.device}")
del x
torch.cuda.empty_cache()
print("  [OK] GPU tensor creation and cleanup successful.")
print()
print("=" * 50)
print("  Environment verification PASSED")
print("=" * 50)
