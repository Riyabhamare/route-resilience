"""
Section 8 -- Baseline U-Net Training Script (Small Subset)

Trains the baseline U-Net on a SMALL SUBSET of the DeepGlobe train/ data
to prove the pipeline works end-to-end before committing to a full run.

IMPORTANT: Since valid/ has NO masks (only satellite images), we split
train/ into train and validation subsets ourselves (80/20 within the subset).

Usage:
    python scripts/train_baseline.py

Checkpoints saved to: models/baseline/baseline_unet_epoch{N}.pth
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import RoadDataset
from src.data.augmentations import get_train_transform, get_val_transform
from src.models.baseline_unet import get_baseline_model
from src.models.losses import bce_dice_loss
from src.training.trainer import train_one_epoch, validate
from src.training.metrics import compute_iou

# ====================================================================
# CONFIGURATION
# ====================================================================
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'baseline')

TRAIN_SUBSET_SIZE = 300   # First 300 for training
VAL_SUBSET_SIZE = 60      # Next 60 for validation (from same folder, since valid/ has no masks)
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# ====================================================================
# SETUP
# ====================================================================
def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*60}")
    print(f"  Baseline U-Net Training (Small Subset)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_gb:.2f} GB")
    else:
        print("  [WARN] Running on CPU -- training will be very slow!")
    print()

    # Verify data directory
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training data directory not found: {TRAIN_DIR}")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ====================================================================
    # DATA LOADING
    # ====================================================================
    print("Loading datasets...")

    # We create train and val from the SAME folder, using different index ranges
    # Train subset: indices 0..299 (first 300)
    # Val subset:   indices 300..359 (next 60)
    full_train_dataset = RoadDataset(TRAIN_DIR, transform=get_train_transform())
    full_val_dataset = RoadDataset(TRAIN_DIR, transform=get_val_transform())

    total_available = len(full_train_dataset)
    print(f"  Total images in train/: {total_available}")

    if total_available < TRAIN_SUBSET_SIZE + VAL_SUBSET_SIZE:
        raise ValueError(
            f"Not enough data: need {TRAIN_SUBSET_SIZE + VAL_SUBSET_SIZE} but only {total_available} available"
        )

    train_indices = list(range(0, TRAIN_SUBSET_SIZE))
    val_indices = list(range(TRAIN_SUBSET_SIZE, TRAIN_SUBSET_SIZE + VAL_SUBSET_SIZE))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    print(f"  Train subset: {len(train_dataset)} images")
    print(f"  Val subset:   {len(val_dataset)} images")
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ====================================================================
    # MODEL & OPTIMIZER
    # ====================================================================
    print("Initializing model...")
    model = get_baseline_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: Baseline U-Net (ResNet34 encoder)")
    print(f"  Parameters: {total_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = bce_dice_loss

    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    best_iou = 0.0
    best_epoch = -1

    print(f"Training for {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val IoU':>10}")
    print(f"{'-'*60}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        try:
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

            # Validate
            val_loss = validate(model, val_loader, loss_fn, device)

            # Compute validation IoU
            model.eval()
            total_iou = 0.0
            iou_count = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    batch_iou = compute_iou(outputs, masks)
                    total_iou += batch_iou * images.shape[0]
                    iou_count += images.shape[0]
            val_iou = total_iou / max(iou_count, 1)

        except RuntimeError as e:
            if 'NaN loss' in str(e):
                print(f"\n[FATAL] {e}")
                print(f"Training aborted at epoch {epoch}.")
                sys.exit(1)
            elif 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n[FATAL] CUDA OOM at epoch {epoch}.")
                print(f"  Suggestion: reduce BATCH_SIZE from {BATCH_SIZE} to {max(1, BATCH_SIZE // 2)}")
                sys.exit(1)
            else:
                raise

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {val_iou:>10.4f}  ({elapsed:.1f}s)")

        # Track best
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'best_iou': best_iou,
            'best_epoch': best_epoch,
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'baseline_unet_epoch{epoch}.pth')
        torch.save(checkpoint, ckpt_path)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print(f"{'='*60}")
    print(f"\nTraining complete!")
    print(f"  Best validation IoU: {best_iou:.4f} (epoch {best_epoch})")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")

    # Also save the best model separately for easy loading
    best_ckpt_src = os.path.join(CHECKPOINT_DIR, f'baseline_unet_epoch{best_epoch}.pth')
    best_ckpt_dst = os.path.join(CHECKPOINT_DIR, 'baseline_unet_best.pth')
    import shutil
    shutil.copy2(best_ckpt_src, best_ckpt_dst)
    print(f"  Best model copied to: {best_ckpt_dst}")


if __name__ == '__main__':
    main()
