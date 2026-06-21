"""
PHASE 1 -- Fair Baseline Retrain on Full Dataset

Retrains the standard U-Net baseline on the EXACT same data split
and hyperparameters as HDDNet Stage 2, so the comparison is fair:

    Matching HDDNet:
      - 5292 train / 934 val (85/15 split of 6226, same indices)
      - 32 epochs, batch_size=4, AdamW lr=1e-4
      - CosineAnnealingLR T_max=32
      - Checkpoint every epoch + resume-from-latest

    Differences (by design -- this IS the baseline):
      - Standard U-Net (single decoder, not dual-decoder)
      - bce_dice_loss (not combined_loss with clDice)
      - NO Focus-MIM occlusion during training

Tag: BASELINE_FULL_DATASET
Checkpoints: models/baseline_full/  (separate from the old 300-image run)

Usage:
    python scripts/train_baseline_full.py           # Start/resume
    python scripts/train_baseline_full.py --confirm  # Skip the "are you sure" prompt

No internet required once started.
"""

import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.dataset import RoadDataset
from src.data.augmentations import get_train_transform, get_val_transform
from src.models.baseline_unet import get_baseline_model
from src.models.losses import bce_dice_loss
from src.training.metrics import compute_iou
from tqdm import tqdm

# ====================================================================
# CONFIGURATION -- must match HDDNet Stage 2 exactly
# ====================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'baseline_full')
NUM_WORKERS = 2
NUM_EPOCHS = 32
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TAG = "BASELINE_FULL_DATASET"


def flush_print(*args, **kwargs):
    """Print with immediate flush for live terminal output."""
    print(*args, **kwargs, flush=True)


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    """Train for one epoch with tqdm progress bar."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        loader, desc=f"  Epoch {epoch:>2d} Train",
        leave=False, ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, loss={postfix}]'
    )
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at batch {batch_idx}/{len(loader)}. Training stopped."
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix_str(f'{loss.item():.4f}')

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM at batch {batch_idx}/{len(loader)}. "
                f"batch_size={loader.batch_size}."
            )

    pbar.close()
    return total_loss / max(num_batches, 1)


def validate(model, loader, loss_fn, device, epoch):
    """Validate with tqdm progress bar."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    num_samples = 0

    pbar = tqdm(
        loader, desc=f"  Epoch {epoch:>2d} Val  ",
        leave=False, ncols=100,
    )
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            iou = compute_iou(outputs, masks)

            total_loss += loss.item()
            total_iou += iou * images.shape[0]
            num_batches += 1
            num_samples += images.shape[0]
            pbar.set_postfix_str(f'loss={loss.item():.4f}')

    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)
    avg_iou = total_iou / max(num_samples, 1)
    return avg_loss, avg_iou


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    flush_print(f"{'=' * 75}")
    flush_print(f"  {TAG}")
    flush_print(f"  Fair Baseline Retrain -- Matching HDDNet Stage 2 Conditions")
    flush_print(f"{'=' * 75}")
    flush_print(f"  Device: {device}")
    if device == 'cuda':
        flush_print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        flush_print(f"  VRAM: {vram_gb:.2f} GB")
    flush_print(f"  Epochs: {NUM_EPOCHS}")
    flush_print(f"  Batch size: {BATCH_SIZE}")
    flush_print(f"  LR: {LEARNING_RATE}")
    flush_print(f"  Loss: bce_dice_loss (baseline)")
    flush_print(f"  Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    flush_print(f"  Occlusion: NONE (this is the baseline)")
    flush_print()

    # ================================================================
    # DATA -- EXACT same split as HDDNet
    # ================================================================
    flush_print("Loading datasets...")
    full_train_dataset = RoadDataset(DATA_DIR, transform=get_train_transform())
    full_val_dataset = RoadDataset(DATA_DIR, transform=get_val_transform())
    total_available = len(full_train_dataset)

    # MUST match HDDNet's split exactly: int(0.85 * total)
    split_idx = int(0.85 * total_available)
    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, total_available))
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    flush_print(f"  Total images: {total_available}")
    flush_print(f"  Split index: {split_idx}")
    flush_print(f"  Train: {len(train_dataset)} images (indices 0-{split_idx - 1})")
    flush_print(f"  Val:   {len(val_dataset)} images (indices {split_idx}-{total_available - 1})")
    flush_print()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    flush_print(f"  Train batches/epoch: {len(train_loader)}")
    flush_print(f"  Val batches/epoch:   {len(val_loader)}")
    flush_print()

    # ================================================================
    # MODEL & OPTIMIZER
    # ================================================================
    flush_print("Initializing baseline U-Net (ResNet34 encoder)...")
    model = get_baseline_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    flush_print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = bce_dice_loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ================================================================
    # RESUME
    # ================================================================
    start_epoch = 1
    best_iou = 0.0
    best_epoch = -1
    latest_ckpt = os.path.join(CHECKPOINT_DIR, 'baseline_full_latest.pth')

    if os.path.exists(latest_ckpt):
        flush_print(f"\n  Resuming from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        best_epoch = checkpoint.get('best_epoch', -1)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        flush_print(f"  Resumed at epoch {start_epoch} (best IoU so far: {best_iou:.4f} @ epoch {best_epoch})")

        if start_epoch > NUM_EPOCHS:
            flush_print(f"\n  Training already complete ({NUM_EPOCHS} epochs). Nothing to do.")
            flush_print(f"  Best IoU: {best_iou:.4f} (epoch {best_epoch})")
            return
    flush_print()

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    header = f"  {'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val IoU':>10} | {'LR':>10} | {'Time':>8}"
    sep = f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}"

    remaining = NUM_EPOCHS - start_epoch + 1
    flush_print(f"Training epochs {start_epoch}-{NUM_EPOCHS} ({remaining} remaining)...")
    flush_print(f"  Checkpoints: {CHECKPOINT_DIR}")
    flush_print(f"  Resume: python scripts/train_baseline_full.py")
    flush_print(f"{'=' * 75}")
    flush_print(header)
    flush_print(sep)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
            val_loss, val_iou = validate(model, val_loader, loss_fn, device, epoch)
            scheduler.step()
        except RuntimeError as e:
            if 'NaN' in str(e):
                flush_print(f"\n  [FATAL] {e}")
                sys.exit(1)
            elif 'OOM' in str(e) or 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                flush_print(f"\n  [FATAL] CUDA OOM at epoch {epoch}.")
                flush_print(f"  Resume: python scripts/train_baseline_full.py")
                sys.exit(1)
            else:
                raise

        elapsed = time.time() - t0

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch

        flush_print(
            f"  {epoch:>5} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
            f"{val_iou:>10.4f} | {current_lr:>10.2e} | {elapsed:>6.1f}s"
        )

        # Epoch 1 time estimate
        if epoch == start_epoch:
            epochs_remaining = NUM_EPOCHS - epoch
            est_total = elapsed * epochs_remaining
            hours = int(est_total // 3600)
            mins = int((est_total % 3600) // 60)
            flush_print(
                f"  {'':>5}   "
                f"^ Epoch 1 took {elapsed:.0f}s. "
                f"Estimated remaining: {hours}h {mins}m for {epochs_remaining} epochs."
            )

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'tag': TAG,
        }

        torch.save(checkpoint, latest_ckpt)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'baseline_full_epoch{epoch}.pth')
        torch.save(checkpoint, ckpt_path)

        if epoch == best_epoch:
            best_path = os.path.join(CHECKPOINT_DIR, 'baseline_full_best.pth')
            torch.save(checkpoint, best_path)

    # ================================================================
    # SUMMARY
    # ================================================================
    flush_print(f"{'=' * 75}")
    flush_print()
    flush_print(f"  {TAG} -- Training complete!")
    flush_print(f"  Best validation IoU: {best_iou:.4f} (epoch {best_epoch})")
    flush_print(f"  Best model: {os.path.join(CHECKPOINT_DIR, 'baseline_full_best.pth')}")
    flush_print(f"  Latest:     {latest_ckpt}")
    flush_print()
    flush_print(f"  Next: run occlusion sweep comparison:")
    flush_print(f"    python scripts/occlusion_sweep.py")
    flush_print()


if __name__ == '__main__':
    main()
