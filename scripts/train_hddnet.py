"""
Section 11 -- HDDNet Training Script (Local RTX 4050 Version)

STAGE 1 (local smoke test):
    python scripts/train_hddnet.py
    - Small subset: 300 train / 60 val, 3 epochs
    - Goal: confirm no crashes, no NaN, loss decreases

STAGE 2 (full local training):
    python scripts/train_hddnet.py --full
    - Full train/ dataset (85/15 split), 32 epochs
    - CosineAnnealingLR (T_max=32)
    - Checkpoint saved EVERY epoch + resume-from-latest on restart
    - combined_loss with per-branch auxiliary losses (Fix A)
    - Focus-MIM occlusion on ~60% of training images (Fix B)
    - Per-branch health logging each epoch (Fix D)

Resume after interruption:
    python scripts/train_hddnet.py --full
    (automatically detects and loads models/hddnet/hddnet_latest.pth)

No internet required once started -- encoder weights are cached from
the baseline training run (resnet34 ImageNet weights).
"""

import os
import sys
import time
import random
import argparse

# Force unbuffered stdout so terminal shows live progress
os.environ['PYTHONUNBUFFERED'] = '1'

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.dataset import RoadDataset
from src.data.augmentations import get_train_transform, get_val_transform, apply_focus_mim, apply_contrast_reduction
from src.models.hddnet import HDDNet
from src.models.losses import combined_loss, aux_branch_loss
from src.training.metrics import compute_iou
from tqdm import tqdm

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset', 'deep', 'train')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models', 'hddnet')
NUM_WORKERS = 2


def flush_print(*args, **kwargs):
    """Print with immediate flush so terminal sees output in real time."""
    print(*args, **kwargs, flush=True)


def train_one_epoch_hddnet(model, loader, optimizer, loss_fn, device, epoch,
                           occlusion_prob=0.6):
    """
    Train HDDNet for one epoch with per-branch auxiliary losses.

    Fix A: Loss = 0.5*combined_loss(final) + 0.25*aux_branch(main) + 0.25*aux_branch(occ)
           combined_loss (10-iter clDice) only on final output.
           aux_branch_loss (3-iter clDice) on each branch -- provides
           independent gradient + topology signal at ~30% of full clDice cost.
    """
    model.train()
    total_loss = 0.0
    total_loss_final = 0.0
    total_loss_main = 0.0
    total_loss_occ = 0.0
    sum_main_max = 0.0
    sum_occ_max = 0.0
    sum_main_mean = 0.0
    sum_occ_mean = 0.0
    num_batches = 0
    num_contrast_applied = 0
    num_contrast_eligible = 0

    pbar = tqdm(
        loader, desc=f"  Epoch {epoch:>2d} Train",
        leave=False, ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, loss={postfix}]'
    )
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            # Fix B: Apply occlusion per-image with probability occlusion_prob
            # Some images stay clean so decoder_main can learn visible roads
            processed = []
            for i in range(images.shape[0]):
                img_np = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
                img_np = (img_np * 255).astype(np.uint8)
                
                # Fix C: Synthetic contrast reduction to target extreme low-contrast failures
                num_contrast_eligible += 1
                if random.random() < 0.30:
                    mask_np = masks[i].squeeze().numpy()
                    if mask_np.max() <= 1.0:
                        mask_np = (mask_np * 255).astype(np.uint8)
                    else:
                        mask_np = mask_np.astype(np.uint8)
                    img_np = apply_contrast_reduction(img_np, mask_np)
                    num_contrast_applied += 1

                if random.random() < occlusion_prob:
                    img_np = apply_focus_mim(img_np, occlusion_ratio=0.3, patch_size=32)
                    
                img_np = img_np.astype(np.float32) / 255.0
                processed.append(torch.from_numpy(img_np).permute(2, 0, 1))
            images = torch.stack(processed)

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # HDDNet returns (final, main, occlusion)
            final_out, main_out, occ_out = model(images)

            # Fix A: Per-branch auxiliary losses
            # Full clDice on final; light 3-iter clDice on each branch
            loss_final = loss_fn(final_out, masks)
            loss_main = aux_branch_loss(main_out, masks)
            loss_occ = aux_branch_loss(occ_out, masks)
            loss = 0.5 * loss_final + 0.25 * loss_main + 0.25 * loss_occ

            # NaN check
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at batch {batch_idx}/{len(loader)}. Training stopped."
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_final += loss_final.item()
            total_loss_main += loss_main.item()
            total_loss_occ += loss_occ.item()
            num_batches += 1

            # Fix D: Track per-branch max AND mean probabilities
            with torch.no_grad():
                main_prob = torch.sigmoid(main_out)
                occ_prob = torch.sigmoid(occ_out)
                sum_main_max += main_prob.max().item()
                sum_occ_max += occ_prob.max().item()
                sum_main_mean += main_prob.mean().item()
                sum_occ_mean += occ_prob.mean().item()

            pbar.set_postfix_str(f'{loss.item():.4f}')

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM at batch {batch_idx}/{len(loader)}. "
                f"batch_size={loader.batch_size}. "
                f"Try reducing to {max(1, loader.batch_size // 2)}."
            )

    pbar.close()
    n = max(num_batches, 1)
    contrast_rate = num_contrast_applied / max(num_contrast_eligible, 1)
    return {
        'loss': total_loss / n,
        'loss_final': total_loss_final / n,
        'loss_main': total_loss_main / n,
        'loss_occ': total_loss_occ / n,
        'avg_main_max_prob': sum_main_max / n,
        'avg_occ_max_prob': sum_occ_max / n,
        'avg_main_mean_prob': sum_main_mean / n,
        'avg_occ_mean_prob': sum_occ_mean / n,
        'contrast_applied': num_contrast_applied,
        'contrast_eligible': num_contrast_eligible,
        'contrast_rate': contrast_rate,
    }


def validate_hddnet(model, loader, loss_fn, device, epoch):
    """Validate HDDNet with tqdm progress bar -- NO occlusion.
    Returns dict with per-branch loss/probability stats."""
    model.eval()
    total_loss = 0.0
    total_loss_main = 0.0
    total_loss_occ = 0.0
    total_iou = 0.0
    sum_main_max = 0.0
    sum_occ_max = 0.0
    sum_main_mean = 0.0
    sum_occ_mean = 0.0
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

            final_out, main_out, occ_out = model(images)
            loss = loss_fn(final_out, masks)
            iou = compute_iou(final_out, masks)

            total_loss += loss.item()
            total_loss_main += aux_branch_loss(main_out, masks).item()
            total_loss_occ += aux_branch_loss(occ_out, masks).item()
            total_iou += iou * images.shape[0]
            main_prob = torch.sigmoid(main_out)
            occ_prob = torch.sigmoid(occ_out)
            sum_main_max += main_prob.max().item()
            sum_occ_max += occ_prob.max().item()
            sum_main_mean += main_prob.mean().item()
            sum_occ_mean += occ_prob.mean().item()
            num_batches += 1
            num_samples += images.shape[0]
            pbar.set_postfix_str(f'loss={loss.item():.4f}')

    pbar.close()
    n = max(num_batches, 1)
    return {
        'loss': total_loss / n,
        'loss_main': total_loss_main / n,
        'loss_occ': total_loss_occ / n,
        'iou': total_iou / max(num_samples, 1),
        'avg_main_max_prob': sum_main_max / n,
        'avg_occ_max_prob': sum_occ_max / n,
        'avg_main_mean_prob': sum_main_mean / n,
        'avg_occ_mean_prob': sum_occ_mean / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train HDDNet')
    parser.add_argument('--full', action='store_true',
                        help='Run Stage 2 (full local training, 32 epochs)')
    args = parser.parse_args()

    # Stage configuration
    if args.full:
        stage = 2
        train_subset = None  # Use all data
        val_subset = None
        num_epochs = 32
        batch_size = 4
    else:
        stage = 1
        train_subset = 300
        val_subset = 60
        num_epochs = 3
        batch_size = 4

    lr = 1e-4

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flush_print(f"{'=' * 65}")
    flush_print(f"  HDDNet Training -- Stage {stage} ({'FULL LOCAL' if stage == 2 else 'SMOKE TEST'})")
    flush_print(f"{'=' * 65}")
    flush_print(f"  Device: {device}")
    if device == 'cuda':
        flush_print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        flush_print(f"  VRAM: {vram_gb:.2f} GB")
    else:
        flush_print("  [WARN] Running on CPU -- training will be very slow!")
    flush_print(f"  Epochs: {num_epochs}")
    flush_print(f"  Batch size: {batch_size}")
    flush_print(f"  Learning rate: {lr}")
    flush_print(f"  Loss: 0.5*combined(final) + 0.25*aux_branch(main) + 0.25*aux_branch(occ)")
    flush_print(f"  Final term: 0.4 dice + 0.4 clDice(iter=10) + 0.2 BCE")
    flush_print(f"  Branch terms: 0.4 BCE + 0.3 Dice + 0.3 clDice(iter=3)")
    flush_print(f"  Occlusion: Focus-MIM 30%, applied to 60% of images (Fix B)")
    flush_print(f"  Merge: torch.maximum(main, occ)")
    flush_print(f"  Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    flush_print()

    # Verify data directory
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Training data directory not found: {DATA_DIR}")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ====================================================================
    # DATA LOADING
    # ====================================================================
    flush_print("Loading datasets...")
    full_train_dataset = RoadDataset(DATA_DIR, transform=get_train_transform())
    full_val_dataset = RoadDataset(DATA_DIR, transform=get_val_transform())
    total_available = len(full_train_dataset)
    flush_print(f"  Total images in data dir: {total_available}")

    if train_subset is not None:
        train_indices = list(range(0, train_subset))
        val_indices = list(range(train_subset, train_subset + val_subset))
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)
    else:
        # Full training: use 85/15 split
        split_idx = int(0.85 * total_available)
        train_indices = list(range(0, split_idx))
        val_indices = list(range(split_idx, total_available))
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)

    flush_print(f"  Train: {len(train_dataset)} images")
    flush_print(f"  Val:   {len(val_dataset)} images")
    flush_print()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    flush_print(f"  Train batches/epoch: {num_train_batches}")
    flush_print(f"  Val batches/epoch:   {num_val_batches}")
    flush_print()

    # ====================================================================
    # MODEL & OPTIMIZER
    # ====================================================================
    flush_print("Initializing HDDNet...")
    model = HDDNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    flush_print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = combined_loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ====================================================================
    # RESUME FROM CHECKPOINT
    # ====================================================================
    start_epoch = 1
    best_iou = 0.0
    best_epoch = -1
    latest_ckpt = os.path.join(CHECKPOINT_DIR, 'hddnet_latest.pth')

    if os.path.exists(latest_ckpt) and stage == 2:
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

        if start_epoch > num_epochs:
            flush_print(f"\n  Training already complete ({num_epochs} epochs). Nothing to do.")
            flush_print(f"  Best IoU: {best_iou:.4f} (epoch {best_epoch})")
            flush_print(f"  Best model: {os.path.join(CHECKPOINT_DIR, 'hddnet_best.pth')}")
            return
    flush_print()

    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    header = (f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | "
              f"{'Val IoU':>8} | {'LR':>10} | {'Time':>6} | "
              f"{'MnMaxP':>7} | {'MnMeanP':>7} | {'OcMaxP':>7} | {'OcMeanP':>7}")
    sep = (f"  {'-'*5}-+-{'-'*11}-+-{'-'*11}-+-"
           f"{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-"
           f"{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    remaining = num_epochs - start_epoch + 1
    flush_print(f"Training epochs {start_epoch}-{num_epochs} ({remaining} remaining)...")
    flush_print(f"  Checkpoints saved EVERY epoch to: {CHECKPOINT_DIR}")
    flush_print(f"  Resume with: python scripts/train_hddnet.py --full")
    flush_print(f"{'=' * 100}")
    flush_print(header)
    flush_print(sep)

    epoch1_time = None

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        try:
            # Train with mixed occlusion (Fix B: 60% of images)
            train_stats = train_one_epoch_hddnet(
                model, train_loader, optimizer, loss_fn, device,
                epoch=epoch, occlusion_prob=0.6
            )

            # Validate without occlusion
            val_stats = validate_hddnet(
                model, val_loader, loss_fn, device, epoch=epoch
            )

            scheduler.step()

        except RuntimeError as e:
            if 'NaN loss' in str(e):
                flush_print(f"\n  [FATAL] {e}")
                flush_print(f"  Training aborted at epoch {epoch}.")
                sys.exit(1)
            elif 'OOM' in str(e) or 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                flush_print(f"\n  [FATAL] CUDA OOM at epoch {epoch}.")
                flush_print(f"  Last checkpoint: {latest_ckpt}")
                flush_print(f"  Resume with: python scripts/train_hddnet.py --full")
                sys.exit(1)
            else:
                raise

        elapsed = time.time() - t0
        train_loss = train_stats['loss']
        val_loss = val_stats['loss']
        val_iou = val_stats['iou']

        # Track best
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch

        # Print epoch summary with branch health (Fix D)
        flush_print(
            f"  {epoch:>5} | {train_loss:>11.6f} | {val_loss:>11.6f} | "
            f"{val_iou:>8.4f} | {current_lr:>10.2e} | {elapsed:>4.0f}s | "
            f"{val_stats['avg_main_max_prob']:>7.4f} | {val_stats['avg_main_mean_prob']:>7.4f} | "
            f"{val_stats['avg_occ_max_prob']:>7.4f} | {val_stats['avg_occ_mean_prob']:>7.4f}"
        )

        # Branch health warning (Fix D)
        ratio = val_stats['avg_main_max_prob'] / max(val_stats['avg_occ_max_prob'], 1e-10)
        if ratio < 0.2:
            flush_print(
                f"          [WARN] decoder_main maxP/decoder_occ maxP = {ratio:.3f} "
                f"-- main branch may be collapsing!"
            )

        # After epoch 1: print time estimate
        if epoch == start_epoch:
            epoch1_time = elapsed
            epochs_remaining = num_epochs - epoch
            est_total_remaining = epoch1_time * epochs_remaining
            hours = int(est_total_remaining // 3600)
            mins = int((est_total_remaining % 3600) // 60)
            flush_print(
                f"  {'':>5}   "
                f"^ Epoch 1 took {elapsed:.0f}s. "
                f"Estimated remaining: {hours}h {mins}m for {epochs_remaining} epochs."
            )
            # Print per-branch loss breakdown for first epoch
            flush_print(
                f"  {'':>5}   "
                f"  Train breakdown: L_final={train_stats['loss_final']:.4f}, "
                f"L_main={train_stats['loss_main']:.4f}, "
                f"L_occ={train_stats['loss_occ']:.4f}"
            )
            flush_print(
                f"  {'':>5}   "
                f"  Contrast reduction: {train_stats['contrast_applied']}/{train_stats['contrast_eligible']} images "
                f"({train_stats['contrast_rate']*100:.1f}% actual vs 30% target)"
            )

        # Save checkpoint EVERY epoch
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
            'train_stats': train_stats,
            'val_stats': val_stats,
        }

        # Save latest (always overwritten -- used for resume)
        torch.save(checkpoint, latest_ckpt)

        # Save numbered checkpoint (permanent)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'hddnet_epoch{epoch}.pth')
        torch.save(checkpoint, ckpt_path)

        # Save best model separately
        if epoch == best_epoch:
            best_path = os.path.join(CHECKPOINT_DIR, 'hddnet_best.pth')
            torch.save(checkpoint, best_path)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    flush_print(f"{'=' * 75}")
    flush_print()
    flush_print(f"  Training complete!")
    flush_print(f"  Best validation IoU: {best_iou:.4f} (epoch {best_epoch})")
    flush_print(f"  Best model: {os.path.join(CHECKPOINT_DIR, 'hddnet_best.pth')}")
    flush_print(f"  Latest:     {latest_ckpt}")
    flush_print()
    flush_print(f"  Next step: run evaluation against baseline:")
    flush_print(f"    python scripts/evaluate.py --hddnet-ckpt models/hddnet/hddnet_best.pth")
    flush_print()


if __name__ == '__main__':
    main()
