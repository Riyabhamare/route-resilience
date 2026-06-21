"""
Part 1: Training Capacity Check — extract epoch-by-epoch metrics from V2 checkpoints.
Loads only the metadata (train_loss, val_loss, val_iou) from each epoch checkpoint.
Produces: table of all 32 epochs, plot, and focus on epochs 25-32.
"""
import os, sys, torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT_DIR = os.path.join(PROJECT_ROOT, 'models', 'hddnet_archive_v2')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'capacity_check')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract metrics from all epoch checkpoints
print("=" * 75)
print("  PART 1: Training Capacity Check — V2 HDDNet (32 epochs)")
print("=" * 75)
print(f"  Checkpoint dir: {CKPT_DIR}")

epochs = []
for ep in range(1, 33):
    ckpt_path = os.path.join(CKPT_DIR, f'hddnet_epoch{ep}.pth')
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Missing epoch {ep}")
        continue
    # Load only metadata, not full model weights — use map_location='cpu'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    epochs.append({
        'epoch': ckpt.get('epoch', ep),
        'train_loss': float(ckpt.get('train_loss', float('nan'))),
        'val_loss': float(ckpt.get('val_loss', float('nan'))),
        'val_iou': float(ckpt.get('val_iou', float('nan'))),
        'best_iou': float(ckpt.get('best_iou', float('nan'))),
        'best_epoch': ckpt.get('best_epoch', '?'),
    })
    print(f"  Loaded epoch {ep} metadata", flush=True)

print(f"\n  Total epochs loaded: {len(epochs)}")

# ── Full table ────────────────────────────────────────────────────────
print(f"\n{'=' * 75}")
print("  FULL EPOCH-BY-EPOCH METRICS")
print(f"{'=' * 75}")
header = f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'Val IoU':>10} | {'Best IoU':>10} | {'Best Ep':>7}"
print(header)
print("  " + "-" * (len(header) - 2))
for e in epochs:
    print(f"  {e['epoch']:>5d} | {e['train_loss']:>11.6f} | {e['val_loss']:>11.6f} | "
          f"{e['val_iou']:>10.6f} | {e['best_iou']:>10.6f} | {e['best_epoch']:>7}")

# ── Focus on epochs 25-32 ─────────────────────────────────────────────
print(f"\n{'=' * 75}")
print("  FOCUS: Epochs 25-32 (literal numbers)")
print(f"{'=' * 75}")
print(header)
print("  " + "-" * (len(header) - 2))
for e in epochs:
    if e['epoch'] >= 25:
        print(f"  {e['epoch']:>5d} | {e['train_loss']:>11.6f} | {e['val_loss']:>11.6f} | "
              f"{e['val_iou']:>10.6f} | {e['best_iou']:>10.6f} | {e['best_epoch']:>7}")

# ── Epoch-over-epoch deltas for epochs 25-32 ──────────────────────────
print(f"\n{'=' * 75}")
print("  EPOCH-OVER-EPOCH DELTAS (25-32)")
print(f"{'=' * 75}")
late_epochs = [e for e in epochs if e['epoch'] >= 24]  # include 24 for delta calc
print(f"  {'Epoch':>5} | {'dTrain Loss':>12} | {'dVal Loss':>12} | {'dVal IoU':>12}")
print("  " + "-" * 50)
for i in range(1, len(late_epochs)):
    prev = late_epochs[i - 1]
    curr = late_epochs[i]
    if curr['epoch'] >= 25:
        dt = curr['train_loss'] - prev['train_loss']
        dv = curr['val_loss'] - prev['val_loss']
        di = curr['val_iou'] - prev['val_iou']
        print(f"  {curr['epoch']:>5d} | {dt:>+12.6f} | {dv:>+12.6f} | {di:>+12.6f}")

# ── Plot ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep_nums = [e['epoch'] for e in epochs]
    train_losses = [e['train_loss'] for e in epochs]
    val_losses = [e['val_loss'] for e in epochs]
    val_ious = [e['val_iou'] for e in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    ax1.plot(ep_nums, train_losses, 'b-o', markersize=4, label='Train Loss')
    ax1.plot(ep_nums, val_losses, 'r-o', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train & Val Loss (V2 HDDNet, 32 epochs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(25, 32, alpha=0.1, color='yellow', label='Focus region')

    # Val IoU curve
    ax2.plot(ep_nums, val_ious, 'g-o', markersize=4, label='Val IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU (V2 HDDNet, 32 epochs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(25, 32, alpha=0.1, color='yellow', label='Focus region')
    # Mark best epoch
    best_ep_idx = np.argmax(val_ious)
    ax2.axvline(x=ep_nums[best_ep_idx], color='green', linestyle='--', alpha=0.5,
                label=f'Best: epoch {ep_nums[best_ep_idx]}')
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'training_curves_v2.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")
except Exception as ex:
    print(f"\n  Plot failed: {ex}")

print(f"\n{'=' * 75}")
print("  PART 1 COMPLETE")
print(f"{'=' * 75}")
