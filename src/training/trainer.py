"""
Training loop utilities for road segmentation models.

Provides:
- train_one_epoch: Standard training loop with tqdm progress bar
- validate: Evaluation loop with model.eval() and no_grad()
"""

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer instance
        loss_fn: Loss function (pred, target) -> scalar loss
        device: 'cuda' or 'cpu'

    Returns:
        float: Average training loss across all batches

    Raises:
        RuntimeError: If loss becomes NaN (includes batch index in message)
        RuntimeError: If CUDA runs out of memory (includes batch_size suggestion)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            # NaN check
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss detected at batch {batch_idx}/{len(loader)}. "
                    f"Training stopped. Check learning rate and input data."
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory at batch {batch_idx}/{len(loader)}. "
                f"Current batch_size={loader.batch_size}. "
                f"Try reducing batch_size to {max(1, loader.batch_size // 2)}."
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, loader, loss_fn, device):
    """
    Validate the model on validation data.

    Args:
        model: PyTorch model
        loader: DataLoader for validation data
        loss_fn: Loss function (pred, target) -> scalar loss
        device: 'cuda' or 'cpu'

    Returns:
        float: Average validation loss across all batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
