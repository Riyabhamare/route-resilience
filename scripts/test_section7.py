"""
Section 7 -- Combined test: baseline model forward pass + loss computation.
Verifies the full forward chain produces correct shapes and finite loss.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# Print device info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Create model
from src.models.baseline_unet import get_baseline_model
model = get_baseline_model().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {type(model).__name__}, {total_params:,} parameters")

# 2. Create dummy data
dummy_input = torch.randn(2, 3, 512, 512, device=device)
dummy_target = torch.randint(0, 2, (2, 1, 512, 512), device=device).float()

# 3. Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"\nForward pass:")
print(f"  Input shape:  {dummy_input.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == (2, 1, 512, 512), f"Expected (2,1,512,512) but got {output.shape}"
print(f"  [OK] Output shape correct.")

# 4. Compute loss
from src.models.losses import bce_dice_loss
loss = bce_dice_loss(output, dummy_target)
print(f"\nLoss computation:")
print(f"  bce_dice_loss = {loss.item():.6f}")
assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
print(f"  [OK] Loss is a single finite float.")

# 5. Backward pass (verify gradients flow)
model.train()
output = model(dummy_input)
loss = bce_dice_loss(output, dummy_target)
loss.backward()
print(f"\nBackward pass:")
print(f"  [OK] Gradients computed successfully.")

# Cleanup
del model, dummy_input, dummy_target, output, loss
torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"  Section 7 verification PASSED")
print(f"{'='*50}")
