"""
Channel compatibility test for HDDNet before writing the full module.
Tests: encoder.out_channels -> UnetDecoder -> 1x1 Conv head
"""
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# 1. Encoder channels
encoder = smp.encoders.get_encoder('resnet34', weights='imagenet')
enc_channels = encoder.out_channels
print(f"encoder.out_channels = {list(enc_channels)}")
print(f"  len = {len(enc_channels)}")
print()

# 2. Decoder channels
dec_channels = (256, 128, 64, 32, 16)
print(f"decoder_channels = {list(dec_channels)}")
print(f"  len = {len(dec_channels)}")
print()

# 3. Compatibility check
n_blocks = len(dec_channels)
print(f"n_blocks = {n_blocks}")
print(f"Required: len(encoder_channels) = n_blocks + 1 = {n_blocks + 1}")
print(f"Actual:   len(encoder_channels) = {len(enc_channels)}")
print(f"Match: {len(enc_channels) == n_blocks + 1}")
print()

# 4. Build encoder + decoder + head and test end-to-end
x = torch.randn(2, 3, 512, 512)
features = encoder(x)

print("Encoder feature maps:")
for i, f in enumerate(features):
    print(f"  features[{i}]: {f.shape}")
print()

decoder = UnetDecoder(
    encoder_channels=enc_channels,
    decoder_channels=dec_channels,
    n_blocks=n_blocks,
)
decoder_out = decoder(features)  # takes a list, not unpacked
print(f"Decoder output: {decoder_out.shape}")

head = nn.Conv2d(16, 1, kernel_size=1)
head_out = head(decoder_out)
print(f"Head output:    {head_out.shape}")
print()

print("=" * 50)
print("COMPATIBILITY CONFIRMED")
print("=" * 50)
print(f"  encoder_channels = {list(enc_channels)}")
print(f"  decoder_channels = {list(dec_channels)}")
print(f"  Final output: {head_out.shape}")
