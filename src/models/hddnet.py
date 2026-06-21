"""
HDDNet -- Dual-Decoder Segmentation Model for Occlusion-Robust Road Extraction.

Architecture:
    - Shared encoder: ResNet34 (ImageNet pretrained) via segmentation_models_pytorch
    - Two decoder branches (both smp UnetDecoder):
        - decoder_main: detects clearly visible roads
        - decoder_occlusion: reconstructs roads under occlusion
    - Each decoder feeds into its own 1x1 Conv2d head (16 -> 1 channel)
    - final_output = torch.maximum(main_output, occlusion_output)

Channel configuration (verified compatible):
    encoder.out_channels = [3, 64, 64, 128, 256, 512]  (6 values)
    decoder_channels     = (256, 128, 64, 32, 16)       (5 values)
    n_blocks = 5 = len(decoder_channels)
    len(encoder_channels) = 6 = n_blocks + 1  --> compatible
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class HDDNet(nn.Module):
    """
    Dual-decoder segmentation model for occlusion-robust road extraction.

    Args:
        encoder_name (str): Encoder backbone name. Default "resnet34".
        encoder_weights (str): Pretrained weights. Default "imagenet".
        in_channels (int): Number of input image channels. Default 3.
        decoder_channels (tuple): Channel counts for each decoder stage.
            Default (256, 128, 64, 32, 16).

    Returns (from forward):
        tuple of 3 tensors, each [B, 1, H, W]:
            - final_output: element-wise maximum of main and occlusion outputs
            - main_output: prediction from the main (visible road) decoder
            - occlusion_output: prediction from the occlusion decoder
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()

        # Shared encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels
        n_blocks = len(decoder_channels)

        # Dual decoders -- both use the same encoder features
        self.decoder_main = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
        )
        self.decoder_occlusion = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
        )

        # Segmentation heads: 1x1 Conv from last decoder channel -> 1 class
        last_decoder_ch = decoder_channels[-1]  # 16
        self.head_main = nn.Conv2d(last_decoder_ch, 1, kernel_size=1)
        self.head_occlusion = nn.Conv2d(last_decoder_ch, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through shared encoder and both decoder branches.

        Args:
            x (Tensor): Input image batch, shape [B, 3, H, W]

        Returns:
            tuple: (final_output, main_output, occlusion_output)
                Each tensor has shape [B, 1, H, W] (raw logits, no sigmoid)
        """
        # Shared encoder features (list of tensors at different scales)
        features = self.encoder(x)

        # Main decoder branch (visible roads)
        main_decoded = self.decoder_main(features)
        main_output = self.head_main(main_decoded)

        # Occlusion decoder branch (roads under canopy/shadow)
        occ_decoded = self.decoder_occlusion(features)
        occlusion_output = self.head_occlusion(occ_decoded)

        # Final output: element-wise maximum (either branch confident = road)
        final_output = torch.maximum(main_output, occlusion_output)

        return final_output, main_output, occlusion_output


if __name__ == '__main__':
    """Standalone shape test as required by Section 9."""
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Instantiate and move to GPU
    print("Instantiating HDDNet...")
    model = HDDNet().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_main_params = sum(p.numel() for p in model.decoder_main.parameters())
    decoder_occ_params = sum(p.numel() for p in model.decoder_occlusion.parameters())
    head_params = sum(p.numel() for p in model.head_main.parameters()) + \
                  sum(p.numel() for p in model.head_occlusion.parameters())

    print(f"Total parameters:     {total_params:,}")
    print(f"  Encoder (shared):   {encoder_params:,}")
    print(f"  Decoder main:       {decoder_main_params:,}")
    print(f"  Decoder occlusion:  {decoder_occ_params:,}")
    print(f"  Heads (2x):         {head_params:,}")
    print()

    # Dummy input
    x = torch.randn(2, 3, 512, 512, device=device)
    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        final_out, main_out, occ_out = model(x)

    print(f"\nOutput shapes:")
    print(f"  final_output:     {final_out.shape}")
    print(f"  main_output:      {main_out.shape}")
    print(f"  occlusion_output: {occ_out.shape}")

    # Assertions
    expected = (2, 1, 512, 512)
    assert final_out.shape == expected, f"final_output: expected {expected}, got {final_out.shape}"
    assert main_out.shape == expected, f"main_output: expected {expected}, got {main_out.shape}"
    assert occ_out.shape == expected, f"occlusion_output: expected {expected}, got {occ_out.shape}"

    print(f"\n[OK] All three outputs have correct shape {expected}.")

    # Verify final = max(main, occ)
    recomputed = torch.maximum(main_out, occ_out)
    assert torch.equal(final_out, recomputed), "final_output != max(main, occ)!"
    print("[OK] final_output == torch.maximum(main_output, occlusion_output)")

    # Cleanup
    del model, x, final_out, main_out, occ_out
    torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"  Section 9 shape test PASSED")
    print(f"{'='*50}")
