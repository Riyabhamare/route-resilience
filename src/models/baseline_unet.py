"""
Baseline U-Net model using segmentation_models_pytorch.

Encoder: ResNet34, pretrained on ImageNet.
Single-channel output (road probability via sigmoid at inference time).
"""

import segmentation_models_pytorch as smp


def get_baseline_model():
    """
    Returns a pre-configured U-Net model for binary road segmentation.

    Architecture:
        - Encoder: ResNet34 (ImageNet pretrained)
        - Decoder: Standard U-Net decoder from smp
        - Output: 1 channel (road vs background)
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


if __name__ == '__main__':
    import torch

    model = get_baseline_model()
    print(f"Model type: {type(model).__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Quick forward pass test
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 512, 512), f"Expected (2,1,512,512) but got {y.shape}"
    print("[OK] Forward pass shape correct.")
