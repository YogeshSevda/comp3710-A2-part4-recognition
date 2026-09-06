"""
UNet for OASIS brain MRI segmentation (Task 2).

Shape, in plain terms:
  Encoder (contracting path): repeatedly halves spatial size while doubling
    channel depth -- learns WHAT structures are present at increasingly
    abstract levels.
  Decoder (expanding path): mirrors the encoder, growing spatial size back
    up while halving channels -- learns WHERE those structures are.
  Skip connections: at each resolution level, the encoder's feature map is
    concatenated onto the decoder's corresponding feature map. Without this,
    the decoder only has the heavily-compressed bottleneck to work from and
    produces blurry, imprecise boundaries -- skip connections hand back the
    fine spatial detail the encoder saw at full resolution.

Output: (NUM_CLASSES, H, W) raw logits -- apply softmax externally if you
need probabilities (the Dice loss function handles this internally).
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Two 3x3 convs + ReLU, the basic repeated unit of UNet."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = conv_block(in_channels, c)       # 128 -> 128
        self.enc2 = conv_block(c, c * 2)              # 64  -> 64
        self.enc3 = conv_block(c * 2, c * 4)          # 32  -> 32
        self.enc4 = conv_block(c * 4, c * 8)          # 16  -> 16
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(c * 8, c * 16)   # 8 -> 8

        # Decoder (upsample + concat skip + conv_block)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)  # c*16 because concatenated with enc4

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)  # 1x1 conv -> per-class logits

    def forward(self, x):
        # Encoder, saving each level's output for the skip connections
        e1 = self.enc1(x)              # (c,    128, 128)
        e2 = self.enc2(self.pool(e1))  # (2c,   64,  64)
        e3 = self.enc3(self.pool(e2))  # (4c,   32,  32)
        e4 = self.enc4(self.pool(e3))  # (8c,   16,  16)

        b = self.bottleneck(self.pool(e4))  # (16c, 8, 8)

        # Decoder: upsample, concat matching encoder level, conv_block
        d4 = self.up4(b)                         # (8c, 16, 16)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # skip connection

        d3 = self.up3(d4)                        # (4c, 32, 32)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)                        # (2c, 64, 64)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                        # (c, 128, 128)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # (num_classes, 128, 128) raw logits


if __name__ == "__main__":
    # Quick shape sanity check
    model = UNet(in_channels=1, num_classes=4)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # expect (2, 4, 128, 128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")