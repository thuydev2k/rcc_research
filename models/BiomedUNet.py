import torch.nn as nn
import torch
from models.biomed_encoder import BiomedCLIPEncoder
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up_sample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class BiomedTransUNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=4):
        super(BiomedTransUNet, self).__init__()

        # ---------- UNet Encoder ----------
        self.inc = DoubleConv(in_channels, 64)    # 224 x 224
        self.down1 = DownBlock(64, 128)           # 112 x 112
        self.down2 = DownBlock(128, 256)          # 56 x 56
        self.down3 = DownBlock(256, 512)          # 28 x 28
        self.down4 = DownBlock(512, 512)          # 14 x 14

        # ---------- BiomedCLIP Bottleneck ----------
        self.biomed_encoder = BiomedCLIPEncoder(embed_dim=512)

        # Fuse UNet bottleneck + BiomedCLIP feature
        self.fusion = DoubleConv(512 + 512, 512)

        # ---------- UNet Decoder ----------
        self.up1 = UpBlock(512, 512, 256)         # 14 -> 28
        self.up2 = UpBlock(256, 256, 128)         # 28 -> 56
        self.up3 = UpBlock(128, 128, 64)          # 56 -> 112
        self.up4 = UpBlock(64, 64, 64)            # 112 -> 224

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # ---------- UNet Encoder ----------
        x1 = self.inc(x)       # [B, 64, 224, 224]
        x2 = self.down1(x1)    # [B, 128, 112, 112]
        x3 = self.down2(x2)    # [B, 256, 56, 56]
        x4 = self.down3(x3)    # [B, 512, 28, 28]
        x5 = self.down4(x4)    # [B, 512, 14, 14]

        # ---------- BiomedCLIP Feature ----------
        b = self.biomed_encoder(x)  # [B, 512, 14, 14]

        if b.shape[-2:] != x5.shape[-2:]:
            b = F.interpolate(
                b,
                size=x5.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # ---------- Fusion ----------
        x = torch.cat([x5, b], dim=1)  # [B, 1024, 14, 14]
        x = self.fusion(x)             # [B, 512, 14, 14]

        # ---------- Decoder ----------
        x = self.up1(x, x4)            # [B, 256, 28, 28]
        x = self.up2(x, x3)            # [B, 128, 56, 56]
        x = self.up3(x, x2)            # [B, 64, 112, 112]
        x = self.up4(x, x1)            # [B, 64, 224, 224]

        out = self.out_conv(x)         # [B, 4, 224, 224]

        return out