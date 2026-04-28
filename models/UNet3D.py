import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = DoubleConv3D(in_channels, out_channels)
        self.down_sample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.double_conv(x)
        x = self.down_sample(skip)
        return x, skip


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up_sample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.double_conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up_sample(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_classes=4, base_channels=16):
        super(UNet3D, self).__init__()

        self.down1 = DownBlock3D(in_channels, base_channels)
        self.down2 = DownBlock3D(base_channels, base_channels * 2)
        self.down3 = DownBlock3D(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock3D(base_channels * 4, base_channels * 8)

        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        self.up4 = UpBlock3D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock3D(base_channels * 2, base_channels, base_channels)

        self.final_conv = nn.Conv3d(base_channels, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return self.final_conv(x)