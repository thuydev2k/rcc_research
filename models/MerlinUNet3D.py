import torch
import torch.nn as nn
import torch.nn.functional as F


class MerlinImageEncoder3D(nn.Module):
    def __init__(self, merlin_model, freeze=True):
        super().__init__()

        # Merlin wrapper -> MerlinArchitecture -> ImageEncoder -> I3ResNet
        self.encoder = merlin_model.model.encode_image.i3_resnet

        # Don't use classifier / contrastive head for segmentation
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, 1, D, H, W]
        return:
            skip1:      [B, 64,   D,    H/2,  W/2]
            skip2:      [B, 256,  D/2,  H/4,  W/4]
            skip3:      [B, 512,  D/4,  H/8,  W/8]
            skip4:      [B, 1024, D/8,  H/16, W/16]
            bottleneck: [B, 2048, D/16, H/32, W/32]
        """

        # Merlin conv1 expects 3 input channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        skip1 = x

        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        skip2 = x

        x = self.encoder.layer2(x)
        skip3 = x

        x = self.encoder.layer3(x)
        skip4 = x

        x = self.encoder.layer4(x)
        bottleneck = x

        return skip1, skip2, skip3, skip4, bottleneck


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv3D(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MerlinUNet3D(nn.Module):
    def __init__(
        self,
        merlin_model,
        out_classes=4,
        freeze_encoder=True,
        decoder_channels=(1024, 512, 256, 128),
    ):
        super().__init__()

        self.encoder = MerlinImageEncoder3D(
            merlin_model=merlin_model,
            freeze=freeze_encoder,
        )

        # Merlin encoder channels
        c1 = 64
        c2 = 256
        c3 = 512
        c4 = 1024
        c5 = 2048

        d4, d3, d2, d1 = decoder_channels

        self.up4 = UpBlock3D(c5, c4, d4)
        self.up3 = UpBlock3D(d4, c3, d3)
        self.up2 = UpBlock3D(d3, c2, d2)
        self.up1 = UpBlock3D(d2, c1, d1)

        self.final_conv = nn.Conv3d(d1, out_classes, kernel_size=1)

    def forward(self, x):
        """
        x:      [B, 1, D, H, W]
        logits: [B, 4, D, H, W]
        """

        input_size = x.shape[-3:]

        skip1, skip2, skip3, skip4, bottleneck = self.encoder(x)

        x = self.up4(bottleneck, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        logits = self.final_conv(x)

        # Because Merlin conv1/maxpool downsample strongly,
        # resize logits back to original input size.
        if logits.shape[-3:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="trilinear",
                align_corners=False,
            )

        return logits