import torch.nn as nn
import torch
from models.biomed_encoder import BiomedCLIPEncoder
from models.FiLM import FiLM
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chanels, out_channels, kernel_size=3, padding=1),
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
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up_sample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class BiomedCLIPUNetFiLM(nn.Module):
    def __init__(self, in_classes=1, out_classes=4, n_clinical=17, biomed_embed_dim=512):
        super().__init__()

        self.inc = DoubleConv(in_classes, 64)
        self.down_conv1 = DownBlock(64, 128)    # 224 -> 112
        self.down_conv2 = DownBlock(128, 256)           # 112 -> 56
        self.down_conv3 = DownBlock(256, 512)          # 56 -> 28
        self.down_conv4 = DownBlock(512, 1024)          # 28 -> 14

        self.biomed_encoder = BiomedCLIPEncoder(embed_dim=biomed_embed_dim)

        self.fusion = DoubleConv(1024 + biomed_embed_dim, 512)

        self.film_bottleneck = FiLM(n_features=512, n_clinical=n_clinical)

        self.up_conv4 = UpBlock(512, 1024, 256)        # 14 -> 28
        self.up_conv3 = UpBlock(256, 512, 128)         # 28 -> 56
        self.up_conv2 = UpBlock(128, 256, 64)         # 56 -> 112
        self.up_conv1 = UpBlock(64, 128, 64)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, clinical_data=None):
        image_input = x

        x = self.inc(x)
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)

        biomed_feat = self.biomed_encoder(image_input)

        if biomed_feat.shape[-2:] != x.shape[-2:]:
            biomed_feat = F.interpolate(
                biomed_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True
            )

        x = torch.cat([x, biomed_feat], dim=1)
        x = self.fusion(x)  # [B, 512, 14, 14]

        if clinical_data is not None:
            x = self.film_bottleneck(x, clinical_data)

        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        out = self.out_conv(x)

        return out
