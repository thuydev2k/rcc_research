import torch
import torch.nn as nn
import torch.nn.functional as F

from models.biomed_encoder import BiomedCLIPEncoder
from models.FiLM import FiLM
from models.clinical_encoder import ClinicalFeatureEncoder


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.double_conv(x)
        x = self.down_sample(skip)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up_sample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BiomedCLIPUNetFiLMClinicalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 4,
        biomed_embed_dim: int = 512,
        clinical_embed_dim: int = 64,
        n_numerical: int = 4,
        n_comorbidities: int = 1,
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down_conv1 = DownBlock(64, 128)
        self.down_conv2 = DownBlock(128, 256)
        self.down_conv3 = DownBlock(256, 512)
        self.down_conv4 = DownBlock(512, 1024)

        self.biomed_encoder = BiomedCLIPEncoder(embed_dim=biomed_embed_dim)
        self.fusion = DoubleConv(1024 + biomed_embed_dim, 512)

        self.clinical_encoder = ClinicalFeatureEncoder(
            n_numerical=n_numerical,
            n_comorbidities=n_comorbidities,
            out_dim=clinical_embed_dim,
        )
        self.film_bottleneck = FiLM(n_features=512, n_clinical=clinical_embed_dim)

        self.up_conv4 = UpBlock(512, 1024, 256)
        self.up_conv3 = UpBlock(256, 512, 128)
        self.up_conv2 = UpBlock(128, 256, 64)
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
                align_corners=False,
            )

        x = torch.cat([x, biomed_feat], dim=1)
        x = self.fusion(x)

        if clinical_data is not None:
            clinical_vector = self.clinical_encoder(clinical_data)
            x = self.film_bottleneck(x, clinical_vector)

        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        return self.out_conv(x)
