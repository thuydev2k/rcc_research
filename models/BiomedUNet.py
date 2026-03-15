import torch.nn as nn
import torch
from models.biomed_encoder import BiomedCLIPEncoder
from models.FiLM import FiLM
import torch.nn.functional as F

class PyramidFeatures(nn.Module):
    def __init__(self, base_dim=128, n_clinical=0):
        super().__init__()

        self.n_clinical = n_clinical

        self.conv1 = nn.Sequential(
            nn.Conv2d(base_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        self.relu = nn.ReLU(inplace=True)

        if n_clinical > 0:
            self.film1 = FiLM(128, n_clinical)
            self.film2 = FiLM(256, n_clinical)
            self.film3 = FiLM(512, n_clinical)
            self.film4 = FiLM(512, n_clinical)

    def forward(self, x, clinical_data=None):
        f1 = self.conv1(x)
        if self.n_clinical > 0 and clinical_data is not None:
           f1 = self.film1(f1, clinical_data)
        f1 = self.relu(f1)

        f2 = self.conv2(f1)
        if self.n_clinical > 0 and clinical_data is not None:
           f2 = self.film2(f2, clinical_data)
        f2 = self.relu(f2)

        f3 = self.conv3(f2)
        if self.n_clinical > 0 and clinical_data is not None:
           f3 = self.film3(f3, clinical_data)
        f3 = self.relu(f3)

        f4 = self.conv4(f3)
        if self.n_clinical > 0 and clinical_data is not None:
           f4 = self.film4(f4, clinical_data)
        f4 = self.relu(f4)

        return f1, f2, f3, f4

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
        x = self.conv(x)
        return x
    
class BiomedTransUNet(nn.Module):
    def __init__(self, out_classes=4, embed_dim=128, n_clinical=0):
        super().__init__()

        self.encoder = BiomedCLIPEncoder(embed_dim)
        self.pyramid = PyramidFeatures(embed_dim, n_clinical)

        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x, clinical_data=None):
        x = self.encoder(x)

        f1, f2, f3, f4 = self.pyramid(x, clinical_data)

        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)

        d_final = self.final_up(d3)

        out = self.out_conv(d_final)

        return out
