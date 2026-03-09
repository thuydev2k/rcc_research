import torch.nn as nn
import torch
import torch.nn.functional as F

from models.biomed_encoder import BiomedCLIPEncoder

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
    def __init__(self, out_classes=4, embed_dim=128):
        super().__init__()

        self.encoder = BiomedCLIPEncoder(embed_dim)

        self.down1 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), # Optional: highly recommend keeping BatchNorms!
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)

        f2 = self.down1(f2)
        f3 = self.down2(f3)
        f4 = self.down3(f4)

        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)

        x = self.final_up(d3)
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=True)
        out = self.out_conv(x)

        return out
