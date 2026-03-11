import torch.nn as nn
import torch
from models.biomed_encoder import BiomedCLIPEncoder
import torch.nn.functional as F

class PyramidFeatures(nn.Module):
    def __init__(self, base_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(base_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)

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
    def __init__(self, out_classes=4, embed_dim=128):
        super().__init__()

        self.encoder = BiomedCLIPEncoder(embed_dim)
        self.pyramid = PyramidFeatures(embed_dim)

        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)

        self.final_up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final_up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  
        self.final_up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final_up4 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        self.out_conv = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)

        f1, f2, f3, f4 = self.pyramid(x)

        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)

        d4 = self.final_up1(d3)
        d5 = self.final_up2(d4)
        d6 = self.final_up3(d5)
        d7 = self.final_up4(d6)

        out = self.out_conv(d7)

        return out
