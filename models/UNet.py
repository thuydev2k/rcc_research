import torch.nn as nn
import torch

from models.FiLM import FiLM

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
    def __init__(self, in_channels, out_channels, n_clinical):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

        self.n_clinical = n_clinical
        if n_clinical > 0:
            self.film = FiLM(out_channels, n_clinical)

    def forward(self, x, clinical_data=None):
        skip_out = self.double_conv(x)

        if self.n_clinical > 0 and clinical_data is not None:
            skip_out = self.film(skip_out, clinical_data)

        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_classes=1, out_classes=4, n_clinical=0, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.n_clinical = n_clinical
        self.up_sample_mode = up_sample_mode

        # Downsampling Path
        self.down_conv1 = DownBlock(in_classes, 64, n_clinical)
        self.down_conv2 = DownBlock(64, 128, n_clinical)
        self.down_conv3 = DownBlock(128, 256, n_clinical)
        self.down_conv4 = DownBlock(256, 512, n_clinical)

        # Bottleneck(Downsampling path - Upsampling path, connecting path)
        self.double_conv = DoubleConv(512, 1024)
        if n_clinical > 0:
            self.film_bottleneck = FiLM(1024, n_clinical)

        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, clinical_data=None):
        x, skip1_out = self.down_conv1(x, clinical_data)
        x, skip2_out = self.down_conv2(x, clinical_data)
        x, skip3_out = self.down_conv3(x, clinical_data)
        x, skip4_out = self.down_conv4(x, clinical_data)
        
        x = self.double_conv(x)
        if self.n_clinical > 0 and clinical_data is not None:
            x = self.film_bottleneck(x, clinical_data)

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        
        x = self.conv_last(x)

        return x