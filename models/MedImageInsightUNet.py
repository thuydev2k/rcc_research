import torch.nn as nn
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import sys
from pathlib import Path

from MedImageInsights.MedImageInsight.ImageEncoder.davit_v1 import create_encoder
from MedImageInsights.MedImageInsight.Utils.Arguments import load_opt_from_config_files

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
        if skip is None:
            return self.conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)

class MedImageInsightEncoder(nn.Module):
    """
    Use MedImageInsight's DaViT image encoder as a multi-scale feature extractor.
    It returns stage features instead of the final pooled embedding.
    """

    def __init__(
        self,
        medimageinsight_repo_root,
        config_path,
        checkpoint_path,
        freeze_encoder=True,
    ):
        super().__init__()

        repo_root = Path(medimageinsight_repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        opt = load_opt_from_config_files([str(config_path)])
        image_cfg = opt["IMAGE_ENCODER"]

        self.encoder_img_size = tuple(image_cfg["IMAGE_SIZE"])   # [480, 480]
        self.out_channels = list(image_cfg["SPEC"]["DIM_EMBED"]) if "DIM_EMBED" in image_cfg["SPEC"] else list(image_cfg["SPEC"]["DIM_EMBED"])
        # fallback if config parser structure differs:
        if not self.out_channels:
            self.out_channels = [256, 512, 1024, 2048]

        image_cfg["LOAD_PRETRAINED"] = False
        image_cfg["PRETRAINED"] = ""

        self.encoder = create_encoder(image_cfg)

        ckpt = load_file(str(checkpoint_path))
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        self.encoder.from_state_dict(ckpt, pretrained_layers=["*"], verbose=False)

        # safer than depending on config parser structure
        self.out_channels = list(self.encoder.embed_dims)

        mean = torch.tensor(image_cfg["IMAGE_MEAN"]).view(1, 3, 1, 1)
        std = torch.tensor(image_cfg["IMAGE_STD"]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if freeze_encoder:
            print('log a b c freeze pretrained encoder')
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _prepare_input(self, x):
        # expected input: [B, C, H, W]
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {x.shape}")

        # Your current dataset appears to use 3 CT-window channels.
        # For grayscale experiments, repeat 1 -> 3.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"MedImageInsight encoder expects 1 or 3 channels, but got {x.shape[1]}"
            )

        return x

    def forward(self, x):
        x = self._prepare_input(x)

        original_hw = x.shape[-2:]

        # resize to MedImageInsight encoder resolution
        if x.shape[-2:] != self.encoder_img_size:
            x = F.interpolate(x, size=self.encoder_img_size, mode="bilinear", align_corners=False)

        x = (x - self.mean) / self.std

        input_size = (x.size(2), x.size(3))
        stage_features = []

        tokens = x
        for conv, block in zip(self.encoder.convs, self.encoder.blocks):
            tokens, input_size = conv(tokens, input_size)
            tokens, input_size = block(tokens, input_size)

            b, n, c = tokens.shape
            h, w = input_size

            feat = tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()
            stage_features.append(feat)

        return stage_features, original_hw

class MedImageInsightUNet(nn.Module):
    """
    Encoder: MedImageInsight DaViT
    Decoder: U-Net style decoder
    """

    def __init__(
        self,
        medimageinsight_repo_root,
        config_path,
        checkpoint_path,
        out_classes=4,
        freeze_encoder=True,
    ):
        super().__init__()

        self.encoder = MedImageInsightEncoder(
            medimageinsight_repo_root=medimageinsight_repo_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            freeze_encoder=freeze_encoder,
        )

        chs = self.encoder.out_channels   # [256, 512, 1024, 2048]

        # Decoder
        self.up4 = UpBlock(chs[3], chs[2], 512)   # 2048 + 1024 -> 512
        self.up3 = UpBlock(512, chs[1], 256)      # 512 + 512 -> 256
        self.up2 = UpBlock(256, chs[0], 128)      # 256 + 256 -> 128
        self.up1 = UpBlock(128, 0, 64)            # no skip
        self.up0 = UpBlock(64, 0, 32)             # no skip

        self.conv_last = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
        stage_features, original_hw = self.encoder(x)
        f1, f2, f3, f4 = stage_features   # strides ~ 4, 8, 16, 32

        x = self.up4(f4, f3)
        x = self.up3(x, f2)
        x = self.up2(x, f1)
        x = self.up1(x, None)
        x = self.up0(x, None)

        x = self.conv_last(x)

        if x.shape[-2:] != original_hw:
            x = F.interpolate(x, size=original_hw, mode="bilinear", align_corners=False)

        return x