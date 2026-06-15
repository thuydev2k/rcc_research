import torch
import torch.nn as nn
import torch.nn.functional as F

from models.biomed_encoder import BiomedCLIPEncoder
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


class CTEHRCrossAttention(nn.Module):
    """
    CT-EHR cross-attention module.

    Input:
        image_feat:
            [B, C, H, W]

        clinical_vector:
            [B, clinical_embed_dim]

    Output:
        fused image feature:
            [B, C, H, W]

    Logic:
        image tokens are queries.
        clinical tokens are keys and values.
    """

    def __init__(
        self,
        image_dim: int = 512,
        clinical_embed_dim: int = 64,
        num_clinical_tokens: int = 4,
        num_heads: int = 8,
        dropout: float = 0.10,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()

        if image_dim % num_heads != 0:
            raise ValueError(
                f"image_dim={image_dim} must be divisible by num_heads={num_heads}"
            )

        self.image_dim = image_dim
        self.num_clinical_tokens = num_clinical_tokens

        self.clinical_tokenizer = nn.Sequential(
            nn.LayerNorm(clinical_embed_dim),
            nn.Linear(
                clinical_embed_dim,
                num_clinical_tokens * image_dim,
            ),
        )

        self.image_norm = nn.LayerNorm(image_dim)
        self.clinical_norm = nn.LayerNorm(image_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(dropout)

        hidden_dim = int(image_dim * ffn_ratio)

        self.ffn_norm = nn.LayerNorm(image_dim)
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_dim),
            nn.Dropout(dropout),
        )

        self.attn_scale = nn.Parameter(torch.ones(1) * 1e-4)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, image_feat, clinical_vector):
        B, C, H, W = image_feat.shape

        image_tokens = image_feat.flatten(2).transpose(1, 2)

        clinical_tokens = self.clinical_tokenizer(clinical_vector)
        clinical_tokens = clinical_tokens.view(
            B,
            self.num_clinical_tokens,
            C,
        )

        query = self.image_norm(image_tokens)
        key_value = self.clinical_norm(clinical_tokens)

        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        image_tokens = image_tokens + self.attn_scale * self.attn_dropout(attn_out)

        ffn_out = self.ffn(self.ffn_norm(image_tokens))
        image_tokens = image_tokens + self.ffn_scale * ffn_out

        fused_feat = image_tokens.transpose(1, 2).reshape(B, C, H, W)

        return fused_feat


class SlicePresenceHead(nn.Module):
    """
    Auxiliary slice-level presence classifier.

    Input:
        bottleneck feature:
            [B, 512, 14, 14]

    Output:
        presence_logits:
            [B, 2]

    Channels:
        0 = has_tumor = tumor exists
        1 = has_cyst  = cyst exists
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_dim: int = 128,
        out_dim: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.classifier(x)


class BiomedCLIPUNetCTEHRAttentionPresence(nn.Module):
    """
    New Exp1:

    BiomedUNet
    + CT-EHR attention at bottleneck
    + auxiliary tumor/cyst/mass presence head
    + U-Net decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 4,
        biomed_embed_dim: int = 512,
        clinical_embed_dim: int = 64,
        n_numerical: int = 4,
        n_comorbidities: int = 1,
        num_clinical_tokens: int = 4,
        num_heads: int = 8,
        presence_hidden_dim: int = 128,
    ):
        super().__init__()

        # ---------- U-Net encoder ----------
        self.inc = DoubleConv(in_channels, 64)

        self.down_conv1 = DownBlock(64, 128)
        self.down_conv2 = DownBlock(128, 256)
        self.down_conv3 = DownBlock(256, 512)
        self.down_conv4 = DownBlock(512, 1024)

        # ---------- BiomedCLIP bottleneck ----------
        self.biomed_encoder = BiomedCLIPEncoder(embed_dim=biomed_embed_dim)

        self.fusion = DoubleConv(1024 + biomed_embed_dim, 512)

        # ---------- Clinical encoder ----------
        self.clinical_encoder = ClinicalFeatureEncoder(
            n_numerical=n_numerical,
            n_comorbidities=n_comorbidities,
            out_dim=clinical_embed_dim,
        )

        # ---------- CT-EHR attention ----------
        self.ct_ehr_attention = CTEHRCrossAttention(
            image_dim=512,
            clinical_embed_dim=clinical_embed_dim,
            num_clinical_tokens=num_clinical_tokens,
            num_heads=num_heads,
            dropout=0.10,
        )

        # ---------- Auxiliary presence head ----------
        self.presence_head = SlicePresenceHead(
            in_channels=512,
            hidden_dim=presence_hidden_dim,
            out_dim=2,
            dropout=0.10,
        )

        # ---------- U-Net decoder ----------
        self.up_conv4 = UpBlock(512, 1024, 256)
        self.up_conv3 = UpBlock(256, 512, 128)
        self.up_conv2 = UpBlock(128, 256, 64)
        self.up_conv1 = UpBlock(64, 128, 64)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, clinical_data=None, return_aux: bool = True):
        image_input = x

        # ---------- U-Net encoder ----------
        x = self.inc(x)

        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)

        # x: [B, 1024, 14, 14] for 224x224 input

        # ---------- BiomedCLIP feature ----------
        biomed_feat = self.biomed_encoder(image_input)

        if biomed_feat.shape[-2:] != x.shape[-2:]:
            biomed_feat = F.interpolate(
                biomed_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ---------- Image feature fusion ----------
        x = torch.cat([x, biomed_feat], dim=1)
        x = self.fusion(x)

        # x: [B, 512, 14, 14]

        # ---------- CT-EHR attention ----------
        if clinical_data is not None:
            clinical_vector = self.clinical_encoder(clinical_data)
            x = self.ct_ehr_attention(x, clinical_vector)

        # ---------- Auxiliary presence prediction ----------
        # This branch supervises the same bottleneck feature used by the decoder.
        presence_logits = self.presence_head(x)

        # ---------- U-Net decoder ----------
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        seg_logits = self.out_conv(x)

        if return_aux:
            return {
                "out": seg_logits,
                "presence_logits": presence_logits,
            }

        return seg_logits