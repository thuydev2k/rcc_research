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

        self.up_sample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up_sample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ClinicalContextTokenizer(nn.Module):
    """
    Convert one clinical embedding vector into multiple clinical context tokens.

    Input:
        clinical_vector: [B, clinical_embed_dim]

    Output:
        clinical_tokens: [B, num_tokens, token_dim]
    """

    def __init__(
        self,
        clinical_embed_dim: int = 64,
        token_dim: int = 128,
        num_tokens: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.tokenizer = nn.Sequential(
            nn.LayerNorm(clinical_embed_dim),
            nn.Linear(clinical_embed_dim, clinical_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clinical_embed_dim * 2, num_tokens * token_dim),
        )

    def forward(self, clinical_vector):
        B = clinical_vector.shape[0]

        tokens = self.tokenizer(clinical_vector)
        tokens = tokens.view(B, self.num_tokens, self.token_dim)

        return tokens


class CrossAttention2D(nn.Module):
    """
    Cross-attention between image feature map and clinical context tokens.

    Image feature:
        [B, C, H, W]

    Clinical tokens:
        [B, T, D]

    Logic:
        Query = image tokens
        Key   = clinical tokens
        Value = clinical tokens

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        image_dim: int,
        token_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.10,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()

        if image_dim % num_heads != 0:
            raise ValueError(
                f"image_dim={image_dim} must be divisible by num_heads={num_heads}"
            )

        self.image_norm = nn.LayerNorm(image_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=token_dim,
            vdim=token_dim,
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

        # Start close to identity to avoid damaging image features early.
        self.attn_scale = nn.Parameter(torch.ones(1) * 1e-4)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 1e-4)

    def forward(self, image_feat, clinical_tokens):
        B, C, H, W = image_feat.shape

        # [B, C, H, W] -> [B, H*W, C]
        image_tokens = image_feat.flatten(2).transpose(1, 2)

        query = self.image_norm(image_tokens)
        key_value = self.token_norm(clinical_tokens)

        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        image_tokens = image_tokens + self.attn_scale * self.attn_dropout(attn_out)

        ffn_out = self.ffn(self.ffn_norm(image_tokens))
        image_tokens = image_tokens + self.ffn_scale * ffn_out

        # [B, H*W, C] -> [B, C, H, W]
        out = image_tokens.transpose(1, 2).reshape(B, C, H, W)

        return out


class TextStyleFiLM2D(nn.Module):
    """
    FiLM modulation using pooled clinical/context tokens.

    This follows the paper's idea:
        context embedding → gamma, beta
        output = (1 + gamma) * image_feat + beta

    Here the context is clinical tokens, not text tokens.
    """

    def __init__(
        self,
        image_dim: int,
        token_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.controller = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_dim * 2),
        )

        # Identity initialization.
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, image_feat, clinical_tokens):
        B, C, H, W = image_feat.shape

        # Pool clinical tokens into one context vector.
        pooled_context = clinical_tokens.mean(dim=1)  # [B, token_dim]

        params = self.controller(pooled_context)

        gamma, beta = torch.split(params, C, dim=1)

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        out = (1.0 + gamma) * image_feat + beta

        return out


class CrossAttentionFiLMBlock2D(nn.Module):
    """
    One fusion block:
        image feature + clinical tokens
        → CrossAttention
        → FiLM
        → refined image feature
    """

    def __init__(
        self,
        image_dim: int,
        token_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.cross_attn = CrossAttention2D(
            image_dim=image_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.film = TextStyleFiLM2D(
            image_dim=image_dim,
            token_dim=token_dim,
            hidden_dim=128,
            dropout=dropout,
        )

    def forward(self, image_feat, clinical_tokens):
        x = self.cross_attn(image_feat, clinical_tokens)
        x = self.film(x, clinical_tokens)
        return x


class BiomedCLIPUNetMultiScaleClinicalFiLM(nn.Module):
    """
    BiomedCLIP + U-Net + multi-scale clinical CrossAttention + FiLM.

    This adapts the paper's:
        image feature + text embedding → CrossAtt + FiLM → decoder

    to your setting:
        image feature + clinical context tokens → CrossAtt + FiLM → decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 4,
        biomed_embed_dim: int = 512,
        clinical_embed_dim: int = 64,
        token_dim: int = 128,
        num_clinical_tokens: int = 4,
        n_numerical: int = 4,
        n_comorbidities: int = 1,
        num_heads: int = 8,
        use_bottleneck_fusion: bool = True,
    ):
        super().__init__()

        self.use_bottleneck_fusion = use_bottleneck_fusion

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

        self.clinical_tokenizer = ClinicalContextTokenizer(
            clinical_embed_dim=clinical_embed_dim,
            token_dim=token_dim,
            num_tokens=num_clinical_tokens,
            dropout=0.10,
        )

        # ---------- Multi-scale CrossAttention + FiLM ----------
        # F1: skip1 [B, 128, 224, 224]
        self.fuse_skip1 = CrossAttentionFiLMBlock2D(
            image_dim=128,
            token_dim=token_dim,
            num_heads=8,
            dropout=0.10,
        )

        # F2: skip2 [B, 256, 112, 112]
        self.fuse_skip2 = CrossAttentionFiLMBlock2D(
            image_dim=256,
            token_dim=token_dim,
            num_heads=8,
            dropout=0.10,
        )

        # F3: skip3 [B, 512, 56, 56]
        self.fuse_skip3 = CrossAttentionFiLMBlock2D(
            image_dim=512,
            token_dim=token_dim,
            num_heads=8,
            dropout=0.10,
        )

        # F4: skip4 [B, 1024, 28, 28]
        self.fuse_skip4 = CrossAttentionFiLMBlock2D(
            image_dim=1024,
            token_dim=token_dim,
            num_heads=8,
            dropout=0.10,
        )

        # Optional: bottleneck [B, 512, 14, 14]
        self.fuse_bottleneck = CrossAttentionFiLMBlock2D(
            image_dim=512,
            token_dim=token_dim,
            num_heads=8,
            dropout=0.10,
        )

        # ---------- U-Net decoder ----------
        self.up_conv4 = UpBlock(512, 1024, 256)
        self.up_conv3 = UpBlock(256, 512, 128)
        self.up_conv2 = UpBlock(128, 256, 64)
        self.up_conv1 = UpBlock(64, 128, 64)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, clinical_data=None):
        image_input = x

        # ---------- U-Net encoder ----------
        x = self.inc(x)

        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)

        # For 224x224 input:
        # skip1: [B, 128, 224, 224]
        # skip2: [B, 256, 112, 112]
        # skip3: [B, 512, 56, 56]
        # skip4: [B, 1024, 28, 28]
        # x:     [B, 1024, 14, 14]

        # ---------- BiomedCLIP feature ----------
        biomed_feat = self.biomed_encoder(image_input)

        if biomed_feat.shape[-2:] != x.shape[-2:]:
            biomed_feat = F.interpolate(
                biomed_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ---------- Bottleneck image fusion ----------
        x = torch.cat([x, biomed_feat], dim=1)
        x = self.fusion(x)

        # x: [B, 512, 14, 14]

        # ---------- Clinical CrossAttention + FiLM ----------
        if clinical_data is not None:
            clinical_vector = self.clinical_encoder(clinical_data)
            clinical_tokens = self.clinical_tokenizer(clinical_vector)

            skip1 = self.fuse_skip1(skip1, clinical_tokens)
            skip2 = self.fuse_skip2(skip2, clinical_tokens)
            skip3 = self.fuse_skip3(skip3, clinical_tokens)
            skip4 = self.fuse_skip4(skip4, clinical_tokens)

            if self.use_bottleneck_fusion:
                x = self.fuse_bottleneck(x, clinical_tokens)

        # ---------- U-Net decoder ----------
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        out = self.out_conv(x)

        return out