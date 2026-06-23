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
    CT-EHR cross-attention.

    image_feat:
        [B, C, H, W]

    clinical_vector:
        [B, clinical_embed_dim]

    output:
        [B, C, H, W]
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
            nn.Linear(clinical_embed_dim, num_clinical_tokens * image_dim),
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

        # Start close to image-only behavior.
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


class MultiScaleSideAwareTumorCystContextHead(nn.Module):
    """
    Multi-scale side-aware tumor/cyst classifier.

    Uses:
        skip1      [B, 128, 224, 224]
        skip2      [B, 256, 112, 112]
        skip3      [B, 512, 56, 56]
        skip4      [B, 1024, 28, 28]
        bottleneck [B, 512, 14, 14]

    Output:
        side_logits [B, 4]

    Logit order:
        0 = image-left tumor presence
        1 = image-left cyst presence
        2 = image-right tumor presence
        3 = image-right cyst presence
    """

    def __init__(
        self,
        scale_channels=(128, 256, 512, 1024, 512),
        scale_embed_dim: int = 128,
        fused_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_scales = len(scale_channels)

        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(ch),
                nn.Linear(ch, scale_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for ch in scale_channels
        ])

        concat_dim = self.num_scales * scale_embed_dim

        self.region_fuser = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared classifier for left/right side.
        # For each side, output:
        #   0 = tumor
        #   1 = cyst
        self.side_classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _pool_global_left_right(self, feat):
        B, C, H, W = feat.shape
        mid = W // 2

        global_feat = self.pool(feat).flatten(1)
        left_feat = self.pool(feat[:, :, :, :mid]).flatten(1)
        right_feat = self.pool(feat[:, :, :, mid:]).flatten(1)

        return global_feat, left_feat, right_feat

    def _project_and_concat(self, pooled_features):
        projected = []

        for feat, projector in zip(pooled_features, self.scale_projectors):
            projected.append(projector(feat))

        return torch.cat(projected, dim=1)

    def forward(self, bottleneck, skip1, skip2, skip3, skip4):
        features = [skip1, skip2, skip3, skip4, bottleneck]

        global_feats = []
        left_feats = []
        right_feats = []

        for feat in features:
            g, l, r = self._pool_global_left_right(feat)
            global_feats.append(g)
            left_feats.append(l)
            right_feats.append(r)

        global_concat = self._project_and_concat(global_feats)
        left_concat = self._project_and_concat(left_feats)
        right_concat = self._project_and_concat(right_feats)

        global_feat = self.region_fuser(global_concat)  # [B, 512]
        left_feat = self.region_fuser(left_concat)      # [B, 512]
        right_feat = self.region_fuser(right_concat)    # [B, 512]

        left_logits = self.side_classifier(left_feat)    # [B, 2] tumor, cyst
        right_logits = self.side_classifier(right_feat)  # [B, 2] tumor, cyst

        side_logits = torch.cat(
            [
                left_logits[:, 0:1],   # left tumor
                left_logits[:, 1:2],   # left cyst
                right_logits[:, 0:1],  # right tumor
                right_logits[:, 1:2],  # right cyst
            ],
            dim=1,
        )  # [B, 4]

        return side_logits, global_feat, left_feat, right_feat


class SemanticTumorCystContextTokenGenerator(nn.Module):
    """
    Generate 5 semantic context tokens from:
        global feature
        image-left feature
        image-right feature
        detached side tumor/cyst probabilities

    Input:
        global_feat         [B, 512]
        left_feat           [B, 512]
        right_feat          [B, 512]
        side_context_values [B, 4]

    Output:
        context_tokens [B, 5, context_dim]

    Token meaning:
        0 = global token
        1 = left tumor token
        2 = left cyst token
        3 = right tumor token
        4 = right cyst token
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_side_values: int = 4,
        num_context_tokens: int = 5,
        context_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.num_context_tokens = num_context_tokens
        self.context_dim = context_dim

        input_dim = feature_dim * 3 + num_side_values

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_context_tokens * context_dim),
        )

    def forward(self, global_feat, left_feat, right_feat, side_context_values):
        B = global_feat.shape[0]

        context_input = torch.cat(
            [
                global_feat,
                left_feat,
                right_feat,
                side_context_values,
            ],
            dim=1,
        )

        context_tokens = self.mlp(context_input)
        context_tokens = context_tokens.view(
            B,
            self.num_context_tokens,
            self.context_dim,
        )

        return context_tokens


class SideMaskedContextCrossAttention2D(nn.Module):
    """
    Side-masked context fusion.

    Context tokens:
        token 0 = global
        token 1 = left tumor
        token 2 = right tumor

    Left half attends to:
        global + left tumor

    Right half attends to:
        global + right tumor
    """

    def __init__(
        self,
        image_dim: int,
        context_dim: int = 128,
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
        self.context_norm = nn.LayerNorm(context_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
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

    def _attend_half(self, image_half, context_subset):
        """
        image_half:
            [B, C, H, W_half]

        context_subset:
            [B, T_subset, D]
        """

        B, C, H, W_half = image_half.shape

        image_tokens = image_half.flatten(2).transpose(1, 2)

        query = self.image_norm(image_tokens)
        key_value = self.context_norm(context_subset)

        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        image_tokens = image_tokens + self.attn_scale * self.attn_dropout(attn_out)

        ffn_out = self.ffn(self.ffn_norm(image_tokens))
        image_tokens = image_tokens + self.ffn_scale * ffn_out

        out = image_tokens.transpose(1, 2).reshape(B, C, H, W_half)

        return out

    def forward(self, image_feat, context_tokens):
        """
        image_feat:
            [B, C, H, W]

        context_tokens:
            [B, 3, D]
        """

        B, C, H, W = image_feat.shape
        mid = W // 2

        left_feat = image_feat[:, :, :, :mid]
        right_feat = image_feat[:, :, :, mid:]

        # Token indices:
        # 0 = global
        # 1 = left tumor
        # 2 = left cyst
        # 3 = right tumor
        # 4 = right cyst
        left_context = context_tokens[:, [0, 1, 2], :]
        right_context = context_tokens[:, [0, 3, 4], :]

        left_out = self._attend_half(left_feat, left_context)
        right_out = self._attend_half(right_feat, right_context)

        out = torch.cat([left_out, right_out], dim=3)

        return out
    

class BiomedCLIPUNetCTEHRAttention(nn.Module):
    """
    Side-aware context-token-guided BiomedUNet.

    Architecture:
        CT slice
        -> U-Net encoder
        -> BiomedCLIP feature
        -> BiomedCLIP + U-Net bottleneck fusion
        -> CT-EHR cross-attention
        -> side-aware classifier
        -> semantic context tokens
        -> context-token cross-attention at bottleneck and skips
        -> U-Net decoder
        -> kidney/tumor/cyst mask
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
        side_hidden_dim: int = 128,
        context_dim: int = 128,
    ):
        super().__init__()


        # ---------- U-Net encoder ----------
        self.inc = DoubleConv(in_channels, 64)

        self.down_conv1 = DownBlock(64, 128)
        self.down_conv2 = DownBlock(128, 256)
        self.down_conv3 = DownBlock(256, 512)
        self.down_conv4 = DownBlock(512, 1024)

        # ---------- BiomedCLIP ----------
        self.biomed_encoder = BiomedCLIPEncoder(embed_dim=biomed_embed_dim)

        # U-Net bottleneck 1024 + BiomedCLIP 512 -> 512
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

        # ---------- Side-aware context branch ----------
        # ---------- Side-aware tumor context branch ----------
        self.side_context_head = MultiScaleSideAwareTumorCystContextHead(
            scale_channels=(128, 256, 512, 1024, 512),
            scale_embed_dim=128,
            fused_dim=512,
            hidden_dim=side_hidden_dim,
            dropout=0.10,
        )

        self.context_token_generator = SemanticTumorCystContextTokenGenerator(
            feature_dim=512,
            num_side_values=4,
            num_context_tokens=5,
            context_dim=context_dim,
            hidden_dim=256,
            dropout=0.10,
        )

        self.context_fuse_bottleneck = SideMaskedContextCrossAttention2D(
            image_dim=512,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=0.10,
        )

        self.context_fuse_skip4 = SideMaskedContextCrossAttention2D(
            image_dim=1024,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=0.10,
        )

        self.context_fuse_skip3 = SideMaskedContextCrossAttention2D(
            image_dim=512,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=0.10,
        )

        self.context_fuse_skip2 = SideMaskedContextCrossAttention2D(
            image_dim=256,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=0.10,
        )

        self.context_fuse_skip1 = SideMaskedContextCrossAttention2D(
            image_dim=128,
            context_dim=context_dim,
            num_heads=num_heads,
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

        # ---------- Bottleneck fusion ----------
        x = torch.cat([x, biomed_feat], dim=1)
        x = self.fusion(x)

        # x: [B, 512, 14, 14]

        # ---------- CT-EHR attention ----------
        if clinical_data is not None:
            clinical_vector = self.clinical_encoder(clinical_data)
            x = self.ct_ehr_attention(x, clinical_vector)


        side_logits, global_feat, left_feat, right_feat = self.side_context_head(
            bottleneck=x,
            skip1=skip1,
            skip2=skip2,
            skip3=skip3,
            skip4=skip4,
        )

        side_context_values = torch.sigmoid(side_logits.detach())

        context_tokens = self.context_token_generator(
            global_feat=global_feat,
            left_feat=left_feat,
            right_feat=right_feat,
            side_context_values=side_context_values,
        )

        x = self.context_fuse_bottleneck(x, context_tokens)

        # ---------- Side-masked context fusion ----------
        skip4 = self.context_fuse_skip4(skip4, context_tokens)
        skip3 = self.context_fuse_skip3(skip3, context_tokens)
        skip2 = self.context_fuse_skip2(skip2, context_tokens)
        skip1 = self.context_fuse_skip1(skip1, context_tokens)

        # ---------- Decoder ----------
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        seg_logits = self.out_conv(x)

        return {
            "out": seg_logits,
            "side_logits": side_logits,
            "side_context_values": side_context_values,
            "context_tokens": context_tokens,
        }
