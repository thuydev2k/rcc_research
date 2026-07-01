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

    image_feat:      [B, C, H, W]
    clinical_vector: [B, clinical_embed_dim]
    output:          [B, C, H, W]
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
            raise ValueError(f"image_dim={image_dim} must be divisible by num_heads={num_heads}")

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

        image_tokens = image_feat.flatten(2).transpose(1, 2)  # [B, HW, C]

        clinical_tokens = self.clinical_tokenizer(clinical_vector)
        clinical_tokens = clinical_tokens.view(B, self.num_clinical_tokens, C)

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


class MultiScaleSideAwareTumorMultiplicityContextHead(nn.Module):
    """
    Multi-scale side-aware tumor presence classifier + tumor multiplicity classifier.

    Outputs:
        side_logits:         [B, 2]
        multiplicity_logits: [B, 2, 3]

    Side order:
        0 = left_tumor
        1 = right_tumor

    Multiplicity classes:
        0 = none
        1 = single
        2 = multiple
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

        # One tumor-presence logit per side.
        self.presence_classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # One 3-class tumor-multiplicity prediction per side.
        self.multiplicity_classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
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

        global_feat = self.region_fuser(self._project_and_concat(global_feats))
        left_feat = self.region_fuser(self._project_and_concat(left_feats))
        right_feat = self.region_fuser(self._project_and_concat(right_feats))

        left_presence_logit = self.presence_classifier(left_feat)    # [B, 1]
        right_presence_logit = self.presence_classifier(right_feat)  # [B, 1]

        side_logits = torch.cat(
            [
                left_presence_logit,   # left tumor
                right_presence_logit,  # right tumor
            ],
            dim=1,
        )  # [B, 2]

        left_mult_logits = self.multiplicity_classifier(left_feat)    # [B, 3]
        right_mult_logits = self.multiplicity_classifier(right_feat)  # [B, 3]

        multiplicity_logits = torch.stack(
            [
                left_mult_logits,   # left tumor: none/single/multiple
                right_mult_logits,  # right tumor: none/single/multiple
            ],
            dim=1,
        )  # [B, 2, 3]

        return side_logits, multiplicity_logits, global_feat, left_feat, right_feat


class SemanticTumorMultiplicityContextTokenGenerator(nn.Module):
    """
    Generate 5 explicit tumor context tokens.

    Input:
        global_feat                 [B, 512]
        left_feat                   [B, 512]
        right_feat                  [B, 512]
        side_context_values         [B, 2]
        multiplicity_context_values [B, 2, 3]

    Output:
        context_tokens [B, 5, context_dim]

    Token meaning:
        0 = global
        1 = left tumor semantic
        2 = right tumor semantic
        3 = left tumor multiplicity
        4 = right tumor multiplicity
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_side_values: int = 2,
        num_multiplicity_values: int = 6,
        context_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.context_dim = context_dim
        input_dim = feature_dim * 3 + num_side_values + num_multiplicity_values

        # Semantic tokens: global, left tumor, right tumor.
        self.semantic_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * context_dim),
        )

        # Multiplicity tokens: left tumor multiplicity, right tumor multiplicity.
        self.multiplicity_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * context_dim),
        )

    def forward(
        self,
        global_feat,
        left_feat,
        right_feat,
        side_context_values,
        multiplicity_context_values,
    ):
        B = global_feat.shape[0]

        multiplicity_context_values = multiplicity_context_values.flatten(1)  # [B, 6]

        context_input = torch.cat(
            [
                global_feat,
                left_feat,
                right_feat,
                side_context_values,
                multiplicity_context_values,
            ],
            dim=1,
        )
        # [B, 512*3 + 2 + 6] = [B, 1544]

        semantic_tokens = self.semantic_mlp(context_input).view(B, 3, self.context_dim)
        multiplicity_tokens = self.multiplicity_mlp(context_input).view(B, 2, self.context_dim)

        context_tokens = torch.cat([semantic_tokens, multiplicity_tokens], dim=1)
        return context_tokens  # [B, 5, context_dim]


class SideMaskedContextCrossAttention2D(nn.Module):
    """
    Side-masked context fusion for 5 tumor-only context tokens.

    Context tokens:
        0 = global
        1 = left tumor semantic
        2 = right tumor semantic
        3 = left tumor multiplicity
        4 = right tumor multiplicity

    Left half attends to:
        global + left tumor semantic + left tumor multiplicity

    Right half attends to:
        global + right tumor semantic + right tumor multiplicity
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
            raise ValueError(f"image_dim={image_dim} must be divisible by num_heads={num_heads}")

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
        B, C, H, W_half = image_half.shape
        image_tokens = image_half.flatten(2).transpose(1, 2)  # [B, H*W_half, C]

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
        B, C, H, W = image_feat.shape
        mid = W // 2

        left_feat = image_feat[:, :, :, :mid]
        right_feat = image_feat[:, :, :, mid:]

        left_context = context_tokens[:, [0, 1, 3], :]
        right_context = context_tokens[:, [0, 2, 4], :]

        left_out = self._attend_half(left_feat, left_context)
        right_out = self._attend_half(right_feat, right_context)

        return torch.cat([left_out, right_out], dim=3)


class BiomedCLIPUNetCTEHRAttention(nn.Module):
    """
    BiomedUNet + CT-EHR fusion + tumor presence/multiplicity context.

    Context tokens:
        0 = global
        1 = left tumor semantic
        2 = right tumor semantic
        3 = left tumor multiplicity
        4 = right tumor multiplicity
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

        # ---------- Tumor context branch ----------
        self.side_context_head = MultiScaleSideAwareTumorMultiplicityContextHead(
            scale_channels=(128, 256, 512, 1024, 512),
            scale_embed_dim=128,
            fused_dim=512,
            hidden_dim=side_hidden_dim,
            dropout=0.10,
        )

        self.context_token_generator = SemanticTumorMultiplicityContextTokenGenerator(
            feature_dim=512,
            num_side_values=2,
            num_multiplicity_values=6,
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
        x = self.fusion(x)  # [B, 512, 14, 14]

        # ---------- CT-EHR attention ----------
        if clinical_data is not None:
            clinical_vector = self.clinical_encoder(clinical_data)
            x = self.ct_ehr_attention(x, clinical_vector)

        # ---------- Tumor context branch ----------
        side_logits, multiplicity_logits, global_feat, left_feat, right_feat = self.side_context_head(
            bottleneck=x,
            skip1=skip1,
            skip2=skip2,
            skip3=skip3,
            skip4=skip4,
        )

        side_context_values = torch.sigmoid(side_logits.detach())  # [B, 2]
        multiplicity_context_values = torch.softmax(
            multiplicity_logits.detach(),
            dim=-1,
        )  # [B, 2, 3]

        context_tokens = self.context_token_generator(
            global_feat=global_feat,
            left_feat=left_feat,
            right_feat=right_feat,
            side_context_values=side_context_values,
            multiplicity_context_values=multiplicity_context_values,
        )  # [B, 5, context_dim]

        # ---------- Side-masked context fusion ----------
        x = self.context_fuse_bottleneck(x, context_tokens)
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
            "multiplicity_logits": multiplicity_logits,
            "side_context_values": side_context_values,
            "multiplicity_context_values": multiplicity_context_values,
            "context_tokens": context_tokens,
        }
