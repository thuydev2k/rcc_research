from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep


class ConvPatchEmbedding3D(nn.Module):
    """
    USCNet-style visual transformation for 3D CT.

    Input:
        x: [B, 1, D, H, W]

    Output:
        tokens: [B, N, hidden_size]

    This replaces the image part of USCNet's VTT module:
        3D CT volume -> non-overlapping 3D patches -> patch tokens + positional embedding
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int] = 16,
        hidden_size: int = 384,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)

        for img_d, p_d in zip(img_size, patch_size):
            if img_d % p_d != 0:
                raise ValueError(
                    f"img_size must be divisible by patch_size. "
                    f"Got img_size={img_size}, patch_size={patch_size}"
                )

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.n_patches = int(self.feat_size[0] * self.feat_size[1] * self.feat_size[2])

        self.patch_embeddings = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)             # [B, hidden_size, D/16, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)         # [B, N, hidden_size]
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class ViTEncoderNoEmbed(nn.Module):
    """
    ViT encoder without patch embedding.

    This mirrors the ViTNoEmbed idea in KidneyStoneSC:
    patch embedding is done before this module, then the token sequence is passed
    through Transformer blocks.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        mlp_dim: int = 1536,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads. "
                f"Got hidden_size={hidden_size}, num_heads={num_heads}"
            )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        hidden_states_out = []

        for block in self.blocks:
            x = block(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        return x, hidden_states_out


class ViTUNetSeg3D(nn.Module):
    """
    Exp 1: image-only ViT-UNetSeg for KiTS23 segmentation.

    Based on the USCNet / KidneyStoneSC segmentation path:
        image -> patch embedding -> ViT encoder -> Z3/Z6/Z9/Z12 -> UNETR-style decoder

    Differences from the original KidneyStoneSC KSCNet:
        - no EHR input
        - no CT-EHR attention
        - no SMA/MSAF
        - no classification head
        - out_channels=4 for KiTS23:
            0 background, 1 kidney, 2 tumor, 3 cyst
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        img_size: Union[Sequence[int], int] = (96, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 384,
        mlp_dim: int = 1536,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: int = 16,
        dropout_rate: float = 0.0,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.img_embed = ConvPatchEmbedding3D(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.vit = ViTEncoderNoEmbed(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # Same UNETR-style skip/decoder blocks used by KidneyStoneSC's UNETR/KSCNet.
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token sequence [B, N, hidden_size] to 3D feature map:
            [B, hidden_size, D/16, H/16, W/16]
        """
        new_view = (x.size(0), *self.feat_size, self.hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(self.feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: [B, 1, D, H, W]
        """
        tokens = self.img_embed(x_in)                 # [B, N, C]
        z12, hidden_states_out = self.vit(tokens)     # final tokens and intermediate Z features

        enc1 = self.encoder1(x_in)

        # Match the indexing style in KidneyStoneSC nets.py:
        # hidden_states_out[3], [6], [9], and final z12.
        z3 = hidden_states_out[3]
        z6 = hidden_states_out[6]
        z9 = hidden_states_out[9]

        enc2 = self.encoder2(self.proj_feat(z3))
        enc3 = self.encoder3(self.proj_feat(z6))
        enc4 = self.encoder4(self.proj_feat(z9))

        dec4 = self.proj_feat(z12)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        logits = self.out(dec0)
        return logits


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
