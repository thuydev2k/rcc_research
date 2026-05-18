from typing import Sequence, Union

import torch

from models.ViT_UNetSeg3D import ViTUNetSeg3D, count_parameters
from models.msaf_modules import MSAFFusionForSegmentation


class ViTUNetSeg3D_MSAF(ViTUNetSeg3D):
    """
    Exp2 model = Exp1 ViTUNetSeg3D + clinical Textual Transformation + MSAF.

    This class reuses the Exp1 implementation directly:
        - img_embed
        - vit
        - encoder1/2/3/4
        - decoder5/4/3/2
        - out
        - proj_feat

    Only the bottleneck path is changed:
        Exp1: dec4 = proj_feat(z12)
        Exp2: fused_z12 = MSAF(z12, clinical), then dec4 = proj_feat(fused_z12)
    """

    def __init__(
        self,
        clinical_dim: int,
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
        num_clinical_tokens: int = 1,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout_rate=dropout_rate,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )

        self.clinical_dim = clinical_dim
        self.num_clinical_tokens = num_clinical_tokens

        self.msaf = MSAFFusionForSegmentation(
            clinical_dim=clinical_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clinical_tokens=num_clinical_tokens,
            dropout_rate=dropout_rate,
        )

    def forward(self, x_in: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_in:     [B, 1, D, H, W]
            clinical: [B, clinical_dim]

        Returns:
            logits: [B, 4, D, H, W]
        """
        tokens = self.img_embed(x_in)
        z12, hidden_states_out = self.vit(tokens)

        enc1 = self.encoder1(x_in)

        # Same intermediate ViT features as Exp1.
        z3 = hidden_states_out[3]
        z6 = hidden_states_out[6]
        z9 = hidden_states_out[9]

        enc2 = self.encoder2(self.proj_feat(z3))
        enc3 = self.encoder3(self.proj_feat(z6))
        enc4 = self.encoder4(self.proj_feat(z9))

        # Only new part in Exp2: fuse z12 with clinical representation.
        fused_z12, _ = self.msaf(
            visual_tokens=z12,
            segmentation_tokens=z12,
            clinical=clinical,
        )

        dec4 = self.proj_feat(fused_z12)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        logits = self.out(dec0)
        return logits
