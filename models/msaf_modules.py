import torch
import torch.nn as nn


class ClinicalTextualTransformation(nn.Module):
    """
    Exp2 Textual Transformation.

    Input:
        clinical: [B, clinical_dim]

    Output:
        clinical_tokens: [B, num_clinical_tokens, hidden_size]

    This is the clinical/EHR branch added on top of Exp1.
    """

    def __init__(
        self,
        clinical_dim: int,
        hidden_size: int = 384,
        num_clinical_tokens: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.clinical_dim = clinical_dim
        self.hidden_size = hidden_size
        self.num_clinical_tokens = num_clinical_tokens

        self.proj = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * num_clinical_tokens),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        x = clinical.float()
        x = self.proj(x)
        x = x.view(x.shape[0], self.num_clinical_tokens, self.hidden_size)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block used for CEA and SMA.

    Input:
        q_tokens:  [B, Nq, C]
        kv_tokens: [B, Nk, C]

    Output:
        out: [B, Nq, C]
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads. "
                f"Got hidden_size={hidden_size}, num_heads={num_heads}"
            )

        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(hidden_size)

        mlp_dim = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(q_tokens)
        kv = self.kv_norm(kv_tokens)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        x = self.attn_norm(q_tokens + attn_out)
        x = self.ffn_norm(x + self.ffn(x))
        return x


class MSAFFusionForSegmentation(nn.Module):
    """
    Exp2 MSAF adapted for segmentation.

    Paper idea:
        CEA: CT visual representation attends to EHR representation.
        SMA: segmentation feature map attends to CEFR.

    This segmentation-only version:
        clinical -> TextualTransformation -> clinical_tokens
        CEA: Q=z12 visual tokens, K/V=clinical_tokens -> CEFR
        SMA: Q=z12 segmentation tokens, K/V=CEFR -> MSFR
        fused_z12 = z12 + learnable_scale * MSFR

    The fused z12 is then sent to the same UNETR-style decoder from Exp1.
    """

    def __init__(
        self,
        clinical_dim: int,
        hidden_size: int = 384,
        num_heads: int = 12,
        num_clinical_tokens: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.textual_transform = ClinicalTextualTransformation(
            clinical_dim=clinical_dim,
            hidden_size=hidden_size,
            num_clinical_tokens=num_clinical_tokens,
            dropout_rate=dropout_rate,
        )

        self.cea = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.sma = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        # Start conservatively so Exp2 initially behaves close to Exp1.
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self.fusion_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        segmentation_tokens: torch.Tensor,
        clinical: torch.Tensor,
    ):
        clinical_tokens = self.textual_transform(clinical)

        cefr = self.cea(
            q_tokens=visual_tokens,
            kv_tokens=clinical_tokens,
        )

        msfr = self.sma(
            q_tokens=segmentation_tokens,
            kv_tokens=cefr,
        )

        fused_tokens = self.fusion_norm(segmentation_tokens + self.fusion_scale * msfr)

        aux = {
            "clinical_tokens": clinical_tokens,
            "cefr": cefr,
            "msfr": msfr,
        }
        return fused_tokens, aux
