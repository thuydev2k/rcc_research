import torch
import torch.nn as nn


class ClinicalFeatureEncoder(nn.Module):
    """
    Encode clinical data before FiLM.

    This avoids treating categorical IDs as continuous values.
    Categorical variables use embeddings; numerical variables and missing flags
    are concatenated with comorbidity binary indicators.
    """
    def __init__(
        self,
        n_numerical: int = 4,
        n_comorbidities: int = 1,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.gender_emb = nn.Embedding(num_embeddings=4, embedding_dim=4)

        embedding_dim = 4
        tabular_dim = n_numerical + n_numerical + n_comorbidities + embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, clinical_batch: dict) -> torch.Tensor:
        numerical = clinical_batch["numerical"].float()
        numerical_missing = clinical_batch["numerical_missing"].float()
        comorbidities = clinical_batch["comorbidities"].float()

        gender = self.gender_emb(clinical_batch["gender"].long())

        x = torch.cat(
            [
                numerical,
                numerical_missing,
                comorbidities,
                gender,
            ],
            dim=1,
        )

        return self.encoder(x)
