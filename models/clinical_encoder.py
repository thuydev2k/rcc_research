import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- embeddings ----
        self.gender_emb = nn.Embedding(3, 4)
        self.smoking_history_emb = nn.Embedding(4, 8)
        self.surgery_type_emb = nn.Embedding(4, 8)
        self.surgical_approach_emb = nn.Embedding(4, 4)
        self.tumor_histologic_subtype_emb = nn.Embedding(15, 16)
        self.pathology_t_stage_emb = nn.Embedding(8, 4)
        # ---- MLP ----
        input_dim = (
            5 +          # numericals
            6 +          # comorbidities
            4 + 8 + 8 + 4 + 16 +  # embeddings
            4            # t_stage
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        emb = torch.cat([
            x["numerical"],
            x["comorbidities"],
            self.gender_emb(x["gender"]),
            self.smoking_history_emb(x["smoking_history"]),
            self.surgery_type_emb(x["surgery_type"]),
            self.surgical_approach_emb(x["surgical_approach"]),
            self.tumor_histologic_subtype_emb(x["tumor_histologic_subtype"]),
            self.pathology_t_stage_emb(x["pathology_t_stage"])
        ], dim=1)

        return self.mlp(emb)