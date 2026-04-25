import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, n_features, n_clinical):
        super(FiLM, self).__init__()
        
        self.controller = nn.Sequential(
            nn.Linear(n_clinical, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_features * 2)
        )

    def forward(self, x, clinical_vector):
        params = self.controller(clinical_vector)
        
        gamma, beta = torch.split(params, x.shape[1], dim=1)
        
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        
        return (1 + gamma) * x + beta