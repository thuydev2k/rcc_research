import torch
import torch.nn as nn
import open_clip
from torchinfo import summary

class BiomedCLIPEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        )

        self.visual = model.visual

        del model

        for param in self.visual.parameters():
            param.requires_grad = False
        
        self.hidden_dim = 768
        self.proj = nn.Conv2d(self.hidden_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.visual.trunk.forward_features(x)

        x = x[:, 1:, :]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.proj(x)

        return x