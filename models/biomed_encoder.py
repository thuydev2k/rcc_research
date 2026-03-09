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
        
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.hidden_dim = 768
        
        self.proj = nn.Conv2d(self.hidden_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.visual.trunk.patch_embed(x)

        cls_token = self.visual.trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.visual.trunk.pos_embed
        x = self.visual.trunk.pos_drop(x)

        features = []

        for i, blk in enumerate(self.visual.trunk.blocks):
            x = blk(x)

            if i in [2,5,8,11]:
                feat = x[:, 1:, :]
                B, N, C = feat.shape
                H = W = int(N ** 0.5)
                feat = feat.transpose(1, 2).reshape(B, C, H, W)
                feat = self.proj(feat)

                features.append(feat)

        return features