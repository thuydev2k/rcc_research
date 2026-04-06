import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_labels(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    target_onehot = F.one_hot(target.long(), num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    return target_onehot

class AsymmetricFocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 2.0,
        ignore_index: int | None = None,
        eps: float = 1e-7,
        background_index: int = 0,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.background_index = background_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        target_onehot = one_hot_labels(target, num_classes).to(logits.device)

        ce = -target_onehot * torch.log(probs)

        bg_weight = 1.0 - self.delta
        fg_weight = self.delta

        class_weights = torch.ones(num_classes, device=logits.device, dtype=logits.dtype) * fg_weight
        class_weights[0] = bg_weight
        class_weights = class_weights.view(1, num_classes, 1, 1)

        focal_weight = torch.ones_like(probs)
        focal_weight[:, 0] = (1.0 - probs[:, 0]) ** self.gamma

        loss = class_weights * focal_weight * ce
        return loss.mean()

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        delta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_onehot = one_hot_labels(target, num_classes).to(logits.device)

        dims = (2, 3)
       
        tp = torch.sum(probs * target_onehot, dim=dims)   
        fn = torch.sum(target_onehot * (1.0 - probs), dim=dims)
        fp = torch.sum((1.0 - target_onehot) * probs, dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.delta * fn + (1.0 - self.delta) * fp + self.smooth
        )

        loss = torch.zeros_like(tversky)
        loss[:, 0] = 1.0 - tversky[:, 0]

        if num_classes > 1:
            base = torch.clamp(1.0 - tversky[:, 1:], min=self.smooth)
            loss[:, 1:] = base ** (1.0 - self.gamma)

        return loss.mean()

class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.7, gamma_ce=2.0, gamma_tversky=0.75):
        super().__init__()
        self.weight = weight
        self.afce = AsymmetricFocalCrossEntropyLoss(delta=delta, gamma=gamma_ce)
        self.aftl = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma_tversky)

    def forward(self, logits, target):
        afce_loss = self.afce(logits, target)
        aftl_loss = self.aftl(logits, target)
        return self.weight * afce_loss + (1.0 - self.weight) * aftl_loss