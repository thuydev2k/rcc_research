import torch.nn as nn
import torch
import torch.nn.functional as F

from losses.SoftDiceLoss import SoftDiceLoss2D

CLASS_WEIGHTS = (0.05, 1, 2, 3)

class SoftDiceCrossEntropyLoss2D(nn.Module):
    def __init__(
        self,
        ce_weight=0.5,
        dice_weight=0.5,
        ignore_bg=True,
        class_weights=CLASS_WEIGHTS,
    ):
        super(SoftDiceCrossEntropyLoss2D, self).__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_bg = ignore_bg
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.dice_loss = SoftDiceLoss2D()

    def forward(self, predicted, target):

        ce_loss = F.cross_entropy(predicted, target.long())

        dice_loss = self.dice_loss(
            predicted,
            target,
            ignore_bg=self.ignore_bg,
        )

        total_loss = (ce_loss * self.ce_weight) + (dice_loss * self.dice_weight)

        return total_loss