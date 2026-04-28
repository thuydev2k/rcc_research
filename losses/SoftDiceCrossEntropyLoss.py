import torch.nn as nn
import torch.nn.functional as F

from losses.SoftDiceLoss3D import SoftDiceLoss3D


class SoftDiceCrossEntropyLoss3D(nn.Module):
    def __init__(
        self,
        ce_weight=0.5,
        dice_weight=0.5,
        ignore_bg=False,
    ):
        super(SoftDiceCrossEntropyLoss3D, self).__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_bg = ignore_bg
        self.dice_loss = SoftDiceLoss3D()

    def forward(self, predicted, target):
        """
        predicted: [B, C, D, H, W]
        target:    [B, D, H, W]
        """

        ce_loss = F.cross_entropy(predicted, target.long())

        dice_loss = self.dice_loss(
            predicted,
            target,
            ignore_bg=self.ignore_bg,
        )

        total_loss = (
            ce_loss * self.ce_weight
            + dice_loss * self.dice_weight
        )

        return total_loss