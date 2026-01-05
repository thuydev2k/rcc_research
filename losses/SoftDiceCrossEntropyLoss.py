import torch.nn as nn
import torch.nn.functional as F

from losses.SoftDiceLoss import SoftDiceLoss2D

class SoftDiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftDiceCrossEntropyLoss, self).__init__()

    def forward(self, predicted, target, ce_weight=0.5, dice_weight=0.5):
        ce_loss = F.cross_entropy(predicted, target)

        dice_loss = SoftDiceLoss2D()(predicted, target)
        total_loss = ce_loss * ce_weight + dice_loss* dice_weight

        return total_loss