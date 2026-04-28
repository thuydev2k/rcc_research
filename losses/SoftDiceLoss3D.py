import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss3D(nn.Module):
    def __init__(self):
        super(SoftDiceLoss3D, self).__init__()

    def forward(self, predicted, target, smooth=1e-10, ignore_bg=False):
        """
        predicted: [B, C, D, H, W]
        target:    [B, D, H, W]
        """

        B, C, D, H, W = predicted.shape

        predicted = torch.softmax(predicted, dim=1)

        target = F.one_hot(target.long(), num_classes=C)
        target = target.permute(0, 4, 1, 2, 3).float()
        target = target.to(predicted.device)

        if ignore_bg:
            target = target[:, 1:]
            predicted = predicted[:, 1:]

        intersection = torch.sum(predicted * target, dim=(2, 3, 4))
        gt = torch.sum(target, dim=(2, 3, 4))
        pred = torch.sum(predicted, dim=(2, 3, 4))

        total = gt + pred

        dice = (2 * intersection + smooth) / (total + smooth)
        dice_loss = (1 - dice).mean()

        return dice_loss