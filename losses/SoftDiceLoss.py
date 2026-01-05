import torch
import torch.nn as nn

class SoftDiceLoss2D(nn.Module):
    def __init__(self):
        super(SoftDiceLoss2D, self).__init__()
        
    def forward(self, predicted, target, smooth=1e-10, ignore_bg=False):
        predicted = torch.softmax(predicted, dim=1)

        if ignore_bg:
            target = target[:, 1:]
            predicted = predicted[:, 1:]

        intersection = torch.sum(predicted * target, dim=(2, 3))
        gt = torch.sum(target, dim=(2, 3))
        pred = torch.sum(predicted, dim=(2, 3))
        total = gt + pred

        dice = (2 * intersection + smooth) / (total + smooth)
        dice_loss = (1 - dice).mean()

        return dice_loss