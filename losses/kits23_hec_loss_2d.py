from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


HEC_REGION_DEFS = {
    'kidney_and_masses': [1, 2, 3],
    'masses': [2, 3],
    'tumor': [2],
}


def labels_to_region_mask(labels: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
    """
    labels: [B, H, W]
    return: [B, H, W] float mask
    """
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for c in class_ids:
        mask |= labels == c
    return mask.float()


class WeightedCEDiceHECLoss2D(nn.Module):
    """
    Loss for KiTS23 2D segmentation.

    Total loss:
        total = ce_weight * WeightedCrossEntropy
              + dice_weight * DiceLossIgnoreBackground
              + hec_weight * HECRegionDiceLoss

    Inputs:
        logits: [B, C, H, W]
        labels: [B, H, W]

    Default class weights are intentionally moderate:
        background: 0.05
        kidney:     1.00
        tumor:      3.00
        cyst:       5.00
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[List[float]] = None,
        ce_weight: float = 0.3,
        dice_weight: float = 0.4,
        hec_weight: float = 0.3,
        smooth: float = 1e-6,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.hec_weight = hec_weight
        self.smooth = smooth

        if class_weights is None:
            class_weights = [0.05, 1.0, 3.0, 5.0]

        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def dice_loss_ignore_background(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Multi-class soft Dice loss over foreground classes only.

        probs:  [B, C, H, W]
        labels: [B, H, W]
        """
        labels_one_hot = F.one_hot(labels.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

        # Ignore background class 0.
        probs_fg = probs[:, 1:, :, :]
        target_fg = labels_one_hot[:, 1:, :, :]

        dims = (0, 2, 3)
        intersection = (probs_fg * target_fg).sum(dim=dims)
        denominator = probs_fg.sum(dim=dims) + target_fg.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()

    def hec_region_dice_loss(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        losses = []

        for _, class_ids in HEC_REGION_DEFS.items():
            pred_region = probs[:, class_ids, :, :].sum(dim=1)  # [B, H, W]
            target_region = labels_to_region_mask(labels, class_ids)

            intersection = (pred_region * target_region).sum()
            denominator = pred_region.sum() + target_region.sum()
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            losses.append(1.0 - dice)

        return torch.stack(losses).mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        probs = torch.softmax(logits, dim=1)

        ce = F.cross_entropy(logits, labels, weight=self.class_weights)
        dice_no_bg = self.dice_loss_ignore_background(probs, labels)
        hec_dice = self.hec_region_dice_loss(probs, labels)

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice_no_bg
            + self.hec_weight * hec_dice
        )
        return total

    @torch.no_grad()
    def components(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        labels = labels.long()
        probs = torch.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, labels, weight=self.class_weights)
        dice_no_bg = self.dice_loss_ignore_background(probs, labels)
        hec_dice = self.hec_region_dice_loss(probs, labels)
        total = self.forward(logits, labels)
        return {
            'loss_total': float(total.item()),
            'loss_weighted_ce': float(ce.item()),
            'loss_dice_no_bg': float(dice_no_bg.item()),
            'loss_hec_dice': float(hec_dice.item()),
        }
