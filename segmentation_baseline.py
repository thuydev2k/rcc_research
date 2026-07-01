import gc
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.BiomedUNet_CTEHR_CrossAttention import BiomedCLIPUNetCTEHRAttention
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss2D


def move_clinical_to_device(clinical_batch: dict, device):
    out = {}
    for key, value in clinical_batch.items():
        if key == "case_id":
            out[key] = value
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


# ============================================================
# Segmentation metric helper functions
# ============================================================

def _safe_div(numerator, denominator, eps=1e-10):
    if denominator <= eps:
        return 0.0
    return float(numerator / denominator)


def _init_counts():
    names = [
        "kidney",
        "tumor",
        "cyst",
        "kidney_and_masses",
        "masses",
    ]
    return {name: {"tp": 0, "fp": 0, "fn": 0} for name in names}


def _update_region_counts(counts, name, pred_mask, target_mask):
    tp = torch.logical_and(pred_mask, target_mask).sum().item()
    fp = torch.logical_and(pred_mask, ~target_mask).sum().item()
    fn = torch.logical_and(~pred_mask, target_mask).sum().item()

    counts[name]["tp"] += tp
    counts[name]["fp"] += fp
    counts[name]["fn"] += fn


def _update_all_counts(counts, logits, labels):
    """
    logits: [B, C, H, W]
    labels: [B, H, W]

    Classes:
        0 = background
        1 = kidney
        2 = tumor
        3 = cyst
    """
    pred = torch.argmax(logits, dim=1)

    kidney_pred = pred == 1
    kidney_gt = labels == 1

    tumor_pred = pred == 2
    tumor_gt = labels == 2

    cyst_pred = pred == 3
    cyst_gt = labels == 3

    _update_region_counts(counts, "kidney", kidney_pred, kidney_gt)
    _update_region_counts(counts, "tumor", tumor_pred, tumor_gt)
    _update_region_counts(counts, "cyst", cyst_pred, cyst_gt)

    kidney_and_masses_pred = (pred == 1) | (pred == 2) | (pred == 3)
    kidney_and_masses_gt = (labels == 1) | (labels == 2) | (labels == 3)

    masses_pred = (pred == 2) | (pred == 3)
    masses_gt = (labels == 2) | (labels == 3)

    _update_region_counts(
        counts,
        "kidney_and_masses",
        kidney_and_masses_pred,
        kidney_and_masses_gt,
    )
    _update_region_counts(counts, "masses", masses_pred, masses_gt)


def _compute_dice_iou_from_counts(counts):
    metrics = {}
    for name, c in counts.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        iou = _safe_div(tp, tp + fp + fn)
        metrics[f"{name}_dice"] = dice
        metrics[f"{name}_iou"] = iou

    metrics["mean_fg_dice"] = np.mean([
        metrics["kidney_dice"],
        metrics["tumor_dice"],
        metrics["cyst_dice"],
    ])
    metrics["mean_fg_iou"] = np.mean([
        metrics["kidney_iou"],
        metrics["tumor_iou"],
        metrics["cyst_iou"],
    ])
    metrics["mean_hec_dice"] = np.mean([
        metrics["kidney_and_masses_dice"],
        metrics["masses_dice"],
        metrics["tumor_dice"],
    ])
    metrics["mean_hec_iou"] = np.mean([
        metrics["kidney_and_masses_iou"],
        metrics["masses_iou"],
        metrics["tumor_iou"],
    ])
    return metrics


# ============================================================
# Tumor-side presence + multiplicity helper functions
# ============================================================

SIDE_CLASS_NAMES = [
    "left_tumor",
    "right_tumor",
]

MULTIPLICITY_CLASS_NAMES = [
    "none",
    "single",
    "multiple",
]


def count_connected_components(mask_np):
    """Count 2D connected components using 8-connectivity."""
    structure = np.ones((3, 3), dtype=np.int32)
    _, num_components = ndimage.label(mask_np.astype(np.uint8), structure=structure)
    return int(num_components)


def count_to_multiplicity_class(count: int) -> int:
    """
    0 components  -> none
    1 component   -> single
    2+ components -> multiple
    """
    if count == 0:
        return 0
    if count == 1:
        return 1
    return 2


def build_side_tumor_presence_multiplicity_targets(labels):
    """
    Build side-aware tumor presence and tumor multiplicity targets.

    Args:
        labels: [B, H, W]

    Returns:
        presence_targets:     [B, 2], float
        multiplicity_targets: [B, 2], long

    Side order:
        0 = left_tumor
        1 = right_tumor

    Multiplicity classes:
        0 = none
        1 = single
        2 = multiple
    """
    B, H, W = labels.shape
    mid = W // 2

    labels_np = labels.detach().cpu().numpy()

    presence_list = []
    multiplicity_list = []

    for b in range(B):
        label = labels_np[b]
        left_half = label[:, :mid]
        right_half = label[:, mid:]

        masks = [
            left_half == 2,   # left tumor
            right_half == 2,  # right tumor
        ]

        presence_values = []
        multiplicity_values = []

        for mask in masks:
            count = count_connected_components(mask)
            presence_values.append(1.0 if count > 0 else 0.0)
            multiplicity_values.append(count_to_multiplicity_class(count))

        presence_list.append(presence_values)
        multiplicity_list.append(multiplicity_values)

    presence_targets = torch.tensor(
        presence_list,
        dtype=torch.float32,
        device=labels.device,
    )
    multiplicity_targets = torch.tensor(
        multiplicity_list,
        dtype=torch.long,
        device=labels.device,
    )

    return presence_targets, multiplicity_targets


def multilabel_focal_loss_with_logits(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="mean",
):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * ((1.0 - p_t) ** gamma) * bce

    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


@torch.no_grad()
def compute_side_tumor_pos_weight(
    train_loader,
    device,
    max_pos_weight=10.0,
    eps=1e-6,
):
    """
    pos_weight[c] = num_negative[c] / num_positive[c]

    Classes:
        0 = left_tumor
        1 = right_tumor
    """
    pos = torch.zeros(2, dtype=torch.float64)
    neg = torch.zeros(2, dtype=torch.float64)

    for _, labels, _ in tqdm(train_loader, desc="Computing side tumor pos_weight"):
        labels = labels.long()
        targets, _ = build_side_tumor_presence_multiplicity_targets(labels)
        targets = targets.double()
        pos += targets.sum(dim=0)
        neg += (1.0 - targets).sum(dim=0)

    pos_weight = neg / (pos + eps)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_pos_weight)

    print("\nSide-aware tumor target statistics:")
    for idx, name in enumerate(SIDE_CLASS_NAMES):
        print(
            f"{name:>12} | "
            f"positive: {int(pos[idx].item())} | "
            f"negative: {int(neg[idx].item())} | "
            f"pos_weight: {pos_weight[idx].item():.4f}"
        )

    return pos_weight.float().to(device)


def compute_presence_loss(
    side_logits,
    side_targets,
    lambda_bce=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
    pos_weight=None,
):
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=side_logits.device, dtype=side_logits.dtype)

    bce_loss = F.binary_cross_entropy_with_logits(
        side_logits,
        side_targets,
        pos_weight=pos_weight,
    )

    focal_loss = multilabel_focal_loss_with_logits(
        logits=side_logits,
        targets=side_targets,
        alpha=focal_alpha,
        gamma=focal_gamma,
        reduction="mean",
    )

    presence_loss = lambda_bce * bce_loss + (1.0 - lambda_bce) * focal_loss
    return presence_loss, bce_loss, focal_loss


def compute_total_loss(
    outputs,
    labels,
    seg_criterion,
    side_weight=0.03,
    multiplicity_weight=0.2,
    lambda_bce=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
    side_pos_weight=None,
):
    """
    L_total = L_seg + alpha_dynamic * L_context
    L_context = L_presence + multiplicity_weight * L_multiplicity
    """
    seg_logits = outputs["out"]
    side_logits = outputs["side_logits"]                  # [B, 2]
    multiplicity_logits = outputs["multiplicity_logits"]  # [B, 2, 3]

    seg_loss = seg_criterion(seg_logits, labels)

    side_targets, multiplicity_targets = build_side_tumor_presence_multiplicity_targets(labels)
    side_targets = side_targets.to(device=side_logits.device, dtype=side_logits.dtype)
    multiplicity_targets = multiplicity_targets.to(
        device=multiplicity_logits.device,
        dtype=torch.long,
    )

    presence_loss, bce_loss, focal_loss = compute_presence_loss(
        side_logits=side_logits,
        side_targets=side_targets,
        lambda_bce=lambda_bce,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=side_pos_weight,
    )

    B, S, C = multiplicity_logits.shape
    multiplicity_loss = F.cross_entropy(
        multiplicity_logits.reshape(B * S, C),
        multiplicity_targets.reshape(B * S),
    )

    context_loss = presence_loss + multiplicity_weight * multiplicity_loss
    total_loss = seg_loss + side_weight * context_loss

    return (
        total_loss,
        seg_logits,
        seg_loss,
        context_loss,
        presence_loss,
        bce_loss,
        focal_loss,
        multiplicity_loss,
        side_logits,
        side_targets,
        multiplicity_logits,
        multiplicity_targets,
    )


# ============================================================
# Trackers
# ============================================================

class SideTumorMultiplicityMetricTracker:
    """
    Presence metrics for left/right tumor + multiplicity accuracy.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = torch.zeros(2)
        self.fp = torch.zeros(2)
        self.fn = torch.zeros(2)
        self.tn = torch.zeros(2)

        self.mult_correct = torch.zeros(2)
        self.mult_total = torch.zeros(2)

        self.total_context_loss = 0.0
        self.total_presence_loss = 0.0
        self.total_bce_loss = 0.0
        self.total_focal_loss = 0.0
        self.total_multiplicity_loss = 0.0
        self.num_batches = 0

    @torch.no_grad()
    def update(
        self,
        side_logits,
        side_targets,
        multiplicity_logits=None,
        multiplicity_targets=None,
        context_loss=None,
        presence_loss=None,
        bce_loss=None,
        focal_loss=None,
        multiplicity_loss=None,
    ):
        probs = torch.sigmoid(side_logits)
        preds = (probs >= self.threshold).float()
        targets = side_targets.float()

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        self.tp += ((preds == 1) & (targets == 1)).sum(dim=0)
        self.fp += ((preds == 1) & (targets == 0)).sum(dim=0)
        self.fn += ((preds == 0) & (targets == 1)).sum(dim=0)
        self.tn += ((preds == 0) & (targets == 0)).sum(dim=0)

        if multiplicity_logits is not None and multiplicity_targets is not None:
            mult_preds = torch.argmax(multiplicity_logits.detach().cpu(), dim=-1)  # [B,2]
            mult_targets = multiplicity_targets.detach().cpu()                    # [B,2]
            self.mult_correct += (mult_preds == mult_targets).sum(dim=0)
            self.mult_total += torch.ones_like(mult_targets, dtype=torch.float32).sum(dim=0)

        if context_loss is not None:
            self.total_context_loss += context_loss.item()
        if presence_loss is not None:
            self.total_presence_loss += presence_loss.item()
        if bce_loss is not None:
            self.total_bce_loss += bce_loss.item()
        if focal_loss is not None:
            self.total_focal_loss += focal_loss.item()
        if multiplicity_loss is not None:
            self.total_multiplicity_loss += multiplicity_loss.item()

        self.num_batches += 1

    def compute(self):
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        mult_accuracy = self.mult_correct / (self.mult_total + eps)

        n = max(self.num_batches, 1)

        metrics = {
            "context_loss": self.total_context_loss / n,
            "presence_loss": self.total_presence_loss / n,
            "bce_loss": self.total_bce_loss / n,
            "focal_loss": self.total_focal_loss / n,
            "multiplicity_loss": self.total_multiplicity_loss / n,
            "mean_f1": f1.mean().item(),
            "mean_precision": precision.mean().item(),
            "mean_recall": recall.mean().item(),
            "mean_accuracy": accuracy.mean().item(),
            "multiplicity_mean_accuracy": mult_accuracy.mean().item(),
        }

        for idx, name in enumerate(SIDE_CLASS_NAMES):
            metrics[f"{name}_f1"] = f1[idx].item()
            metrics[f"{name}_precision"] = precision[idx].item()
            metrics[f"{name}_recall"] = recall[idx].item()
            metrics[f"{name}_accuracy"] = accuracy[idx].item()
            metrics[f"{name}_multiplicity_accuracy"] = mult_accuracy[idx].item()

        return metrics


class DynamicTumorContextWeight:
    """
    Dynamic auxiliary tumor-context loss weight.

    Weight increases only when:
        1. tumor segmentation Dice improves
        2. side tumor presence classifier validation mean F1 is acceptable
    """

    def __init__(
        self,
        alpha_min=0.03,
        alpha_max=0.13,
        tumor_threshold=0.85,
        cls_threshold=0.70,
    ):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tumor_threshold = tumor_threshold
        self.cls_threshold = cls_threshold
        self.current_weight = alpha_min

    def get(self):
        return self.current_weight

    def update(self, tumor_dice, classifier_mean_f1):
        tumor_score = float(tumor_dice)
        cls_score = float(classifier_mean_f1)

        tumor_ratio = min(max(tumor_score / self.tumor_threshold, 0.0), 1.0)
        cls_ratio = min(max(cls_score / self.cls_threshold, 0.0), 1.0)

        self.current_weight = self.alpha_min + (
            self.alpha_max - self.alpha_min
        ) * tumor_ratio * cls_ratio

        info = {
            "tumor_score": tumor_score,
            "cls_score": cls_score,
            "tumor_ratio": tumor_ratio,
            "cls_ratio": cls_ratio,
            "next_weight": self.current_weight,
        }
        return self.current_weight, info


# ============================================================
# Training entry point
# ============================================================

def segmentation_baseline(
    train_loader,
    valid_loader,
    device,
    epoch,
    lr,
    out_classes=4,
    n_numerical=4,
    n_comorbidities=1,
    save_dir="saved_BiomedCLIP_UNet_CTEHR_TumorMultiplicity5Tokens_Dynamic_model",
):
    os.makedirs(save_dir, exist_ok=True)

    model = BiomedCLIPUNetCTEHRAttention(
        in_channels=1,
        out_classes=out_classes,
        biomed_embed_dim=512,
        clinical_embed_dim=64,
        n_numerical=n_numerical,
        n_comorbidities=n_comorbidities,
        num_clinical_tokens=4,
        num_heads=8,
        side_hidden_dim=128,
        context_dim=128,
    ).to(device)

    scaler = torch.amp.GradScaler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epoch,
        eta_min=lr * 0.01,
    )
    criterion = SoftDiceCrossEntropyLoss2D().to(device)
    writer = SummaryWriter()

    lambda_bce = 0.5
    focal_alpha = 0.25
    focal_gamma = 2.0
    multiplicity_weight = 0.2

    dynamic_context_weight = DynamicTumorContextWeight(
        alpha_min=0.03,
        alpha_max=0.13,
        tumor_threshold=0.85,
        cls_threshold=0.70,
    )

    side_pos_weight = compute_side_tumor_pos_weight(
        train_loader=train_loader,
        device=device,
        max_pos_weight=10.0,
    )

    best_valid_loss = np.inf
    best_mean_hec_dice = -np.inf
    save_check = 0

    for i in range(epoch):
        if save_check > 50:
            print(f"Early stopping at epoch {i}")
            break

        side_weight = dynamic_context_weight.get()

        train_loss, train_seg_loss, train_context_metrics = train_fn(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
            side_weight=side_weight,
            multiplicity_weight=multiplicity_weight,
            lambda_bce=lambda_bce,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            side_pos_weight=side_pos_weight,
        )

        valid_loss, valid_seg_loss, metrics, valid_context_metrics = eval_fn(
            valid_loader,
            model,
            device,
            criterion,
            side_weight=side_weight,
            multiplicity_weight=multiplicity_weight,
            lambda_bce=lambda_bce,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            side_pos_weight=side_pos_weight,
        )

        next_side_weight, dynamic_info = dynamic_context_weight.update(
            tumor_dice=metrics["tumor_dice"],
            classifier_mean_f1=valid_context_metrics["mean_f1"],
        )

        if math.isnan(valid_loss) or math.isnan(train_loss):
            print(f"Early stopping at epoch {i} by NaN")
            break

        current_mean_hec_dice = metrics["mean_hec_dice"]

        if current_mean_hec_dice > best_mean_hec_dice:
            save_check = 0
            best_mean_hec_dice = current_mean_hec_dice
            best_valid_loss = valid_loss

            torch.save({
                "epoch": i + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "lrscheduler": scheduler.state_dict(),
                "best_valid_loss": best_valid_loss,
                "best_mean_hec_dice": best_mean_hec_dice,
                "metrics": metrics,
                "valid_context_metrics": valid_context_metrics,
                "side_weight": side_weight,
                "multiplicity_weight": multiplicity_weight,
                "lambda_bce": lambda_bce,
                "focal_alpha": focal_alpha,
                "focal_gamma": focal_gamma,
                "dynamic_info": dynamic_info,
                "save_criterion": "best_mean_hec_dice",
            }, os.path.join(save_dir, "best_model_tumor_multiplicity_5tokens_dynamic.pt"))

            print("Model saved")
        else:
            save_check += 1

        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

        # ---------- TensorBoard loss ----------
        writer.add_scalar("Loss/train_total", train_loss, i)
        writer.add_scalar("Loss/valid_total", valid_loss, i)
        writer.add_scalar("Loss/train_seg", train_seg_loss, i)
        writer.add_scalar("Loss/valid_seg", valid_seg_loss, i)

        writer.add_scalar("TumorContext/train_context_loss", train_context_metrics["context_loss"], i)
        writer.add_scalar("TumorContext/train_presence_loss", train_context_metrics["presence_loss"], i)
        writer.add_scalar("TumorContext/train_multiplicity_loss", train_context_metrics["multiplicity_loss"], i)
        writer.add_scalar("TumorContext/train_mean_f1", train_context_metrics["mean_f1"], i)
        writer.add_scalar("TumorContext/train_multiplicity_acc", train_context_metrics["multiplicity_mean_accuracy"], i)

        writer.add_scalar("TumorContext/valid_context_loss", valid_context_metrics["context_loss"], i)
        writer.add_scalar("TumorContext/valid_presence_loss", valid_context_metrics["presence_loss"], i)
        writer.add_scalar("TumorContext/valid_multiplicity_loss", valid_context_metrics["multiplicity_loss"], i)
        writer.add_scalar("TumorContext/valid_mean_f1", valid_context_metrics["mean_f1"], i)
        writer.add_scalar("TumorContext/valid_multiplicity_acc", valid_context_metrics["multiplicity_mean_accuracy"], i)
        writer.add_scalar("TumorContext/dynamic_weight", side_weight, i)
        writer.add_scalar("TumorContext/next_dynamic_weight", next_side_weight, i)

        for class_name in SIDE_CLASS_NAMES:
            writer.add_scalar(
                f"TumorContext/valid_{class_name}_f1",
                valid_context_metrics[f"{class_name}_f1"],
                i,
            )
            writer.add_scalar(
                f"TumorContext/valid_{class_name}_recall",
                valid_context_metrics[f"{class_name}_recall"],
                i,
            )
            writer.add_scalar(
                f"TumorContext/valid_{class_name}_multiplicity_acc",
                valid_context_metrics[f"{class_name}_multiplicity_accuracy"],
                i,
            )

        # ---------- TensorBoard segmentation metrics ----------
        writer.add_scalar("Metrics/mean_fg_dice", metrics["mean_fg_dice"], i)
        writer.add_scalar("Metrics/mean_fg_iou", metrics["mean_fg_iou"], i)
        writer.add_scalar("Metrics/mean_hec_dice", metrics["mean_hec_dice"], i)
        writer.add_scalar("Metrics/mean_hec_iou", metrics["mean_hec_iou"], i)
        writer.add_scalar("Metrics/kidney_dice", metrics["kidney_dice"], i)
        writer.add_scalar("Metrics/tumor_dice", metrics["tumor_dice"], i)
        writer.add_scalar("Metrics/cyst_dice", metrics["cyst_dice"], i)
        writer.add_scalar("Metrics/kidney_and_masses_dice", metrics["kidney_and_masses_dice"], i)
        writer.add_scalar("Metrics/masses_dice", metrics["masses_dice"], i)

        # ---------- Print results ----------
        print(f"EPOCH {i + 1}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")
        print(f"Mean FG Dice: {metrics['mean_fg_dice']:.4f} | Mean FG IoU: {metrics['mean_fg_iou']:.4f}")
        print(f"Kidney Dice: {metrics['kidney_dice']:.4f} | Tumor Dice: {metrics['tumor_dice']:.4f} | Cyst Dice: {metrics['cyst_dice']:.4f}")
        print(f"K&M Dice: {metrics['kidney_and_masses_dice']:.4f} | Masses Dice: {metrics['masses_dice']:.4f}")
        print(f"Mean HEC Dice: {metrics['mean_hec_dice']:.4f} | Mean HEC IoU: {metrics['mean_hec_iou']:.4f}")
        print(f"Best Mean HEC Dice: {best_mean_hec_dice:.4f}")

        print(
            f"Dynamic Tumor Context Weight | Current: {side_weight:.4f} | Next: {next_side_weight:.4f} | "
            f"Tumor Dice: {dynamic_info['tumor_score']:.4f} | Presence F1: {dynamic_info['cls_score']:.4f} | "
            f"Tumor Ratio: {dynamic_info['tumor_ratio']:.4f} | Cls Ratio: {dynamic_info['cls_ratio']:.4f}"
        )

        print(
            f"Train Loss | Total: {train_loss:.4f} | Seg: {train_seg_loss:.4f} | "
            f"Context: {train_context_metrics['context_loss']:.4f} | "
            f"Presence: {train_context_metrics['presence_loss']:.4f} | "
            f"Multiplicity: {train_context_metrics['multiplicity_loss']:.4f}"
        )

        print(
            f"Valid Loss | Total: {valid_loss:.4f} | Seg: {valid_seg_loss:.4f} | "
            f"Context: {valid_context_metrics['context_loss']:.4f} | "
            f"Presence: {valid_context_metrics['presence_loss']:.4f} | "
            f"Multiplicity: {valid_context_metrics['multiplicity_loss']:.4f}"
        )

        print(
            f"Valid Tumor Presence F1 | Left: {valid_context_metrics['left_tumor_f1']:.4f} | "
            f"Right: {valid_context_metrics['right_tumor_f1']:.4f} | Mean: {valid_context_metrics['mean_f1']:.4f}"
        )
        print(
            f"Valid Tumor Multiplicity Acc | Left: {valid_context_metrics['left_tumor_multiplicity_accuracy']:.4f} | "
            f"Right: {valid_context_metrics['right_tumor_multiplicity_accuracy']:.4f} | "
            f"Mean: {valid_context_metrics['multiplicity_mean_accuracy']:.4f}"
        )

    writer.flush()
    writer.close()


def train_fn(
    loader,
    model,
    optimizer,
    device,
    criterion,
    scaler=None,
    side_weight=0.03,
    multiplicity_weight=0.2,
    lambda_bce=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
    side_pos_weight=None,
):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    tracker = SideTumorMultiplicityMetricTracker(threshold=0.5)
    use_amp = scaler is not None and str(device).startswith("cuda")

    for images, labels, clinical_batch in tqdm(loader):
        images = images.float().to(device)
        labels = labels.long().to(device)
        clinical_batch = move_clinical_to_device(clinical_batch, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, clinical_batch)
                (
                    loss,
                    seg_logits,
                    seg_loss,
                    context_loss,
                    presence_loss,
                    bce_loss,
                    focal_loss,
                    multiplicity_loss,
                    side_logits,
                    side_targets,
                    multiplicity_logits,
                    multiplicity_targets,
                ) = compute_total_loss(
                    outputs=outputs,
                    labels=labels,
                    seg_criterion=criterion,
                    side_weight=side_weight,
                    multiplicity_weight=multiplicity_weight,
                    lambda_bce=lambda_bce,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    side_pos_weight=side_pos_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, clinical_batch)
            (
                loss,
                seg_logits,
                seg_loss,
                context_loss,
                presence_loss,
                bce_loss,
                focal_loss,
                multiplicity_loss,
                side_logits,
                side_targets,
                multiplicity_logits,
                multiplicity_targets,
            ) = compute_total_loss(
                outputs=outputs,
                labels=labels,
                seg_criterion=criterion,
                side_weight=side_weight,
                multiplicity_weight=multiplicity_weight,
                lambda_bce=lambda_bce,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                side_pos_weight=side_pos_weight,
            )
            loss.backward()
            optimizer.step()

        tracker.update(
            side_logits=side_logits,
            side_targets=side_targets,
            multiplicity_logits=multiplicity_logits,
            multiplicity_targets=multiplicity_targets,
            context_loss=context_loss,
            presence_loss=presence_loss,
            bce_loss=bce_loss,
            focal_loss=focal_loss,
            multiplicity_loss=multiplicity_loss,
        )

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()

    return total_loss / len(loader), total_seg_loss / len(loader), tracker.compute()


def eval_fn(
    loader,
    model,
    device,
    criterion,
    side_weight=0.03,
    multiplicity_weight=0.2,
    lambda_bce=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
    side_pos_weight=None,
):
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    counts = _init_counts()
    tracker = SideTumorMultiplicityMetricTracker(threshold=0.5)
    use_amp = str(device).startswith("cuda")

    with torch.no_grad():
        for images, labels, clinical_batch in tqdm(loader):
            images = images.float().to(device)
            labels = labels.long().to(device)
            clinical_batch = move_clinical_to_device(clinical_batch, device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images, clinical_batch)
                    (
                        loss,
                        seg_logits,
                        seg_loss,
                        context_loss,
                        presence_loss,
                        bce_loss,
                        focal_loss,
                        multiplicity_loss,
                        side_logits,
                        side_targets,
                        multiplicity_logits,
                        multiplicity_targets,
                    ) = compute_total_loss(
                        outputs=outputs,
                        labels=labels,
                        seg_criterion=criterion,
                        side_weight=side_weight,
                        multiplicity_weight=multiplicity_weight,
                        lambda_bce=lambda_bce,
                        focal_alpha=focal_alpha,
                        focal_gamma=focal_gamma,
                        side_pos_weight=side_pos_weight,
                    )
            else:
                outputs = model(images, clinical_batch)
                (
                    loss,
                    seg_logits,
                    seg_loss,
                    context_loss,
                    presence_loss,
                    bce_loss,
                    focal_loss,
                    multiplicity_loss,
                    side_logits,
                    side_targets,
                    multiplicity_logits,
                    multiplicity_targets,
                ) = compute_total_loss(
                    outputs=outputs,
                    labels=labels,
                    seg_criterion=criterion,
                    side_weight=side_weight,
                    multiplicity_weight=multiplicity_weight,
                    lambda_bce=lambda_bce,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    side_pos_weight=side_pos_weight,
                )

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()

            tracker.update(
                side_logits=side_logits,
                side_targets=side_targets,
                multiplicity_logits=multiplicity_logits,
                multiplicity_targets=multiplicity_targets,
                context_loss=context_loss,
                presence_loss=presence_loss,
                bce_loss=bce_loss,
                focal_loss=focal_loss,
                multiplicity_loss=multiplicity_loss,
            )

            _update_all_counts(counts=counts, logits=seg_logits, labels=labels)

    valid_loss = total_loss / len(loader)
    valid_seg_loss = total_seg_loss / len(loader)
    metrics = _compute_dice_iou_from_counts(counts)
    context_metrics = tracker.compute()

    return valid_loss, valid_seg_loss, metrics, context_metrics
