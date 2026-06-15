import gc
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.BiomedUNet_CTEHR_CrossAttention import BiomedCLIPUNetCTEHRAttentionPresence
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
# Metric helper functions
# ============================================================

def _safe_div(numerator, denominator, eps=1e-10):
    if denominator <= eps:
        return 0.0
    return float(numerator / denominator)


def _init_counts():
    """
    Dataset-level counts.
    These are accumulated over the whole validation set.
    """
    names = [
        "kidney",
        "tumor",
        "cyst",
        "kidney_and_masses",
        "masses",
    ]

    counts = {}

    for name in names:
        counts[name] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    return counts


def _update_region_counts(counts, name, pred_mask, target_mask):
    tp = torch.logical_and(pred_mask, target_mask).sum().item()
    fp = torch.logical_and(pred_mask, ~target_mask).sum().item()
    fn = torch.logical_and(~pred_mask, target_mask).sum().item()

    counts[name]["tp"] += tp
    counts[name]["fp"] += fp
    counts[name]["fn"] += fn


def _update_all_counts(counts, logits, labels):
    """
    logits:
        [B, C, H, W]

    labels:
        [B, H, W]

    Classes:
        0 = background
        1 = kidney
        2 = tumor
        3 = cyst
    """
    pred = torch.argmax(logits, dim=1)  # [B, H, W]

    # ---------- class-wise regions ----------
    kidney_pred = pred == 1
    kidney_gt = labels == 1

    tumor_pred = pred == 2
    tumor_gt = labels == 2

    cyst_pred = pred == 3
    cyst_gt = labels == 3

    _update_region_counts(counts, "kidney", kidney_pred, kidney_gt)
    _update_region_counts(counts, "tumor", tumor_pred, tumor_gt)
    _update_region_counts(counts, "cyst", cyst_pred, cyst_gt)

    # ---------- HEC regions ----------
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

    _update_region_counts(
        counts,
        "masses",
        masses_pred,
        masses_gt,
    )


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

def build_presence_targets(labels):
    """
    labels:
        [B, H, W]

    Output:
        presence_targets:
            [B, 2]

    Channels:
        0 = has_tumor
        1 = has_cyst
    """

    # has_mass = ((labels == 2) | (labels == 3)).any(dim=(1, 2)).float()
    has_tumor = (labels == 2).any(dim=(1, 2)).float()
    has_cyst = (labels == 3).any(dim=(1, 2)).float()

    presence_targets = torch.stack(
        [has_tumor, has_cyst],
        dim=1,
    )

    return presence_targets


def compute_seg_presence_loss(
    outputs,
    labels,
    seg_criterion,
    presence_weight: float = 0.1,
    presence_pos_weight=None,
):
    """
    outputs:
        {
            "out": [B, 4, H, W],
            "presence_logits": [B, 2]
        }

    labels:
        [B, H, W]
    """

    seg_logits = outputs["out"]
    presence_logits = outputs["presence_logits"]

    seg_loss = seg_criterion(seg_logits, labels)

    presence_targets = build_presence_targets(labels).to(
        device=presence_logits.device,
        dtype=presence_logits.dtype,
    )

    if presence_pos_weight is not None:
        presence_pos_weight = presence_pos_weight.to(
            device=presence_logits.device,
            dtype=presence_logits.dtype,
        )

    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_targets,
        pos_weight=presence_pos_weight,
    )

    total_loss = seg_loss + presence_weight * presence_loss

    return total_loss, seg_logits, presence_loss, presence_logits, presence_targets

class PresenceMetricTracker:
    """
    Slice-level metrics for auxiliary tumor/cyst presence head.

    Class 0: tumor presence
    Class 1: cyst presence
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = torch.zeros(2)
        self.fp = torch.zeros(2)
        self.fn = torch.zeros(2)
        self.tn = torch.zeros(2)
        self.total_loss = 0.0
        self.num_batches = 0

    @torch.no_grad()
    def update(self, presence_logits, presence_targets, presence_loss=None):
        """
        presence_logits:
            [B, 2]

        presence_targets:
            [B, 2]
        """

        probs = torch.sigmoid(presence_logits)
        preds = (probs >= self.threshold).float()
        targets = presence_targets.float()

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        self.tp += ((preds == 1) & (targets == 1)).sum(dim=0)
        self.fp += ((preds == 1) & (targets == 0)).sum(dim=0)
        self.fn += ((preds == 0) & (targets == 1)).sum(dim=0)
        self.tn += ((preds == 0) & (targets == 0)).sum(dim=0)

        if presence_loss is not None:
            self.total_loss += presence_loss.item()
            self.num_batches += 1

    def compute(self):
        eps = 1e-8

        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (
            self.tp + self.fp + self.fn + self.tn + eps
        )

        avg_loss = self.total_loss / max(self.num_batches, 1)

        metrics = {
            "presence_loss": avg_loss,

            "tumor_precision": precision[0].item(),
            "tumor_recall": recall[0].item(),
            "tumor_dice": dice[0].item(),
            "tumor_accuracy": accuracy[0].item(),

            "cyst_precision": precision[1].item(),
            "cyst_recall": recall[1].item(),
            "cyst_dice": dice[1].item(),
            "cyst_accuracy": accuracy[1].item(),

            "mean_presence_dice": dice.mean().item(),
            "mean_presence_recall": recall.mean().item(),
            "mean_presence_precision": precision.mean().item(),
        }

        return metrics

def segmentation_baseline(
    train_loader,
    valid_loader,
    device,
    epoch,
    lr,
    out_classes=4,
    n_numerical=4,
    n_comorbidities=1,
    save_dir="saved_BiomedCLIP_UNet_CTEHR_model",
):
    os.makedirs(save_dir, exist_ok=True)

    model = BiomedCLIPUNetCTEHRAttentionPresence(
        in_channels=1,
        out_classes=out_classes,
        biomed_embed_dim=512,
        clinical_embed_dim=64,
        n_numerical=n_numerical,
        n_comorbidities=n_comorbidities,
        num_clinical_tokens=4,
        num_heads=8,
        presence_hidden_dim=128,
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

    best_valid_loss = np.inf
    best_mean_hec_dice = -np.inf
    save_check = 0

    for i in range(epoch):
        if save_check > 50:
            print(f"Early stopping at epoch {i}")
            break

        train_loss, train_presence_metrics = train_fn(train_loader, model, optimizer, device, criterion, scaler, presence_weight=0.1)
        valid_loss, metrics, valid_presence_metrics = eval_fn(valid_loader, model, device, criterion)

        if math.isnan(valid_loss) or math.isnan(train_loss):
            print(f"Early stopping at epoch {i} by NaN")
            break

        current_mean_hec_dice = metrics["mean_hec_dice"]

        # Save by Mean HEC Dice instead of validation loss
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
            }, os.path.join(save_dir, "best_model_Attn_Presence_model.pt"))

            print("Model saved")
        else:
            save_check += 1

        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

        # ---------- TensorBoard loss ----------
        writer.add_scalar("Loss/train_total", train_loss, i)
        writer.add_scalar("Loss/valid_total", valid_loss, i)

        # ---------- TensorBoard metrics ----------
        writer.add_scalar("Metrics/mean_fg_dice", metrics["mean_fg_dice"], i)
        writer.add_scalar("Metrics/mean_fg_iou", metrics["mean_fg_iou"], i)
        writer.add_scalar("Metrics/mean_hec_dice", metrics["mean_hec_dice"], i)
        writer.add_scalar("Metrics/mean_hec_iou", metrics["mean_hec_iou"], i)

        writer.add_scalar("Metrics/kidney_dice", metrics["kidney_dice"], i)
        writer.add_scalar("Metrics/tumor_dice", metrics["tumor_dice"], i)
        writer.add_scalar("Metrics/cyst_dice", metrics["cyst_dice"], i)

        writer.add_scalar("Metrics/kidney_and_masses_dice", metrics["kidney_and_masses_dice"], i)
        writer.add_scalar("Metrics/masses_dice", metrics["masses_dice"], i)

        writer.add_scalar(
            "Presence/train_loss",
            train_presence_metrics["presence_loss"],
            i,
        )
        writer.add_scalar(
            "Presence/valid_loss",
            valid_presence_metrics["presence_loss"],
            i,
        )

        writer.add_scalar(
            "Presence/train_tumor_dice",
            train_presence_metrics["tumor_dice"],
            i,
        )
        writer.add_scalar(
            "Presence/valid_tumor_dice",
            valid_presence_metrics["tumor_dice"],
            i,
        )

        writer.add_scalar(
            "Presence/train_cyst_dice",
            train_presence_metrics["cyst_dice"],
            i,
        )
        writer.add_scalar(
            "Presence/valid_cyst_dice",
            valid_presence_metrics["cyst_dice"],
            i,
        )

        writer.add_scalar(
            "Presence/valid_tumor_recall",
            valid_presence_metrics["tumor_recall"],
            i,
        )
        writer.add_scalar(
            "Presence/valid_cyst_recall",
            valid_presence_metrics["cyst_recall"],
            i,
        )

         # ---------- Print classification metrics ----------
        print(
            f"Train Presence Loss: {train_presence_metrics['presence_loss']:.4f} | "
            f"Train Tumor Dice: {train_presence_metrics['tumor_dice']:.4f} | "
            f"Train Tumor Recall: {train_presence_metrics['tumor_recall']:.4f} | "
            f"Train Cyst Dice: {train_presence_metrics['cyst_dice']:.4f} | "
            f"Train Cyst Recall: {train_presence_metrics['cyst_recall']:.4f}"
        )

        print(
            f"Valid Presence Loss: {valid_presence_metrics['presence_loss']:.4f} | "
            f"Valid Tumor Dice: {valid_presence_metrics['tumor_dice']:.4f} | "
            f"Valid Tumor Recall: {valid_presence_metrics['tumor_recall']:.4f} | "
            f"Valid Cyst Dice: {valid_presence_metrics['cyst_dice']:.4f} | "
            f"Valid Cyst Recall: {valid_presence_metrics['cyst_recall']:.4f}"
        )

        # ---------- Print results ----------
        print(
            f"EPOCH {i + 1} : "
            f"train loss : {train_loss:.4f}, "
            f"valid loss : {valid_loss:.4f}"
        )

        print(
            f"Mean FG Dice: {metrics['mean_fg_dice']:.4f} | "
            f"Mean FG IoU: {metrics['mean_fg_iou']:.4f}"
        )

        print(
            f"Kidney Dice: {metrics['kidney_dice']:.4f} | "
            f"Kidney IoU: {metrics['kidney_iou']:.4f}"
        )

        print(
            f"Tumor Dice: {metrics['tumor_dice']:.4f} | "
            f"Tumor IoU: {metrics['tumor_iou']:.4f}"
        )

        print(
            f"Cyst Dice: {metrics['cyst_dice']:.4f} | "
            f"Cyst IoU: {metrics['cyst_iou']:.4f}"
        )

        print(
            f"K&M Dice: {metrics['kidney_and_masses_dice']:.4f} | "
            f"K&M IoU: {metrics['kidney_and_masses_iou']:.4f}"
        )

        print(
            f"Masses Dice: {metrics['masses_dice']:.4f} | "
            f"Masses IoU: {metrics['masses_iou']:.4f}"
        )

        print(
            f"Mean HEC Dice: {metrics['mean_hec_dice']:.4f} | "
            f"Mean HEC IoU: {metrics['mean_hec_iou']:.4f}"
        )

        print(f"Best Mean HEC Dice: {best_mean_hec_dice:.4f}")

    writer.flush()
    writer.close()


def train_fn(loader, model, optimizer, device, criterion, scaler=None, presence_weight=0.1):
    model.train()
    total_loss = 0.0

    presence_tracker = PresenceMetricTracker(threshold=0.5)

    for images, labels, clinical_batch in tqdm(loader):
        images = images.float().to(device)
        labels = labels.long().to(device)
        clinical_batch = move_clinical_to_device(clinical_batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, clinical_batch, return_aux=True)

                loss, seg_out, presence_loss, presence_logits, presence_targets = compute_seg_presence_loss(
                    outputs=outputs,
                    labels=labels,
                    seg_criterion=criterion,
                    presence_weight=presence_weight,
                    presence_pos_weight=None,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images, clinical_batch, return_aux=True)
            loss, seg_out, presence_loss, presence_logits, presence_targets = compute_seg_presence_loss(
                outputs=outputs,
                labels=labels,
                seg_criterion=criterion,
                presence_weight=presence_weight,
                presence_pos_weight=None,
            )
            loss.backward()
            optimizer.step()

        presence_tracker.update(
            presence_logits=presence_logits,
            presence_targets=presence_targets,
            presence_loss=presence_loss,
        )

        total_loss += loss.item()

    avg_train_loss = total_loss / len(loader)
    train_presence_metrics = presence_tracker.compute()

    return avg_train_loss, train_presence_metrics


def eval_fn(loader, model, device, criterion, presence_weight=0.1):
    model.eval()
    total_loss = 0.0

    counts = _init_counts()

    presence_tracker = PresenceMetricTracker(threshold=0.5)

    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")

    with torch.no_grad():
        for images, labels, clinical_batch in tqdm(loader):
            images = images.float().to(device)
            labels = labels.long().to(device)
            clinical_batch = move_clinical_to_device(clinical_batch, device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images, clinical_batch, return_aux=True)

                    loss, predicted, presence_loss, presence_logits, presence_targets = compute_seg_presence_loss(
                        outputs=outputs,
                        labels=labels,
                        seg_criterion=criterion,
                        presence_weight=presence_weight,
                        presence_pos_weight=None,
                    )
            else:
                outputs = model(
                    images,
                    clinical_batch,
                    return_aux=True,
                )

                loss, predicted, presence_loss, presence_logits, presence_targets = compute_seg_presence_loss(
                    outputs=outputs,
                    labels=labels,
                    seg_criterion=criterion,
                    presence_weight=presence_weight,
                    presence_pos_weight=None,
                )

            total_loss += loss.item()

            presence_tracker.update(
                presence_logits=presence_logits,
                presence_targets=presence_targets,
                presence_loss=presence_loss,
            )

            # Update validation metrics
            _update_all_counts(counts=counts, logits=predicted, labels=labels)
            

    valid_loss = total_loss / len(loader)
    metrics = _compute_dice_iou_from_counts(counts)
    presence_metrics = presence_tracker.compute()

    return valid_loss, metrics, presence_metrics