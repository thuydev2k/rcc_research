import gc
import math
import os

import numpy as np
import torch
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

    model = BiomedCLIPUNetCTEHRAttention(
        in_channels=1,
        out_classes=out_classes,
        biomed_embed_dim=512,
        clinical_embed_dim=64,
        n_numerical=n_numerical,
        n_comorbidities=n_comorbidities,
        num_clinical_tokens=4,
        num_heads=8,
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

        train_loss = train_fn(train_loader, model, optimizer, device, criterion, scaler)
        valid_loss, metrics = eval_fn(valid_loader, model, device, criterion)

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
            }, os.path.join(save_dir, "best_model_exp2_film.pt"))

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


def train_fn(loader, model, optimizer, device, criterion, scaler=None):
    model.train()
    total_loss = 0.0

    for images, labels, clinical_batch in tqdm(loader):
        images = images.float().to(device)
        labels = labels.long().to(device)
        clinical_batch = move_clinical_to_device(clinical_batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                seg_out = model(images, clinical_batch)
                loss = criterion(seg_out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            seg_out = model(images, clinical_batch)
            loss = criterion(seg_out, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_fn(loader, model, device, criterion):
    model.eval()
    total_loss = 0.0

    counts = _init_counts()

    with torch.no_grad():
        for images, labels, clinical_batch in tqdm(loader):
            images = images.float().to(device)
            labels = labels.long().to(device)
            clinical_batch = move_clinical_to_device(clinical_batch, device)

            if str(device).startswith("cuda"):
                with torch.amp.autocast(device_type="cuda"):
                    predicted = model(images, clinical_batch)
                    loss = criterion(predicted, labels)
            else:
                predicted = model(images, clinical_batch)
                loss = criterion(predicted, labels)

            total_loss += loss.item()

            # Update validation metrics
            _update_all_counts(counts=counts, logits=predicted, labels=labels)

    valid_loss = total_loss / len(loader)
    metrics = _compute_dice_iou_from_counts(counts)

    return valid_loss, metrics