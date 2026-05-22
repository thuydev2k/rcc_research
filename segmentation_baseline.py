import os
import math
import gc
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.TransUNet_Lite import TransUNet_Lite
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss2D


def _safe_div(numerator, denominator, eps=1e-10):
    return float((numerator + eps) / (denominator + eps))


def _update_region_counts(counts, name, pred_mask, target_mask):
    tp = torch.logical_and(pred_mask, target_mask).sum().item()
    fp = torch.logical_and(pred_mask, ~target_mask).sum().item()
    fn = torch.logical_and(~pred_mask, target_mask).sum().item()

    counts[name]["tp"] += tp
    counts[name]["fp"] += fp
    counts[name]["fn"] += fn


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
    out_classes,
    stop_training=False,
):
    model = TransUNet_Lite(out_classes=out_classes).to(device)
    scaler = torch.amp.GradScaler(device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=lr * 0.01,
    )

    criterion = SoftDiceCrossEntropyLoss2D(
        ce_weight=0.5,
        dice_weight=0.5,
    ).to(device)

    os.makedirs("saved_TransUNet_Lite_model", exist_ok=True)

    start_epoch = 0
    best_mean_hec_dice = -1.0

    if stop_training:
        start_epoch = 6
        model, optimizer, scheduler, scaler = set_weights(
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        best_mean_hec_dice = 0.0

    writer = SummaryWriter()

    save_check = 0

    for i in range(start_epoch, epoch):
        if save_check > 20:
            print(f"Early stopping at epoch {i}")
            break

        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
        )

        valid_loss, metrics = eval_fn(
            valid_loader,
            model,
            device,
            criterion,
        )

        if math.isnan(valid_loss) or math.isnan(train_loss):
            print(f"Early stopping at epoch {i} by nan")
            break

        current_mean_hec_dice = metrics["mean_hec_dice"]

        if current_mean_hec_dice > best_mean_hec_dice:
            save_check = 0
            best_mean_hec_dice = current_mean_hec_dice

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "lrscheduler": scheduler.state_dict(),
                    "best_mean_hec_dice": best_mean_hec_dice,
                    "epoch": i,
                },
                "saved_TransUNet_Lite_model/best_model.pt",
            )

            print("Model Saved")
        else:
            save_check += 1

        scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

        writer.add_scalar("/Loss/train_total", train_loss, i)
        writer.add_scalar("/Loss/valid_total", valid_loss, i)

        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"/Metrics/{metric_name}", metric_value, i)

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


def train_fn(loader, model, optimizer, device, criterion, scaler):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device):
            seg_out = model(images)
            loss = criterion(seg_out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_fn(loader, model, device, criterion):
    model.eval()
    total_loss = 0.0

    counts = {
        "kidney": {"tp": 0, "fp": 0, "fn": 0},
        "tumor": {"tp": 0, "fp": 0, "fn": 0},
        "cyst": {"tp": 0, "fp": 0, "fn": 0},
        "kidney_and_masses": {"tp": 0, "fp": 0, "fn": 0},
        "masses": {"tp": 0, "fp": 0, "fn": 0},
    }

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device):
                predicted = model(images)
                loss = criterion(predicted, labels)

            total_loss += loss.item()

            pred_class = torch.argmax(predicted, dim=1)

            # Class-wise masks
            pred_kidney = pred_class == 1
            gt_kidney = labels == 1

            pred_tumor = pred_class == 2
            gt_tumor = labels == 2

            pred_cyst = pred_class == 3
            gt_cyst = labels == 3

            # HEC masks
            # Kidney and masses = kidney + tumor + cyst
            pred_kidney_and_masses = pred_class > 0
            gt_kidney_and_masses = labels > 0

            # Masses = tumor + cyst
            pred_masses = torch.logical_or(pred_class == 2, pred_class == 3)
            gt_masses = torch.logical_or(labels == 2, labels == 3)

            _update_region_counts(
                counts,
                "kidney",
                pred_kidney,
                gt_kidney,
            )

            _update_region_counts(
                counts,
                "tumor",
                pred_tumor,
                gt_tumor,
            )

            _update_region_counts(
                counts,
                "cyst",
                pred_cyst,
                gt_cyst,
            )

            _update_region_counts(
                counts,
                "kidney_and_masses",
                pred_kidney_and_masses,
                gt_kidney_and_masses,
            )

            _update_region_counts(
                counts,
                "masses",
                pred_masses,
                gt_masses,
            )

    valid_loss = total_loss / len(loader)
    metrics = _compute_dice_iou_from_counts(counts)

    return valid_loss, metrics


def set_weights(model, optimizer, lr_scheduler, scaler, device):
    checkpoint = torch.load(
        "./saved_TransUNet_Lite_model/best_model.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lrscheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    return model, optimizer, lr_scheduler, scaler