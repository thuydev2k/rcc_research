import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models.UNet3D import UNet3D


selected_class = ["background", "kidney", "tumor", "cyst"]

selected_class_rgb = [
    [0, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0],
]

def colour_code_segmentation(mask):
    colour_code = np.array(selected_class_rgb)
    return colour_code[mask.astype(int)]


def DSC_IoU_EachClass_3D(logits, target, out_classes, smooth=1e-10):
    pred = torch.argmax(logits, dim=1)

    dice_list = []
    iou_list = []
    valid_classes = []

    for c in range(out_classes):
        pred_c = pred == c
        target_c = target == c

        if target_c.sum() == 0:
            continue

        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & (~target_c)).sum().float()
        fn = ((~pred_c) & target_c).sum().float()

        dice_c = 2 * tp / (2 * tp + fp + fn + smooth)
        iou_c = tp / (tp + fp + fn + smooth)

        dice_list.append(dice_c)
        iou_list.append(iou_c)
        valid_classes.append(c)

    if len(dice_list) == 0:
        return None, None, None

    return torch.stack(dice_list), torch.stack(iou_list), valid_classes


def DSC_IoU_HEC_3D(logits, target, smooth=1e-10):
    """
    logits: [B, C, D, H, W]
    target: [B, D, H, W]
    """
    pred = torch.argmax(logits, dim=1)

    hec_defs = {
        "kidney_and_masses": [1, 2, 3],
        "kidney_mass": [2, 3],
        "tumor": [2],
    }

    result = {}

    for region_name, class_ids in hec_defs.items():
        pred_region = torch.zeros_like(pred, dtype=torch.bool)
        target_region = torch.zeros_like(target, dtype=torch.bool)

        for c in class_ids:
            pred_region |= pred == c
            target_region |= target == c

        if target_region.sum() == 0:
            result[f"{region_name}_dice"] = np.nan
            result[f"{region_name}_iou"] = np.nan
            continue

        tp = (pred_region & target_region).sum().float()
        fp = (pred_region & (~target_region)).sum().float()
        fn = ((~pred_region) & target_region).sum().float()

        dice = 2 * tp / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)

        result[f"{region_name}_dice"] = dice.item()
        result[f"{region_name}_iou"] = iou.item()

    return result


def update_confusion_matrix_sklearn_3D(total_cm, logits, labels, out_classes):
    """
    logits: [B, C, D, H, W]
    labels: [B, D, H, W]
    """
    pred = torch.argmax(logits, dim=1)

    y_true = labels.detach().cpu().numpy().reshape(-1)
    y_pred = pred.detach().cpu().numpy().reshape(-1)

    valid_mask = (y_true >= 0) & (y_true < out_classes)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    cm_batch = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(out_classes),
    )

    total_cm += cm_batch
    return total_cm

def metrics_from_confusion_matrix(cm, class_names, smooth=1e-10):
    total = cm.sum()
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    records = []

    for c, class_name in enumerate(class_names):
        tp = cm[c, c]
        fp = col_sum[c] - tp
        fn = row_sum[c] - tp
        tn = total - tp - fp - fn

        dice = (2 * tp) / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)

        records.append({
            "class": class_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice_from_cm": dice,
            "iou_from_cm": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        })

    return pd.DataFrame(records)

def compute_hec_dataset_metrics_3D(all_preds, all_targets, smooth=1e-10):
    """
    Dataset-level HEC metrics for 3D ROI inference.

    all_preds:   list of tensors, each shape [B, D, H, W]
    all_targets: list of tensors, each shape [B, D, H, W]

    Works even when each ROI has different D, H, W.
    """

    hec_defs = {
        "kidney_and_masses": [1, 2, 3],
        "kidney_mass": [2, 3],
        "tumor": [2],
    }

    records = []

    for region_name, class_ids in hec_defs.items():
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(all_preds, all_targets):
            pred = pred.cpu()
            target = target.cpu()

            pred_region = torch.zeros_like(pred, dtype=torch.bool)
            target_region = torch.zeros_like(target, dtype=torch.bool)

            for c in class_ids:
                pred_region |= pred == c
                target_region |= target == c

            tp = (pred_region & target_region).sum().item()
            fp = (pred_region & (~target_region)).sum().item()
            fn = ((~pred_region) & target_region).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

        dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + smooth)
        iou = total_tp / (total_tp + total_fp + total_fn + smooth)
        precision = total_tp / (total_tp + total_fp + smooth)
        recall = total_tp / (total_tp + total_fn + smooth)

        records.append({
            "region": region_name,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        })

    return pd.DataFrame(records)

def save_3d_prediction_visualization(image, label, pred, save_path, slice_index=None):
    """
    image: [1, D, H, W]
    label: [D, H, W]
    pred:  [D, H, W]
    """
    image_np = image.squeeze(0).cpu().numpy()
    label_np = label.cpu().numpy()
    pred_np = pred.cpu().numpy()

    D = image_np.shape[0]

    if slice_index is None:
        # choose slice with most foreground in GT
        fg_per_slice = (label_np > 0).sum(axis=(1, 2))
        slice_index = int(np.argmax(fg_per_slice))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"CT slice {slice_index}")
    plt.imshow(image_np[slice_index], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground truth")
    plt.imshow(colour_code_segmentation(label_np[slice_index]))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(colour_code_segmentation(pred_np[slice_index]))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def inference(
    valid_loader,
    valid_set,
    device,
    out_classes=4,
    checkpoint_path="./saved_UNet3D_model/best_model1.pt",
    result_dir="./result_3d",
):
    os.makedirs(result_dir, exist_ok=True)

    best_model = UNet3D(
        in_channels=1,
        out_classes=out_classes,
        base_channels=16,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_model.load_state_dict(checkpoint["model"])
    best_model.eval()

    predictions = []
    all_preds = []
    all_targets = []
    total_cm = np.zeros((out_classes, out_classes), dtype=np.int64)

    # visualization
    vis_indices = list(range(min(10, len(valid_set))))

    with torch.no_grad():
        for i in vis_indices:
            image, label = valid_set[i]

            x_tensor = image.unsqueeze(0).float().to(device)  # [1, 1, D, H, W]
            logits = best_model(x_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

            save_3d_prediction_visualization(
                image=image,
                label=label,
                pred=pred,
                save_path=os.path.join(result_dir, f"prediction_case_{i}.png"),
            )

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.float().to(device)  # [B, 1, D, H, W]
            labels = labels.long().to(device)   # [B, D, H, W]

            logits = best_model(images)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

            # HEC metrics per batch/case
            batch_size = images.shape[0]

            for b in range(batch_size):
                case_logits = logits[b:b + 1]
                case_label = labels[b:b + 1]

                hec_result = DSC_IoU_HEC_3D(case_logits, case_label)
                predictions.append(hec_result)

                dice, iou, valid_classes = DSC_IoU_EachClass_3D(
                    case_logits,
                    case_label,
                    out_classes=out_classes,
                )

                class_result = {}

                for class_name in selected_class:
                    class_result[f"{class_name}_dice"] = np.nan
                    class_result[f"{class_name}_iou"] = np.nan

                if valid_classes is not None:
                    for k, c in enumerate(valid_classes):
                        class_name = selected_class[c]
                        class_result[f"{class_name}_dice"] = dice[k].item()
                        class_result[f"{class_name}_iou"] = iou[k].item()

                predictions.append(class_result)

            total_cm = update_confusion_matrix_sklearn_3D(
                total_cm,
                logits,
                labels,
                out_classes,
            )

    df_csv = pd.DataFrame(predictions)
    df_csv.to_csv(os.path.join(result_dir, "prediction.csv"), index=False)

    print(df_csv.head())
    print(df_csv.describe())

    # HEC summary
    for region in ["kidney_and_masses", "kidney_mass", "tumor"]:
        dice_col = f"{region}_dice"
        iou_col = f"{region}_iou"

        if dice_col in df_csv.columns:
            print(
                f"{region} - "
                f"Mean Dice: {df_csv[dice_col].mean():.6f}, "
                f"Mean IoU: {df_csv[iou_col].mean():.6f}"
            )

    hec_metrics_df = compute_hec_dataset_metrics_3D(all_preds, all_targets)
    hec_metrics_df.to_csv(
        os.path.join(result_dir, "metrics_hec_dataset_level.csv"),
        index=False,
    )

    print("\nDataset-level HEC Metrics:")
    print(hec_metrics_df)

    # confusion matrix
    cm_df = pd.DataFrame(
        total_cm,
        index=selected_class,
        columns=selected_class,
    )
    cm_df.to_csv(os.path.join(result_dir, "confusion_matrix.csv"))

    print("\nConfusion Matrix rows=GT, cols=Pred:")
    print(cm_df)

    cm_metrics_df = metrics_from_confusion_matrix(total_cm, selected_class)
    cm_metrics_df.to_csv(
        os.path.join(result_dir, "metrics_from_confusion_matrix.csv"),
        index=False,
    )

    print("\nMetrics from Confusion Matrix:")
    print(cm_metrics_df)

    avg_dice_fg_cm = cm_metrics_df.loc[
        cm_metrics_df["class"] != "background",
        "dice_from_cm",
    ].mean()

    avg_iou_fg_cm = cm_metrics_df.loc[
        cm_metrics_df["class"] != "background",
        "iou_from_cm",
    ].mean()

    print(f"\nAVG DSC from CM foreground only: {avg_dice_fg_cm:.6f}")
    print(f"AVG IoU from CM foreground only: {avg_iou_fg_cm:.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=total_cm,
        display_labels=selected_class,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix rows=GT, cols=Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()