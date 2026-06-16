import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay

from models.BiomedUNet_CTEHR_CrossAttention import BiomedCLIPUNetCTEHRAttention


CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]

CLASS_RGB = np.array([
    [0, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0],
], dtype=np.uint8)

HEC_DEFS = {
    "kidney_and_masses": [1, 2, 3],
    "masses": [2, 3],
    "tumor": [2],
}

SIDE_CLASS_NAMES = [
    "left_tumor",
    "left_cyst",
    "right_tumor",
    "right_cyst",
]


def build_side_presence_targets(labels):
    """
    Build side-aware tumor/cyst presence targets from segmentation labels.

    labels:
        [B, H, W]

    Output:
        [B, 4]

    Order:
        0 = image-left tumor presence
        1 = image-left cyst presence
        2 = image-right tumor presence
        3 = image-right cyst presence
    """

    B, H, W = labels.shape
    mid = W // 2

    left_half = labels[:, :, :mid]
    right_half = labels[:, :, mid:]

    left_tumor = (left_half == 2).any(dim=(1, 2)).float()
    left_cyst = (left_half == 3).any(dim=(1, 2)).float()

    right_tumor = (right_half == 2).any(dim=(1, 2)).float()
    right_cyst = (right_half == 3).any(dim=(1, 2)).float()

    targets = torch.stack(
        [
            left_tumor,
            left_cyst,
            right_tumor,
            right_cyst,
        ],
        dim=1,
    )

    return targets


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


def colour_code_segmentation(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.int64)
    return CLASS_RGB[mask]


def update_class_confusion_matrix(total_cm, pred, target, num_classes):
    y_true = target.detach().cpu().numpy().astype(np.int64).reshape(-1)
    y_pred = pred.detach().cpu().numpy().astype(np.int64).reshape(-1)

    valid_mask = (
        (y_true >= 0) & (y_true < num_classes) &
        (y_pred >= 0) & (y_pred < num_classes)
    )

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    indices = num_classes * y_true + y_pred
    cm_batch = np.bincount(indices, minlength=num_classes * num_classes)
    cm_batch = cm_batch.reshape(num_classes, num_classes)

    total_cm += cm_batch
    return total_cm


def class_metrics_from_confusion_matrix(cm, class_names, smooth=1e-10):
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
            "class_id": c,
            "class": class_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        })

    df = pd.DataFrame(records)
    mean_metrics = {
        "foreground_mean_dice": float(df["dice"].mean()),
        "foreground_mean_iou": float(df["iou"].mean()),
    }
    return df, mean_metrics


def init_hec_counts():
    return {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in HEC_DEFS}


def update_hec_counts(counts, pred, target):
    for region_name, class_ids in HEC_DEFS.items():
        pred_region = torch.zeros_like(pred, dtype=torch.bool)
        target_region = torch.zeros_like(target, dtype=torch.bool)

        for c in class_ids:
            pred_region |= (pred == c)
            target_region |= (target == c)

        counts[region_name]["tp"] += int(torch.logical_and(pred_region, target_region).sum().item())
        counts[region_name]["fp"] += int(torch.logical_and(pred_region, ~target_region).sum().item())
        counts[region_name]["fn"] += int(torch.logical_and(~pred_region, target_region).sum().item())
        counts[region_name]["tn"] += int(torch.logical_and(~pred_region, ~target_region).sum().item())

    return counts


def hec_metrics_from_counts(counts, smooth=1e-10):
    records = []
    for region_name, c in counts.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        dice = (2 * tp) / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)

        records.append({
            "region": region_name,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
        })

    df = pd.DataFrame(records)
    mean_metrics = {
        "mean_hec_dice": float(df["dice"].mean()),
        "mean_hec_iou": float(df["iou"].mean()),
    }
    return df, mean_metrics

class SidePresenceMetricTracker:
    """
    Test-time metrics for side-aware multi-label classifier.

    Classes:
        0 = left_tumor
        1 = left_cyst
        2 = right_tumor
        3 = right_cyst
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = torch.zeros(4)
        self.fp = torch.zeros(4)
        self.fn = torch.zeros(4)
        self.tn = torch.zeros(4)

    @torch.no_grad()
    def update(self, side_logits, side_targets):
        probs = torch.sigmoid(side_logits)
        preds = (probs >= self.threshold).float()
        targets = side_targets.float()

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        self.tp += ((preds == 1) & (targets == 1)).sum(dim=0)
        self.fp += ((preds == 1) & (targets == 0)).sum(dim=0)
        self.fn += ((preds == 0) & (targets == 1)).sum(dim=0)
        self.tn += ((preds == 0) & (targets == 0)).sum(dim=0)

    def compute(self):
        eps = 1e-8

        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (
            self.tp + self.fp + self.fn + self.tn + eps
        )

        true_positive_rate = (self.tp + self.fn) / (
            self.tp + self.fp + self.fn + self.tn + eps
        )

        predicted_positive_rate = (self.tp + self.fp) / (
            self.tp + self.fp + self.fn + self.tn + eps
        )

        records = []

        for idx, class_name in enumerate(SIDE_CLASS_NAMES):
            records.append({
                "class_id": idx,
                "class": class_name,
                "tp": int(self.tp[idx].item()),
                "fp": int(self.fp[idx].item()),
                "fn": int(self.fn[idx].item()),
                "tn": int(self.tn[idx].item()),
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "f1": float(f1[idx].item()),
                "accuracy": float(accuracy[idx].item()),
                "true_positive_rate": float(true_positive_rate[idx].item()),
                "predicted_positive_rate": float(predicted_positive_rate[idx].item()),
            })

        df = pd.DataFrame(records)

        mean_metrics = {
            "mean_f1": float(df["f1"].mean()),
            "mean_precision": float(df["precision"].mean()),
            "mean_recall": float(df["recall"].mean()),
            "mean_accuracy": float(df["accuracy"].mean()),
        }

        return df, mean_metrics

    def save_confusion_matrices(self, result_dir):
        summary_rows = []

        for idx, class_name in enumerate(SIDE_CLASS_NAMES):
            binary_cm = np.array([
                [int(self.tn[idx].item()), int(self.fp[idx].item())],
                [int(self.fn[idx].item()), int(self.tp[idx].item())],
            ], dtype=np.int64)

            cm_df = pd.DataFrame(
                binary_cm,
                index=[f"GT_not_{class_name}", f"GT_{class_name}"],
                columns=[f"Pred_not_{class_name}", f"Pred_{class_name}"],
            )

            cm_path = os.path.join(
                result_dir,
                f"confusion_matrix_side_classifier_{class_name}.csv",
            )

            cm_df.to_csv(cm_path)

            save_confusion_matrix_plot(
                binary_cm,
                labels=[f"not_{class_name}", class_name],
                title=f"Side Classifier Confusion Matrix: {class_name}",
                save_path=os.path.join(
                    result_dir,
                    f"confusion_matrix_side_classifier_{class_name}.png",
                ),
            )

            summary_rows.append({
                "class": class_name,
                "tn": int(self.tn[idx].item()),
                "fp": int(self.fp[idx].item()),
                "fn": int(self.fn[idx].item()),
                "tp": int(self.tp[idx].item()),
            })

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(result_dir, "confusion_matrix_side_classifier_summary.csv"),
            index=False,
        )

def save_confusion_matrix_plot(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_hec_confusion_matrices(hec_counts, result_dir):
    summary_rows = []
    for region_name, c in hec_counts.items():
        binary_cm = np.array([
            [c["tn"], c["fp"]],
            [c["fn"], c["tp"]],
        ], dtype=np.int64)

        cm_df = pd.DataFrame(
            binary_cm,
            index=[f"GT_not_{region_name}", f"GT_{region_name}"],
            columns=[f"Pred_not_{region_name}", f"Pred_{region_name}"],
        )
        cm_df.to_csv(os.path.join(result_dir, f"confusion_matrix_hec_{region_name}.csv"))

        save_confusion_matrix_plot(
            binary_cm,
            labels=[f"not_{region_name}", region_name],
            title=f"HEC Confusion Matrix: {region_name}",
            save_path=os.path.join(result_dir, f"confusion_matrix_hec_{region_name}.png"),
        )

        summary_rows.append({
            "region": region_name,
            "tn": int(c["tn"]),
            "fp": int(c["fp"]),
            "fn": int(c["fn"]),
            "tp": int(c["tp"]),
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(result_dir, "confusion_matrix_hec_summary.csv"),
        index=False,
    )

def find_tumor_cyst_slice_indices(dataset):
    tumor_indices = []
    cyst_indices = []
    both_indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]

        # If dataset returns (image, label)
        image, label = sample[0], sample[1]

        if torch.is_tensor(label):
            label_np = label.cpu().numpy()
        else:
            label_np = label

        has_tumor = np.any(label_np == 2)
        has_cyst = np.any(label_np == 3)

        if has_tumor and has_cyst:
            both_indices.append(idx)
        elif has_tumor:
            tumor_indices.append(idx)
        elif has_cyst:
            cyst_indices.append(idx)

    print("Found lesion slices:")
    print(f"Tumor-only slices: {len(tumor_indices)}")
    print(f"Cyst-only slices:  {len(cyst_indices)}")
    print(f"Tumor+cyst slices: {len(both_indices)}")

    selected_indices = (
        both_indices[:30]
        + tumor_indices[:30]
        + cyst_indices[:30]
    )

    print(f"Selected visualization slices: {len(selected_indices)}")
    print(selected_indices)

    return selected_indices

def binary_dice(pred_mask, gt_mask, class_id, smooth=1e-10):
    pred_c = pred_mask == class_id
    gt_c = gt_mask == class_id

    if gt_c.sum() == 0:
        return np.nan

    tp = np.logical_and(pred_c, gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()
    fn = np.logical_and(~pred_c, gt_c).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    return dice


def visualize_samples(model, dataset, device, result_dir, sample_indices=None):
    if sample_indices is None:
        sample_indices = find_tumor_cyst_slice_indices(
            dataset,
        )

    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx in sample_indices:
            if idx >= len(dataset):
                continue

            image, label, clinical_data = dataset[idx]
            x = image.float().unsqueeze(0).to(device)
            clinical_batch = {}
            for key, value in clinical_data.items():
                if key == "case_id":
                    clinical_batch[key] = [value]
                elif torch.is_tensor(value):
                    clinical_batch[key] = value.unsqueeze(0).to(device)

            outputs = model(x, clinical_batch)
            logits = outputs["out"]
            side_logits = outputs["side_logits"]

            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            image_np = image.squeeze(0).cpu().numpy()
            label_np = label.cpu().numpy()

            side_info = ""

            if side_logits is not None:
                side_probs = torch.sigmoid(side_logits).squeeze(0).detach().cpu().numpy()
                side_info = (
                    f"\nLT:{side_probs[0]:.2f} LC:{side_probs[1]:.2f} "
                    f"RT:{side_probs[2]:.2f} RC:{side_probs[3]:.2f}"
                )

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title(f"Input CT slice {idx}")
            plt.imshow(image_np, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground truth")
            plt.imshow(colour_code_segmentation(label_np))
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Prediction{side_info}")
            plt.imshow(colour_code_segmentation(pred))
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"prediction_{idx}.png"))
            plt.close()


def inference(
    test_loader,
    test_dataset,
    device,
    out_classes=4,
    checkpoint_path="./saved_BiomedCLIP_UNet_CTEHR_SideContext_Exp1_model/best_model_side_context_exp1.pt",
    result_dir="./result_BiomedCLIP_UNet_CTEHR_SideContext_Exp1",
    save_visuals=True,
    n_numerical=4,
    n_comorbidities=1,
):
    os.makedirs(result_dir, exist_ok=True)

    model = BiomedCLIPUNetCTEHRAttention(
        in_channels=1,
        out_classes=out_classes,
        biomed_embed_dim=512,
        clinical_embed_dim=64,
        n_numerical=n_numerical,
        n_comorbidities=n_comorbidities,
        num_clinical_tokens=4,
        num_heads=8,

        # New side-aware context-token settings
        side_hidden_dim=128,
        num_context_tokens=5,
        context_dim=128,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Loaded checkpoint:", checkpoint_path)
        print("Loaded epoch:", checkpoint.get("epoch", "N/A"))
        print("Best valid metric:", checkpoint.get("best_valid_loss", checkpoint.get("best_mean_hec_dice", "N/A")))
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights:", checkpoint_path)

    model.eval()

    if save_visuals:
        visualize_samples(model, test_dataset, device, result_dir)

    class_cm = np.zeros((out_classes, out_classes), dtype=np.int64)
    hec_counts = init_hec_counts()

    side_tracker = SidePresenceMetricTracker(threshold=0.5)

    with torch.no_grad():
        for images, labels, clinical_data in tqdm(
            test_loader,
            desc="Running side-aware context inference",
        ):
            images = images.float().to(device)
            labels = labels.long().to(device)
            clinical_data = move_clinical_to_device(clinical_data, device)

            outputs = model(
                images,
                clinical_data,
            )

            if isinstance(outputs, dict):
                logits = outputs["out"]
                side_logits = outputs["side_logits"]
            else:
                raise RuntimeError(
                    "Expected model output to be a dict with keys "
                    "'out' and 'side_logits'. "
                    "Please call the side-aware context model with return_aux=True."
                )

            pred = torch.argmax(logits, dim=1)

            class_cm = update_class_confusion_matrix(
                class_cm,
                pred,
                labels,
                out_classes,
            )

            hec_counts = update_hec_counts(
                hec_counts,
                pred,
                labels,
            )

            side_targets = build_side_presence_targets(labels).to(
                device=side_logits.device,
                dtype=side_logits.dtype,
            )

            side_tracker.update(
                side_logits=side_logits,
                side_targets=side_targets,
            )

    class_metrics_df, class_mean_metrics = class_metrics_from_confusion_matrix(class_cm, CLASS_NAMES)
    hec_metrics_df, hec_mean_metrics = hec_metrics_from_counts(hec_counts)
    side_metrics_df, side_mean_metrics = side_tracker.compute()

    class_metrics_path = os.path.join(result_dir, "metrics_classwise.csv")
    hec_metrics_path = os.path.join(result_dir, "metrics_hec.csv")
    side_metrics_path = os.path.join(result_dir, "metrics_side_classifier.csv")

    class_metrics_df.to_csv(class_metrics_path, index=False)
    hec_metrics_df.to_csv(hec_metrics_path, index=False)
    side_metrics_df.to_csv(side_metrics_path, index=False)

    class_cm_df = pd.DataFrame(
        class_cm,
        index=[f"GT_{x}" for x in CLASS_NAMES],
        columns=[f"Pred_{x}" for x in CLASS_NAMES],
    )
    class_cm_df.to_csv(os.path.join(result_dir, "confusion_matrix_classwise.csv"))

    save_confusion_matrix_plot(
        class_cm,
        labels=CLASS_NAMES,
        title="Class-wise Confusion Matrix",
        save_path=os.path.join(result_dir, "confusion_matrix_classwise.png"),
    )
    save_hec_confusion_matrices(hec_counts, result_dir)
    side_metrics_df.to_csv(side_metrics_path, index=False)

    print("\n==============================")
    print("CLASS-WISE METRICS")
    print("==============================")
    for _, row in class_metrics_df.iterrows():
        print(f"{row['class']:>6} | Dice: {row['dice']:.6f} | IoU: {row['iou']:.6f}")
    print(
        f"\nForeground Mean | Dice: {class_mean_metrics['foreground_mean_dice']:.6f} | "
        f"IoU: {class_mean_metrics['foreground_mean_iou']:.6f}"
    )

    print("\n==============================")
    print("HEC METRICS")
    print("==============================")
    for _, row in hec_metrics_df.iterrows():
        print(f"{row['region']:>18} | Dice: {row['dice']:.6f} | IoU: {row['iou']:.6f}")
    print(f"\nMean HEC | Dice: {hec_mean_metrics['mean_hec_dice']:.6f} | IoU: {hec_mean_metrics['mean_hec_iou']:.6f}")

    print("\nSaved files:")
    print(f"- {class_metrics_path}")
    print(f"- {hec_metrics_path}")
    print(f"- {os.path.join(result_dir, 'confusion_matrix_classwise.csv')}")
    print(f"- {os.path.join(result_dir, 'confusion_matrix_classwise.png')}")
    print(f"- {os.path.join(result_dir, 'confusion_matrix_hec_summary.csv')}")

    print("\n==============================")
    print("SIDE-AWARE CLASSIFIER METRICS")
    print("==============================")

    for _, row in side_metrics_df.iterrows():
        print(
            f"{row['class']:>12} | "
            f"F1: {row['f1']:.6f} | "
            f"Precision: {row['precision']:.6f} | "
            f"Recall: {row['recall']:.6f} | "
            f"Accuracy: {row['accuracy']:.6f} | "
            f"GT positive rate: {row['true_positive_rate']:.6f} | "
            f"Pred positive rate: {row['predicted_positive_rate']:.6f}"
        )

    print(
        f"\nSide Classifier Mean | "
        f"F1: {side_mean_metrics['mean_f1']:.6f} | "
        f"Precision: {side_mean_metrics['mean_precision']:.6f} | "
        f"Recall: {side_mean_metrics['mean_recall']:.6f} | "
        f"Accuracy: {side_mean_metrics['mean_accuracy']:.6f}"
    )

    print(f"- {side_metrics_path}")
    print(f"- {os.path.join(result_dir, 'confusion_matrix_side_classifier_summary.csv')}")