import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from models.BiomedUNet import BiomedTransUNet

selected_class_rgb = [
    [0, 0, 0],          # background (black)
    [255, 255, 0],      # kidney (yellow)
    [0, 0, 255],        # tumor (blue)
    [0, 255, 0]         # cyst (green)
]

def colour_code_segmentation(image):
    colour_code = np.array(selected_class_rgb)
    x = colour_code[image.astype(int)]
    return x

def DSC_IoU_EachClass_Softmax(predicted, target, out_classes, smooth=1e-10):
    pred = torch.argmax(predicted, dim=1)

    dice_list = []
    iou_list = []
    valid_classes = []

    for c in range(out_classes):

        pred_c = (pred == c)
        target_c = (target == c)

        if target_c.sum() == 0:
            continue

        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & (~target_c)).sum().float()
        fn = ((~pred_c) & target_c).sum().float()
        # tn = ((~pred_c) & ~target_c).sum().float()

        dice_c = 2 * tp / (2 * tp + fp + fn + smooth)
        iou_c  = tp / (tp + fp + fn + smooth)

        dice_list.append(dice_c)
        iou_list.append(iou_c)
        valid_classes.append(c)

    if len(dice_list) == 0:
        return None, None, None
    
    dice_tensor = torch.stack(dice_list)
    iou_tensor  = torch.stack(iou_list)

    return dice_tensor, iou_tensor, valid_classes

def DSC_IoU_HEC_Softmax(predicted, target, smooth=1e-10):
    """
    predicted: logits, shape [B, C, H, W]
    target:    labels, shape [B, H, W]
    label map:
        0 = background
        1 = kidney
        2 = tumor
        3 = cyst
    HECs:
        - Kidney and Masses = kidney + tumor + cyst
        - Kidney Mass       = tumor + cyst
        - Tumor             = tumor
    """
    pred = torch.argmax(predicted, dim=1)  # [B, H, W]

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
            pred_region |= (pred == c)
            target_region |= (target == c)

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

def update_confusion_matrix_sklearn(total_cm, output, labels, out_classes):
    pred = torch.argmax(output, dim=1)   # [B, H, W]

    y_true = labels.detach().cpu().numpy().reshape(-1)
    y_pred = pred.detach().cpu().numpy().reshape(-1)

    valid_mask = (y_true >= 0) & (y_true < out_classes)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    cm_batch = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(out_classes)
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

def compute_hec_dataset_metrics(all_preds, all_targets, smooth=1e-10):
    """
    all_preds, all_targets: list of tensors [B, H, W] hoặc đã flatten
    Tính metric trên toàn dataset theo HEC.
    """
    hec_defs = {
        "kidney_and_masses": [1, 2, 3],
        "kidney_mass": [2, 3],
        "tumor": [2],
    }

    records = []

    pred_all = torch.cat(all_preds, dim=0)      # [N, H, W]
    target_all = torch.cat(all_targets, dim=0)  # [N, H, W]

    for region_name, class_ids in hec_defs.items():
        pred_region = torch.zeros_like(pred_all, dtype=torch.bool)
        target_region = torch.zeros_like(target_all, dtype=torch.bool)

        for c in class_ids:
            pred_region |= (pred_all == c)
            target_region |= (target_all == c)

        tp = (pred_region & target_region).sum().item()
        fp = (pred_region & (~target_region)).sum().item()
        fn = ((~pred_region) & target_region).sum().item()

        dice = (2 * tp) / (2 * tp + fp + fn + smooth)
        iou = tp / (tp + fp + fn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)

        records.append({
            "region": region_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        })

    return pd.DataFrame(records)

def inference(valid_loader, valid_set, device, out_classes):
    best_model = BiomedTransUNet(
        in_classes=1,
        out_classes=out_classes,
        embed_dim=128,
        n_clinical=17,
        biomed_embed_dim=512
    ).to(device)

    best_checkpoint = torch.load(f'./saved_BiomedCLIP_UNet_model/best_model1.pt')
    best_model.load_state_dict(best_checkpoint['model'])
    best_model.eval()

    selected_class = ['background', 'kidney', 'tumor', 'cyst']

    idx_arr = [10, 20, 50, 70, 80, 100, 120, 150, 170, 200, 250, 300, 350, 400, 450]

    with torch.no_grad():
        for i in idx_arr:
            image, label, clinical_data = valid_set[i]
            x_tensor = image.to(device).unsqueeze(0)
            
            clinical_tensor = torch.cat([
                clinical_data["numerical"].float(),
                clinical_data["comorbidities"].float(),
                clinical_data["gender"].view(1).float(),
                clinical_data["smoking_history"].view(1).float(),
                clinical_data["surgery_type"].view(1).float(),
                clinical_data["surgical_approach"].view(1).float(),
                clinical_data["tumor_histologic_subtype"].view(1).float(),
                clinical_data["pathology_t_stage"].view(1).float()
            ], dim=0)

            clinical_batch = clinical_tensor.unsqueeze(0).to(device)

            pred_mask_logits = best_model(x_tensor, clinical_batch)
            pred_mask = pred_mask_logits.detach().squeeze().cpu().numpy()

            pred_mask = np.transpose(pred_mask, (1, 2, 0))

            channel_num = image.shape[0]

            plt.figure(figsize=(5 * channel_num + 10, 5))

            HU = [1, 2, 3]
            for j in range(channel_num):
                plt.subplot(1, channel_num + 2, j + 1)
                plt.title(f'input image {HU[j]} {i}')
                plt.imshow(image[j], cmap='gray')
            
            plt.subplot(1, channel_num + 2, channel_num + 1)
            plt.title('ground-truth')
            plt.imshow(label)

            plt.subplot(1, channel_num + 2, channel_num + 2)
            plt.title("prediction")
            plt.imshow(colour_code_segmentation(np.argmax(pred_mask, axis=2)))

            if not os.path.exists(f'./result'):
                os.mkdir(f'./result')
            plt.savefig(f'./result/prediction_{i}.png')

    predictions = []
    all_preds = []
    all_targets = []
    total_cm = np.zeros((out_classes, out_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels, clinical_data in iter(valid_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            clinical_tensor = torch.cat([
                clinical_data["numerical"].float(),
                clinical_data["comorbidities"].float(),
                clinical_data["gender"].unsqueeze(1).float(),
                clinical_data["smoking_history"].unsqueeze(1).float(),
                clinical_data["surgery_type"].unsqueeze(1).float(),
                clinical_data["surgical_approach"].unsqueeze(1).float(),
                clinical_data["tumor_histologic_subtype"].unsqueeze(1).float(),
                clinical_data["pathology_t_stage"].unsqueeze(1).float()
            ], dim=1)

            clinical_batch = clinical_tensor.to(device)

            output = best_model(images, clinical_batch)

            pred = torch.argmax(output, dim=1)

            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

            batch_size = images.shape[0]
            for b in range(batch_size):
                case_output = output[b:b+1]
                case_label = labels[b:b+1]

                prediction = DSC_IoU_HEC_Softmax(case_output, case_label)
                predictions.append(prediction)

            total_cm = update_confusion_matrix_sklearn(total_cm, output, labels, out_classes)

            dice, iou = DSC_IoU_EachClass_Softmax(output, labels, out_classes=out_classes)

            prediction = {}

            for idx, class_name in enumerate(selected_class):
                prediction[f'{class_name.lower()}_dice'] = np.nan
                prediction[f'{class_name.lower()}_iou'] = np.nan

            for k, c in enumerate(valid_classes):
                class_name = selected_class[c]

                prediction[f'{class_name.lower()}_dice'] = dice[k].item()
                prediction[f'{class_name.lower()}_iou'] = iou[k].item()

            predictions.append(prediction)

    result_dir = f'./result/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    df_csv = pd.DataFrame(predictions)
    df_csv.to_csv(f"{result_dir}/prediction.csv")

    print(df_csv.head())
    print(df_csv.describe())

    mean_kam_dice = df_csv["kidney_and_masses_dice"].mean()
    mean_km_dice = df_csv["kidney_mass_dice"].mean()
    mean_tumor_dice = df_csv["tumor_dice"].mean()

    mean_kam_iou = df_csv["kidney_and_masses_iou"].mean()
    mean_km_iou = df_csv["kidney_mass_iou"].mean()
    mean_tumor_iou = df_csv["tumor_iou"].mean()

    print(f"Kidney and Masses - Mean Dice: {mean_kam_dice:.6f}, Mean IoU: {mean_kam_iou:.6f}")
    print(f"Kidney Mass       - Mean Dice: {mean_km_dice:.6f}, Mean IoU: {mean_km_iou:.6f}")
    print(f"Tumor             - Mean Dice: {mean_tumor_dice:.6f}, Mean IoU: {mean_tumor_iou:.6f}")

    print(f"AVG HEC Dice: {np.mean([mean_kam_dice, mean_km_dice, mean_tumor_dice]):.6f}")
    print(f"AVG HEC IoU : {np.mean([mean_kam_iou, mean_km_iou, mean_tumor_iou]):.6f}")

    hec_metrics_df = compute_hec_dataset_metrics(all_preds, all_targets)
    hec_metrics_df.to_csv(f"{result_dir}/metrics_hec_dataset_level.csv", index=False)

    print("\nDataset-level HEC Metrics:")
    print(hec_metrics_df)

    cm_df = pd.DataFrame(total_cm, index=selected_class, columns=selected_class)
    cm_df.to_csv(f"{result_dir}/confusion_matrix.csv")

    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    print(cm_df)

    cm_metrics_df = metrics_from_confusion_matrix(total_cm, selected_class)
    cm_metrics_df.to_csv(f"{result_dir}/metrics_from_confusion_matrix.csv", index=False)

    print("\nMetrics from Confusion Matrix:")
    print(cm_metrics_df)

    avg_dice_fg_cm = cm_metrics_df.loc[cm_metrics_df["class"] != "background", "dice_from_cm"].mean()
    avg_iou_fg_cm = cm_metrics_df.loc[cm_metrics_df["class"] != "background", "iou_from_cm"].mean()

    print(f'\nAVG DSC from CM (foreground only): {avg_dice_fg_cm:.6f}')
    print(f'AVG IoU from CM (foreground only): {avg_iou_fg_cm:.6f}')

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=selected_class)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title("Confusion Matrix (rows=GT, cols=Pred)")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/confusion_matrix.png")
    plt.close()