import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from models.MedImageInsightUNet import MedImageInsightUNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def inference(valid_loader, valid_set, device, out_classes, medimageinsight_repo_root, config_path, checkpoint_path):
    best_model = MedImageInsightUNet(
        medimageinsight_repo_root=medimageinsight_repo_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        out_classes=out_classes,
        freeze_encoder=True,
    ).to(device)
    best_checkpoint = torch.load(f'./saved_MedImageInsight_UNet_model/best_model.pt')
    best_model.load_state_dict(best_checkpoint['model'])

    selected_class = ['background', 'kidney', 'tumor', 'cyst']
    best_model.eval()

    idx_arr = [10, 20, 50, 70, 80, 100, 120, 150, 170, 200, 250, 300, 350, 400, 450]

    with torch.no_grad():
        for i in idx_arr:
            image, label = valid_set[i]
            x_tensor = image.to(device).unsqueeze(0)

            pred_mask_logits = best_model(x_tensor)
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
    total_cm = np.zeros((out_classes, out_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in iter(valid_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            output = best_model(images)

            total_cm = update_confusion_matrix_sklearn(total_cm, output, labels, out_classes)

            dice, iou, valid_classes = DSC_IoU_EachClass_Softmax(output, labels, out_classes=out_classes)

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
    df_csv = pd.DataFrame(predictions)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    df_csv.to_csv(f"{result_dir}/prediction.csv")

    mean_dices = []
    mean_ious = []

    print(df_csv.head())
    print(df_csv.describe())

    for c in range(out_classes):
        class_name = selected_class[c]
        mean_dice = df_csv[f'{class_name.lower()}_dice'].mean()
        mean_iou = df_csv[f'{class_name.lower()}_iou'].mean()
        mean_dices.append(mean_dice)
        mean_ious.append(mean_iou)
        print(f'Class: {class_name}, Mean Dice: {mean_dice:.44f}, Mean IoU: {mean_iou:.4f}')

    print(f'AVG DSC: {np.mean(mean_dices[1:])}, AVG IoU: {np.mean(mean_ious[1:])}')

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
