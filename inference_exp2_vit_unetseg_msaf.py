import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from models.ViT_UNetSeg3D_MSAF import ViTUNetSeg3D_MSAF
from inference_exp1_vit_unetseg import (
    selected_class,
    DSC_IoU_EachClass_3D,
    DSC_IoU_HEC_3D,
    update_confusion_matrix_sklearn_3D,
    metrics_from_confusion_matrix,
    compute_hec_dataset_metrics_3D,
    save_3d_prediction_visualization,
)


def inference_exp2_vit_unetseg_msaf(
    valid_loader,
    valid_set,
    device,
    out_classes=4,
    checkpoint_path='./saved_Exp2_ViTUNetSeg3D_MSAF_model/best_model_exp2.pt',
    result_dir='./result_exp2_vit_unetseg_msaf',
):
    os.makedirs(result_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ViTUNetSeg3D_MSAF(
        clinical_dim=checkpoint['clinical_dim'],
        in_channels=1,
        out_channels=checkpoint.get('out_classes', out_classes),
        img_size=checkpoint['roi_size'],
        feature_size=checkpoint.get('feature_size', 16),
        hidden_size=checkpoint.get('hidden_size', 384),
        mlp_dim=checkpoint.get('mlp_dim', 1536),
        num_heads=checkpoint.get('num_heads', 12),
        patch_size=checkpoint.get('patch_size', 16),
        num_clinical_tokens=checkpoint.get('num_clinical_tokens', 1),
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    predictions = []
    all_preds = []
    all_targets = []
    total_cm = np.zeros((out_classes, out_classes), dtype=np.int64)

    vis_indices = list(range(min(10, len(valid_set))))
    with torch.no_grad():
        for i in vis_indices:
            image, label, clinical = valid_set[i]
            x_tensor = image.unsqueeze(0).float().to(device)
            clinical_tensor = clinical.unsqueeze(0).float().to(device)

            logits = model(x_tensor, clinical_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

            save_3d_prediction_visualization(
                image=image,
                label=label,
                pred=pred,
                save_path=os.path.join(result_dir, f'prediction_case_{i}.png'),
            )

    with torch.no_grad():
        for images, labels, clinical in valid_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)
            clinical = clinical.float().to(device)

            logits = model(images, clinical)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

            batch_size = images.shape[0]
            for b in range(batch_size):
                case_logits = logits[b:b + 1]
                case_label = labels[b:b + 1]

                row = {}
                row.update(DSC_IoU_HEC_3D(case_logits, case_label))

                dice, iou, valid_classes = DSC_IoU_EachClass_3D(
                    case_logits,
                    case_label,
                    out_classes=out_classes,
                )

                for class_name in selected_class:
                    row[f'{class_name}_dice'] = np.nan
                    row[f'{class_name}_iou'] = np.nan

                if valid_classes is not None:
                    for k, c in enumerate(valid_classes):
                        class_name = selected_class[c]
                        row[f'{class_name}_dice'] = dice[k].item()
                        row[f'{class_name}_iou'] = iou[k].item()

                predictions.append(row)

            total_cm = update_confusion_matrix_sklearn_3D(total_cm, logits, labels, out_classes)

    df_csv = pd.DataFrame(predictions)
    df_csv.to_csv(os.path.join(result_dir, 'prediction.csv'), index=False)

    print(df_csv.head())
    print(df_csv.describe())

    hec_metrics_df = compute_hec_dataset_metrics_3D(all_preds, all_targets)
    hec_metrics_df.to_csv(os.path.join(result_dir, 'metrics_hec_dataset_level.csv'), index=False)

    print('\nDataset-level HEC Metrics:')
    print(hec_metrics_df)

    cm_df = pd.DataFrame(total_cm, index=selected_class, columns=selected_class)
    cm_df.to_csv(os.path.join(result_dir, 'confusion_matrix.csv'))

    print('\nConfusion Matrix rows=GT, cols=Pred:')
    print(cm_df)

    cm_metrics_df = metrics_from_confusion_matrix(total_cm, selected_class)
    cm_metrics_df.to_csv(os.path.join(result_dir, 'metrics_from_confusion_matrix.csv'), index=False)

    print('\nMetrics from Confusion Matrix:')
    print(cm_metrics_df)

    avg_dice_fg_cm = cm_metrics_df.loc[cm_metrics_df['class'] != 'background', 'dice_from_cm'].mean()
    avg_iou_fg_cm = cm_metrics_df.loc[cm_metrics_df['class'] != 'background', 'iou_from_cm'].mean()

    print(f'\nAVG DSC from CM foreground only: {avg_dice_fg_cm:.6f}')
    print(f'AVG IoU from CM foreground only: {avg_iou_fg_cm:.6f}')

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=selected_class)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title('Confusion Matrix rows=GT, cols=Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()
