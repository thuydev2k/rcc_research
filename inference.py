import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from models.BiomedUNet import BiomedTransUNet
from models.clinical_encoder import ClinicalEncoder

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

def inference(valid_loader, valid_set, device, out_classes):
    best_model = BiomedTransUNet(out_classes=out_classes, n_clinical=128).to(device)
    clinical_model = ClinicalEncoder().to(device)
    best_checkpoint = torch.load(f'./saved_model/best_model.pt')
    best_model.load_state_dict(best_checkpoint['model'])
    clinical_model.load_state_dict(best_checkpoint['clinical_model'])

    selected_class = ['background', 'kidney', 'tumor', 'cyst']
    best_model.eval()

    idx_arr = np.random.randint(30, 500, size=20)

    with torch.no_grad():
        for i in idx_arr:
            image, label, clinical_data = valid_set[i]
            x_tensor = image.to(device).unsqueeze(0)
            clinical_data = {
                k: v.to(device).unsqueeze(0)
                for k, v in clinical_data.items()
            }

            clinical_emb = clinical_model(clinical_data)
            pred_mask_logits = best_model(x_tensor, clinical_emb)
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

    with torch.no_grad():
        for images, labels, clinical_data in iter(valid_loader):
            images = images.float().to(device)
            labels = labels.to(device)
            clinical_data = {k: v.to(device) for k, v in clinical_data.items()}

            clinical_emb = clinical_model(clinical_data)
            output = best_model(images, clinical_emb)

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