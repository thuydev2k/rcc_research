import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from models.UNet import UNet
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
    n_class = out_classes

    predicted = predicted.squeeze(0)
    predicted = torch.softmax(predicted, dim=0)
    predicted = torch.argmax(predicted, dim=0)

    target = target.squeeze(0)

    dice = torch.ones(n_class).float()
    iou = torch.ones(n_class).float()

    for i in range(n_class):
        predicted_temp = torch.eq(predicted, i)
        target_temp = torch.eq(target, i)

        intersection = (predicted_temp & target_temp).float().sum()
        total = target_temp.float().sum() + predicted_temp.float().sum()
        union = total - intersection

        dice[i] = (2 * intersection + smooth) / (total + smooth)
        iou[i] = (intersection + smooth) / (union + smooth)
    
    return dice, iou

def inference(valid_loader, valid_set, device, out_classes):
    best_model = UNet(out_classes=out_classes, n_clinical=128).to(device)
    clinical_model = ClinicalEncoder().to(device)
    best_checkpoint = torch.load(f'./saved_model/best_model.pt')
    best_model.load_state_dict(best_checkpoint['model'])
    clinical_model.load_state_dict(best_checkpoint['clinical_model'])

    selected_class = ['background', 'kidney', 'tumor', 'cyst']
    best_model.eval()

    idx_arr = np.random.randint(30, 400, size=10)

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
    all_patient_features = []

    with torch.no_grad():
        for images, labels, clinical_data in iter(valid_loader):
            images = images.float().to(device)
            labels = labels.to(device)
            clinical_data = {k: v.to(device) for k, v in clinical_data.items()}

            clinical_emb = clinical_model(clinical_data)
            output = best_model(images, clinical_emb)

            dice, iou = DSC_IoU_EachClass_Softmax(output, labels, out_classes=out_classes)

            prediction = {}
            for idx, class_name in enumerate(selected_class):
                prediction[f'{class_name.lower()}_dice'] = dice[idx].item()
                prediction[f'{class_name.lower()}_iou'] = iou[idx].item()

            predictions.append(prediction)

    result_dir = f'./result/'

    df_csv = pd.DataFrame(predictions)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    df_csv.to_csv(f"{result_dir}/prediction.csv")

    # patient_df_csv = pd.DataFrame(all_patient_features)
    # patient_df_csv.to_csv(f"{result_dir}/patient_features.csv")

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