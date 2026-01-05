import os
import pandas as pd
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from preprocess import window_CT

def save_kits23_dataset():
    IMG_DATA_DIR = './dataset/kits23/train/images'
    LBL_DATA_DIR = './dataset/kits23/train/labels'

    OUT_TRAIN_IMG = "./dataset/kits23/labeled/train/images"
    OUT_TRAIN_LBL = "./dataset/kits23/labeled/train/labels"
    OUT_VALID_IMG = "./dataset/kits23/labeled/valid/images"
    OUT_VALID_LBL = "./dataset/kits23/labeled/valid/labels"

    print('image length', len(os.listdir(IMG_DATA_DIR)))
    print('label length', len(os.listdir(LBL_DATA_DIR)))

    IMAGES = []
    LABELS = []
    CASE_IDS = []

    CASES = sorted(os.listdir(IMG_DATA_DIR))
    for c in CASES:
        case_id = c.split('.')[0]
        IMAGES.append(os.path.join(IMG_DATA_DIR, case_id + '.nii.gz'))
        LABELS.append(os.path.join(LBL_DATA_DIR, case_id + '.nii.gz'))
        CASE_IDS.append(case_id)

    df_data = pd.DataFrame({'case_id': CASE_IDS, 'image': IMAGES, 'label': LABELS})

    print(f'Number of data: {len(df_data)}')

    train_df, valid_df = train_test_split(df_data, test_size=0.2, random_state=42)
    print(f'number of train patient: {len(train_df)}, number of valid patient: {len(valid_df)}')

    def process_and_save(row, out_img_dir, out_lbl_dir):
        c_id = row['case_id']
        
        obj_img = nib.load(row['image'])
        obj_lbl = nib.load(row['label'])

        img = obj_img.get_fdata(dtype=np.float32)
        lbl = obj_lbl.get_fdata(dtype=np.float32)

        sample_windowed_ct = np.asarray([window_CT(_, min=-100, max=300) for _ in img])
        processed_image = (sample_windowed_ct * 255).astype(np.uint8)

        processed_label = np.asarray([cv2.resize(_, (512, 512), interpolation=cv2.INTER_NEAREST) for _ in lbl])

        np.savez_compressed(os.path.join(out_img_dir, f"{c_id}.npz"), data=processed_image)
        np.savez_compressed(os.path.join(out_lbl_dir, f"{c_id}.npz"), data=processed_label)

        return obj_img.header.get_zooms()[2] # Return spacing for statistics

    # training
    train_zspacing = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        spacing = process_and_save(row, OUT_TRAIN_IMG, OUT_TRAIN_LBL)
        train_zspacing.append(spacing)

    np.savez_compressed("./dataset/kits23/labeled/train/train_volume_zspacing.npz", data=train_zspacing)

    # validation
    valid_zspacing = []
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        spacing = process_and_save(row, OUT_VALID_IMG, OUT_VALID_LBL)
        valid_zspacing.append(spacing)

    np.savez_compressed("./dataset/kits23/labeled/valid/valid_volume_zspacing.npz", data=valid_zspacing)

    
    # zspacing = []
    # data_image_npz_array = []
    # data_label_npz_array = []

    # for idx in range(len(train_df)):
    #     obj_img = nib.load(train_df['image'].iloc[idx])
    #     obj_lbl = nib.load(train_df['label'].iloc[idx])

    #     img = obj_img.get_fdata(dtype=np.float32)
    #     lbl = obj_lbl.get_fdata(dtype=np.float32)

    #     spacing = obj_img.header.get_zooms()[2]
    #     zspacing.append(spacing)

    #     sample_windowed_ct = np.asarray([window_CT(_, min=-100, max=300) for _ in img])
    #     sample_windowed_ct = (sample_windowed_ct * 255).astype(np.uint8)

    #     lbl = np.asarray([cv2.resize(_, (512, 512), interpolation=cv2.INTER_NEAREST) for _ in lbl])

    #     data_image_npz_array.append(sample_windowed_ct)
    #     data_label_npz_array.append(lbl)

    #     if idx % 5 == 0:
    #         print(f"saved {idx}th train index")
    
    # for i, volume in enumerate(data_image_npz_array):
    #     np.savez_compressed(f"./dataset/kits23/labeled/train/images/train_volume_{i:05d}.npz", data=volume)
    #     np.savez_compressed(f"./dataset/kits23/labeled/train/labels/train_label_{i:05d}.npz",
    #                         data=data_label_npz_array[i])
    # np.savez_compressed(f"./dataset/kits23/labeled/train/train_volume_zspacing.npz", data=zspacing)

    # zspacing = []
    # data_image_npz_array = []
    # data_label_npz_array = []


    # for idx in range(len(valid_df)):
    #     obj_img = nib.load(valid_df['image'].iloc[idx])
    #     obj_lbl = nib.load(valid_df['label'].iloc[idx])

    #     img = obj_img.get_fdata(dtype=np.float32)
    #     lbl = obj_lbl.get_fdata(dtype=np.float32)

    #     spacing = obj_img.header.get_zooms()[2]
    #     zspacing.append(spacing)

    #     sample_windowed_ct = np.asarray([window_CT(_, min=-100, max=300) for _ in img])
    #     sample_windowed_ct = (sample_windowed_ct * 255).astype(np.uint8)

    #     lbl = np.asarray([cv2.resize(_, (512, 512), interpolation=cv2.INTER_NEAREST) for _ in lbl])
    #     data_image_npz_array.append(sample_windowed_ct)
    #     data_label_npz_array.append(lbl)

    #     if idx % 5 == 0:
    #         print(f"saved {idx}th valid index")

    # for i, volume in enumerate(data_image_npz_array):
    #     np.savez_compressed(f"./dataset/kits23/labeled/valid/images/valid_volume_{i:05d}.npz",
    #                         data=volume)
    #     np.savez_compressed(f"./dataset/kits23/labeled/valid/labels/valid_label_{i:05d}.npz",
    #                         data=data_label_npz_array[i])
    # np.savez_compressed(f"./dataset/kits23/labeled/valid/valid_volume_zspacing.npz", data=zspacing)

save_kits23_dataset()