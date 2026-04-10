import os
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataloader import SegmentationDataset2D
from data.clinical_dataloader import KiTS23ClinicalDataset
from segmentation_baseline import segmentation_baseline
from inference import inference

device = 'cuda:1'

SEG_DATA_DIR = 'dataset/kits23/labeled/'
CLINICAL_DATA_DIR = 'dataset/kits23.json'

clinical_dataset = KiTS23ClinicalDataset(CLINICAL_DATA_DIR, stats=None)

train_case_ids = [f"case_{c.split('_')[-1].split('.')[0]}" for c in os.listdir('dataset/kits23/labeled/train/images')]

def get_training_stats(dataset, train_ids):
    stats = {}

    numerical_keys = ["bmi", "age_at_nephrectomy", "pathologic_size", "radiographic_size", "hospitalization"]

    for key in numerical_keys:
        values = []
        for case_id in train_ids:
            if case_id in dataset.data:
                val = dataset.data[case_id].get(key)
                if val is not None:
                    values.append(val)
        
        values = np.array(values)
        stats[key] = (np.mean(values), np.std(values))

    return stats

train_stats = get_training_stats(clinical_dataset, train_case_ids)

clinical_helper = KiTS23ClinicalDataset(CLINICAL_DATA_DIR, stats=train_stats)

clinical_lookup = {case_id: clinical_helper[idx] for idx, case_id in enumerate(clinical_helper.cases)}

seg_train_volume_paths = []
seg_train_label_paths = []
seg_train_clinicals = []

CASES = sorted(os.listdir(SEG_DATA_DIR + 'train/images'))

for case in tqdm(CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    volume_path = SEG_DATA_DIR+f'train/images/{case_key}.npz'
    label_path = SEG_DATA_DIR+f'train/labels/{case_key}.npz'
    
    if case_key in clinical_lookup:
        clinical_data = clinical_lookup[case_key]

        seg_train_volume_paths.append(volume_path)
        seg_train_label_paths.append(label_path)
        seg_train_clinicals.append(clinical_data)

seg_valid_volume_paths = []
seg_valid_label_paths = []
seg_valid_clinicals = []

VALID_CASES = sorted(os.listdir(SEG_DATA_DIR + 'valid/images'))

for case in tqdm(VALID_CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    v = SEG_DATA_DIR+f'valid/images/{case_key}.npz'
    l = SEG_DATA_DIR+f'valid/labels/{case_key}.npz'
    if (case_key in clinical_lookup):
        clinical_data = clinical_lookup[case_key]

        seg_valid_volume_paths.append(v)
        seg_valid_label_paths.append(l)
        seg_valid_clinicals.append(clinical_data)

seg_inference_volume_paths = []
seg_inference_label_paths = []
seg_inference_clinicals = []

INFERENCE_CASES = sorted(os.listdir(SEG_DATA_DIR + 'test/images'))

for case in tqdm(INFERENCE_CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    v = SEG_DATA_DIR+f'test/images/{case_key}.npz'
    l = SEG_DATA_DIR+f'test/labels/{case_key}.npz'
    if (case_key in clinical_lookup):
        clinical_data = clinical_lookup[case_key]

        seg_inference_volume_paths.append(v)
        seg_inference_label_paths.append(l)
        seg_inference_clinicals.append(clinical_data)

def multimodal_collate_fn(batch):
    images, labels, clinical_list = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    clinical_batch = {}
    for key in clinical_list[0].keys():
        clinical_batch[key] = torch.stack(
            [c[key] for c in clinical_list], dim=0
        )

    return images, labels, clinical_batch

train_seg_dataset = SegmentationDataset2D(seg_train_volume_paths, seg_train_label_paths, seg_train_clinicals)
valid_seg_dataset = SegmentationDataset2D(seg_valid_volume_paths, seg_valid_label_paths, seg_valid_clinicals)
inference_seg_dataset = SegmentationDataset2D(seg_inference_volume_paths, seg_inference_label_paths)

train_seg_loader = DataLoader(train_seg_dataset, batch_size=4, shuffle=True, collate_fn=multimodal_collate_fn, num_workers=0)
valid_seg_loader = DataLoader(valid_seg_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=0)
inference_seg_loader = DataLoader(inference_seg_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=0)

segmentation_baseline(train_seg_loader, valid_seg_loader, device, 100, 1e-4, out_classes=4, n_clinical=128)
inference(inference_seg_loader, inference_seg_dataset, device, out_classes=4)
