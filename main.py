import os
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataloader import SegmentationDataset2D
from segmentation_baseline import segmentation_baseline
from inference import inference

device = 'cuda:1'

SEG_DATA_DIR = 'dataset/kits23/labeled/'

train_case_ids = [f"case_{c.split('_')[-1].split('.')[0]}" for c in os.listdir('dataset/kits23/labeled/train/images')]

seg_train_volume_paths = []
seg_train_label_paths = []

CASES = sorted(os.listdir(SEG_DATA_DIR + 'train/images'))

for case in tqdm(CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    volume_path = SEG_DATA_DIR+f'train/images/{case_key}.npz'
    label_path = SEG_DATA_DIR+f'train/labels/{case_key}.npz'
    
    seg_train_volume_paths.append(volume_path)
    seg_train_label_paths.append(label_path)

seg_valid_volume_paths = []
seg_valid_label_paths = []

VALID_CASES = sorted(os.listdir(SEG_DATA_DIR + 'valid/images'))

for case in tqdm(VALID_CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    v = SEG_DATA_DIR+f'valid/images/{case_key}.npz'
    l = SEG_DATA_DIR+f'valid/labels/{case_key}.npz'

    seg_valid_volume_paths.append(v)
    seg_valid_label_paths.append(l)

seg_inference_volume_paths = []
seg_inference_label_paths = []

INFERENCE_CASES = sorted(os.listdir(SEG_DATA_DIR + 'test/images'))

for case in tqdm(INFERENCE_CASES):
    c = case.split('.')[0].split('_')[-1]
    case_key = f"case_{c}"
    v = SEG_DATA_DIR+f'test/images/{case_key}.npz'
    l = SEG_DATA_DIR+f'test/labels/{case_key}.npz'

    seg_inference_volume_paths.append(v)
    seg_inference_label_paths.append(l)

train_seg_dataset = SegmentationDataset2D(seg_train_volume_paths, seg_train_label_paths)
valid_seg_dataset = SegmentationDataset2D(seg_valid_volume_paths, seg_valid_label_paths)
inference_seg_dataset = SegmentationDataset2D(seg_inference_volume_paths, seg_inference_label_paths)

train_seg_loader = DataLoader(train_seg_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_seg_loader = DataLoader(valid_seg_dataset, batch_size=1, shuffle=False, num_workers=0)
inference_seg_loader = DataLoader(inference_seg_dataset, batch_size=1, shuffle=False, num_workers=0)

segmentation_baseline(train_seg_loader, valid_seg_loader, device, 100, 1e-4, out_classes=4)
inference(inference_seg_loader, inference_seg_dataset, device, out_classes=4)
