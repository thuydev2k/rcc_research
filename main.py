import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataloader import KiTS23SliceDataset2D, TumorCystBatchSampler
from segmentation_baseline import segmentation_baseline
from inference import inference

device = 'cuda:2'

train_dataset = KiTS23SliceDataset2D(
    image_dir="kits23_preprocessed_2d_all_slices/train/images",
    label_dir="kits23_preprocessed_2d_all_slices/train/labels",
)

valid_dataset = KiTS23SliceDataset2D(
    image_dir="kits23_preprocessed_2d_all_slices/valid/images",
    label_dir="kits23_preprocessed_2d_all_slices/valid/labels",
)

test_dataset = KiTS23SliceDataset2D(
    image_dir="kits23_preprocessed_2d_all_slices/test/images",
    label_dir="kits23_preprocessed_2d_all_slices/test/labels",
)

train_batch_sampler = TumorCystBatchSampler(
    dataset=train_dataset,
    batch_size=8,
    tumor_ratio=0.375,
    cyst_ratio=0.25,
    num_batches=None,
    seed=42,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=2,
    pin_memory=True,
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

segmentation_baseline(train_loader, valid_loader, device, 100, 1e-4, out_classes=4)
inference(test_loader, test_dataset, device, out_classes=4)