import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataloader import KiTS23SliceDataset2D, TumorCystBatchSampler

from data.clinical_dataloader import (
    KiTS23PreopClinicalDataset,
    get_training_stats,
)

from segmentation_baseline import segmentation_baseline
from inference import inference


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

SEG_DATA_DIR = Path("kits23_preprocessed_2d_all_slices")
CLINICAL_DATA_PATH = "kits23_preprocessed_2d_all_slices/kits23.json"

BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 100
LR = 1e-4

INCLUDE_RADIOGRAPHIC_SIZE = True


def extract_case_id_from_slice_filename(filename):
    """
    Example:
        case_00001_slice_00000.npz
        -> case_00001
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"Cannot extract case_id from filename: {filename}")

    return f"{parts[0]}_{parts[1]}"


def get_split_case_ids(image_dir):
    """
    Get unique case IDs from slice-level files.

    Example files:
        case_00001_slice_00000.npz
        case_00001_slice_00001.npz
        case_00002_slice_00000.npz

    Output:
        ["case_00001", "case_00002"]
    """
    image_dir = Path(image_dir)

    case_ids = set()

    for file_path in image_dir.glob("*.npz"):
        case_id = extract_case_id_from_slice_filename(file_path.name)
        case_ids.add(case_id)

    return sorted(case_ids)


def multimodal_collate_fn(batch):
    images, labels, clinical_list = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    clinical_batch = {}

    for key in clinical_list[0].keys():
        if key == "case_id":
            clinical_batch[key] = [c[key] for c in clinical_list]
        else:
            clinical_batch[key] = torch.stack(
                [c[key] for c in clinical_list],
                dim=0,
            )

    return images, labels, clinical_batch


# =========================
# Build clinical lookup
# =========================

train_image_dir = SEG_DATA_DIR / "train" / "images"
valid_image_dir = SEG_DATA_DIR / "valid" / "images"
test_image_dir = SEG_DATA_DIR / "test" / "images"

train_label_dir = SEG_DATA_DIR / "train" / "labels"
valid_label_dir = SEG_DATA_DIR / "valid" / "labels"
test_label_dir = SEG_DATA_DIR / "test" / "labels"

train_case_ids = get_split_case_ids(train_image_dir)

print(f"Number of train cases: {len(train_case_ids)}")
print(f"Example train case IDs: {train_case_ids[:5]}")


# First load clinical dataset without stats
clinical_dataset_raw = KiTS23PreopClinicalDataset(
    json_path=CLINICAL_DATA_PATH,
    stats=None,
    include_radiographic_size=INCLUDE_RADIOGRAPHIC_SIZE,
)

# Compute normalization stats only from train cases
train_stats = get_training_stats(
    json_path=CLINICAL_DATA_PATH,
    train_case_ids=train_case_ids,
    include_radiographic_size=INCLUDE_RADIOGRAPHIC_SIZE,
)

# Reload clinical dataset with train-only stats
clinical_dataset = KiTS23PreopClinicalDataset(
    json_path=CLINICAL_DATA_PATH,
    stats=train_stats,
    include_radiographic_size=INCLUDE_RADIOGRAPHIC_SIZE,
)

# Build case_id -> clinical_data dictionary
clinical_lookup = {
    case_id: clinical_dataset[idx]
    for idx, case_id in enumerate(clinical_dataset.cases)
}

print(f"Total clinical cases: {len(clinical_lookup)}")


# =========================
# Build datasets
# =========================

train_dataset = KiTS23SliceDataset2D(
    image_dir=train_image_dir,
    label_dir=train_label_dir,
    clinical_lookup=clinical_lookup,
)

valid_dataset = KiTS23SliceDataset2D(
    image_dir=valid_image_dir,
    label_dir=valid_label_dir,
    clinical_lookup=clinical_lookup,
)

test_dataset = KiTS23SliceDataset2D(
    image_dir=test_image_dir,
    label_dir=test_label_dir,
    clinical_lookup=clinical_lookup,
)


# =========================
# Build dataloaders
# =========================

train_batch_sampler = TumorCystBatchSampler(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    tumor_ratio=0.375,
    cyst_ratio=0.25,
    num_batches=None,
    seed=42,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=multimodal_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=multimodal_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=multimodal_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# =========================
# Train
# =========================

segmentation_baseline(
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    epoch=EPOCHS,
    lr=LR,
    out_classes=4,
)


# =========================
# Inference
# =========================

inference(
    test_loader=test_loader,
    test_dataset=test_dataset,
    device=device,
    out_classes=4,
    checkpoint_path="./saved_BiomedCLIP_UNet_CTEHR_SideTumor_Exp2_Dynamic_model/best_model_side_tumor_cyst_exp2_dynamic.pt",
    result_dir="./result_BiomedCLIP_UNet_CTEHR_SideTumor_Cyst_Dynamic",
    save_visuals=True,
)