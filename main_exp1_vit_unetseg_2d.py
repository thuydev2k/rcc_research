import os
import torch
from torch.utils.data import DataLoader

from data.dataloader_exp1_vit_unetseg_2d import SegmentationDataset2D, TumorCystBatchSampler
from segmentation_exp1_vit_unetseg_2d import segmentation_exp1_vit_unetseg_2d
from inference_exp1_vit_unetseg_2d import inference_exp1_vit_unetseg_2d


SEG_DATA_DIR = 'dataset/kits23/labeled/'
OUT_CLASSES = 4

BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 120
LR = 1e-4

CLASS_WEIGHTS = (0.05, 1.0, 3.0, 5.0)
CE_WEIGHT = 0.3
DICE_WEIGHT = 0.4
HEC_WEIGHT = 0.3

EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5

def collect_paths(seg_data_dir, split):
    image_dir = os.path.join(seg_data_dir, split, 'images')
    label_dir = os.path.join(seg_data_dir, split, 'labels')

    volume_paths = []
    label_paths = []
    case_ids = []

    cases = sorted([f for f in os.listdir(image_dir) if f.endswith('.npz')])

    for case in cases:
        case_key = os.path.splitext(case)[0]
        volume_path = os.path.join(image_dir, f'{case_key}.npz')
        label_path = os.path.join(label_dir, f'{case_key}.npz')

        volume_paths.append(volume_path)
        label_paths.append(label_path)
        case_ids.append(case_key)

    print(f"{split}: found {len(volume_paths)} cases")
    return volume_paths, label_paths, case_ids


def main():
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_volume_paths, train_label_paths, train_case_ids = collect_paths(SEG_DATA_DIR, 'train')
    valid_volume_paths, valid_label_paths, valid_case_ids = collect_paths(SEG_DATA_DIR, 'valid')
    test_volume_paths, test_label_paths, test_case_ids = collect_paths(SEG_DATA_DIR, 'test')

    train_dataset = SegmentationDataset2D(
        image_paths=train_volume_paths,
        label_paths=train_label_paths,
    )

    valid_dataset = SegmentationDataset2D(
        image_paths=valid_volume_paths,
        label_paths=valid_label_paths,
    )

    test_dataset = SegmentationDataset2D(
        image_paths=test_volume_paths,
        label_paths=test_label_paths,
    )

    train_batch_sampler = TumorCystBatchSampler(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        tumor_ratio=0.30,
        cyst_ratio=0.30,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    segmentation_exp1_vit_unetseg_2d(
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        out_classes=OUT_CLASSES,
        save_dir='saved_Exp1_ViTUNetSeg2D_HEC_model',
        feature_size=16,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=12,
        patch_size=16,
        use_amp=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    inference_exp1_vit_unetseg_2d(
        test_loader=test_loader,
        test_set=test_dataset,
        device=device,
        out_classes=OUT_CLASSES,
        checkpoint_path='./saved_Exp1_ViTUNetSeg2D_HEC_model/best_model_exp1_2d.pt',
        result_dir='./result_exp1_vit_unetseg_2d_hec',
    )

if __name__ == '__main__':
    main()
