import torch
from torch.utils.data import DataLoader

from data.dataloader_exp2_vit_unetseg_msaf import (
    SegmentationClinicalDataset3DViT,
    collect_paths,
)
from data.clinical_json_exp2 import KiTS23ClinicalJSONEncoder
from segmentation_exp2_vit_unetseg_msaf import segmentation_exp2_vit_unetseg_msaf
from inference_exp2_vit_unetseg_msaf import inference_exp2_vit_unetseg_msaf


SEG_DATA_DIR = 'dataset/kits23/labeled/'
CLINICAL_JSON = 'dataset/kits23.json'

OUT_CLASSES = 4
ROI_SIZE = (128, 256, 256)

# Your saved dataset was windowed to HU [-100, 300] but not normalized.
NORMALIZE = True
WINDOW_MIN = -100.0
WINDOW_MAX = 300.0

BATCH_SIZE = 1
NUM_WORKERS = 4
EPOCHS = 100
LR = 1e-4

# If Exp1 is still training, keep this as None.
# After Exp1 finishes, set it to './saved_Exp1_ViTUNetSeg3D_model/best_model_exp1.pt'
INIT_FROM_EXP1 = None

RUN_TEST_INFERENCE = True


# Use this if you want to force one GPU, e.g. GPU_ID = 1.
# Otherwise keep None and use CUDA_VISIBLE_DEVICES from terminal.
GPU_ID = None

def main():
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_volume_paths, train_label_paths, train_case_ids = collect_paths(SEG_DATA_DIR, 'train')
    valid_volume_paths, valid_label_paths, valid_case_ids = collect_paths(SEG_DATA_DIR, 'valid')
    test_volume_paths, test_label_paths, test_case_ids = collect_paths(SEG_DATA_DIR, 'test')

    clinical_encoder = KiTS23ClinicalJSONEncoder(
        json_path=CLINICAL_JSON,
    )
    clinical_encoder.fit(train_case_ids)

    clinical_dict = clinical_encoder.transform_all()
    clinical_dim = len(clinical_encoder.feature_names)

    print(f'Clinical dim: {clinical_dim}')
    print('Clinical features:')
    for name in clinical_encoder.feature_names:
        print(' -', name)

    train_dataset = SegmentationClinicalDataset3DViT(
        train_volume_paths,
        train_label_paths,
        train_case_ids,
        clinical_dict=clinical_dict,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
        window_min=WINDOW_MIN,
        window_max=WINDOW_MAX,
    )

    valid_dataset = SegmentationClinicalDataset3DViT(
        valid_volume_paths,
        valid_label_paths,
        valid_case_ids,
        clinical_dict=clinical_dict,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
        window_min=WINDOW_MIN,
        window_max=WINDOW_MAX,
    )

    test_dataset = SegmentationClinicalDataset3DViT(
        test_volume_paths,
        test_label_paths,
        test_case_ids,
        clinical_dict=clinical_dict,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
        window_min=WINDOW_MIN,
        window_max=WINDOW_MAX,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    segmentation_exp2_vit_unetseg_msaf(
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        clinical_dim=clinical_dim,
        out_classes=OUT_CLASSES,
        roi_size=ROI_SIZE,
        save_dir='saved_Exp2_ViTUNetSeg3D_MSAF_model',
        feature_size=16,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=12,
        patch_size=16,
        num_clinical_tokens=1,
        use_amp=True,
        init_from_exp1=INIT_FROM_EXP1,
        clinical_feature_names=clinical_encoder.feature_names,
    )

    if RUN_TEST_INFERENCE:
        inference_exp2_vit_unetseg_msaf(
            valid_loader=test_loader,
            valid_set=test_dataset,
            device=device,
            out_classes=OUT_CLASSES,
            checkpoint_path='./saved_Exp2_ViTUNetSeg3D_MSAF_model/best_model_exp2.pt',
            result_dir='./result_exp2_vit_unetseg_msaf',
        )


if __name__ == '__main__':
    main()
