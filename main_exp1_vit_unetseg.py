import torch
from torch.utils.data import DataLoader

from data.dataloader_exp1_vit_unetseg import SegmentationDataset3DViT, collect_paths
from segmentation_exp1_vit_unetseg import segmentation_exp1_vit_unetseg
from inference_exp1_vit_unetseg import inference_exp1_vit_unetseg


SEG_DATA_DIR = 'dataset/kits23/labeled/'
OUT_CLASSES = 4

# Must be divisible by patch size 16.
ROI_SIZE = (128, 256, 256)

# If your .npz files are already normalized, keep False.
# If your .npz files store raw HU values, set True.
NORMALIZE = True


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_volume_paths, train_label_paths = collect_paths(SEG_DATA_DIR, 'train')
    valid_volume_paths, valid_label_paths = collect_paths(SEG_DATA_DIR, 'valid')
    test_volume_paths, test_label_paths = collect_paths(SEG_DATA_DIR, 'test')

    train_dataset = SegmentationDataset3DViT(
        train_volume_paths,
        train_label_paths,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
    )

    valid_dataset = SegmentationDataset3DViT(
        valid_volume_paths,
        valid_label_paths,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
    )

    test_dataset = SegmentationDataset3DViT(
        test_volume_paths,
        test_label_paths,
        roi_size=ROI_SIZE,
        image_key='data',
        label_key='data',
        normalize=NORMALIZE,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    segmentation_exp1_vit_unetseg(
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=200,
        lr=1e-4,
        out_classes=OUT_CLASSES,
        roi_size=ROI_SIZE,
        save_dir='saved_Exp1_ViTUNetSeg3D_model',
        feature_size=16,
        hidden_size=384,
        mlp_dim=1536,
        num_heads=12,
        patch_size=16,
        use_amp=True,
    )

    inference_exp1_vit_unetseg(
        valid_loader=test_loader,
        valid_set=test_dataset,
        device=device,
        out_classes=OUT_CLASSES,
        checkpoint_path='./saved_Exp1_ViTUNetSeg3D_model/best_model_exp1.pt',
        result_dir='./result_exp1_vit_unetseg',
    )


if __name__ == '__main__':
    main()
