import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataloader import SegmentationDataset3D
from segmentation_baseline import segmentation_baseline_3d
from inference import inference

device = "cpu"

SEG_DATA_DIR = "dataset/kits23/labeled/"
out_classes = 4

def collect_paths(split):
    image_dir = os.path.join(SEG_DATA_DIR, split, "images")
    label_dir = os.path.join(SEG_DATA_DIR, split, "labels")

    volume_paths = []
    label_paths = []

    cases = sorted(os.listdir(image_dir))

    for case in tqdm(cases):
        c = case.split(".")[0].split("_")[-1]
        case_key = f"case_{c}"

        volume_path = os.path.join(image_dir, f"{case_key}.npz")
        label_path = os.path.join(label_dir, f"{case_key}.npz")

        volume_paths.append(volume_path)
        label_paths.append(label_path)

    return volume_paths, label_paths

train_volume_paths, train_label_paths = collect_paths("train")
valid_volume_paths, valid_label_paths = collect_paths("valid")
test_volume_paths, test_label_paths = collect_paths("test")

train_dataset = SegmentationDataset3D(
    train_volume_paths,
    train_label_paths,
    margin=(8, 32, 32),
)

valid_dataset = SegmentationDataset3D(
    valid_volume_paths,
    valid_label_paths,
    margin=(8, 32, 32),
)

test_dataset = SegmentationDataset3D(
    test_volume_paths,
    test_label_paths,
    margin=(8, 32, 32),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

segmentation_baseline_3d(
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    epochs=100,
    lr=1e-4,
    out_classes=out_classes,
)

inference(
    valid_loader=test_loader,
    valid_set=test_dataset,
    device=device,
    out_classes=out_classes,
    checkpoint_path="./saved_UNet3D_model/best_model1.pt",
    result_dir="./result_3d",
)