import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def hu_window_to_uint8(ct_slice: np.ndarray, level: float = 60.0, width: float = 200.0) -> np.ndarray:

    hu_min = level - width / 2.0
    hu_max = level + width / 2.0

    x = np.clip(ct_slice, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min)
    x = (x * 255.0).round().astype(np.uint8)

    return x

def resize_ct_uint8(img_uint8: np.ndarray, target_size=(512, 512)) -> np.ndarray:

    target_h, target_w = target_size

    resized = cv2.resize(
        img_uint8,
        dsize=(target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized

def resize_mask_nearest(mask: np.ndarray, target_size=(512, 512)) -> np.ndarray:
    target_h, target_w = target_size

    resized = cv2.resize(
        mask.astype(np.uint8),
        dsize=(target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(np.uint8)

def preprocess_ct_slice(
    ct_slice: np.ndarray,
    level: float = 60.0,
    width: float = 200.0,
    target_size: tuple = (512, 512),
) -> np.ndarray:

    x = hu_window_to_uint8(ct_slice, level=level, width=width)
    x = resize_ct_uint8(x, target_size=target_size)
    x = x.astype(np.float32) / 255.0

    # 5. Add channel dimension: (1, H, W)
    x = np.expand_dims(x, axis=0)

    return x

def load_nifti_as_dhw(path: Path, dtype=None) -> np.ndarray:

    nii = nib.load(str(path))
    arr = np.asanyarray(nii.dataobj)

    if dtype is not None:
        arr = arr.astype(dtype)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape} from {path}")

    return arr

def get_case_dirs(raw_root: Path):
    case_dirs = sorted([
        p for p in raw_root.iterdir()
        if p.is_dir() and p.name.startswith("case_")
    ])

    valid_case_dirs = []
    for case_dir in case_dirs:
        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"

        if img_path.exists() and seg_path.exists():
            valid_case_dirs.append(case_dir)
        else:
            print(f"[WARNING] Missing imaging or segmentation file in {case_dir}")

    return valid_case_dirs

def make_case_level_split(case_dirs, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    case_names = [p.name for p in case_dirs]

    train_val_cases, test_cases = train_test_split(
        case_names,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    val_size_relative = val_ratio / (train_ratio + val_ratio)

    train_cases, val_cases = train_test_split(
        train_val_cases,
        test_size=val_size_relative,
        random_state=seed,
        shuffle=True,
    )

    split_dict = {
        "train": sorted(train_cases),
        "valid": sorted(val_cases),
        "test": sorted(test_cases),
    }

    return split_dict

def save_npz(image_path: Path, label_path: Path, ct: np.ndarray, mask: np.ndarray, compressed: bool = False):

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)


    if compressed:
        np.savez_compressed(image_path, data=ct)
        np.savez_compressed(label_path, data=mask)
    else:
        np.savez(image_path, data=ct)
        np.savez(label_path, data=mask)

def preprocess_one_case(
    case_dir: Path,
    out_split_dir: Path,
    split_name: str,
    level: float,
    width: float,
    clip_limit: float,
    tile_grid_size: tuple,
    target_size: tuple,
    compressed: bool,
):
    case_id = case_dir.name

    img_path = case_dir / "imaging.nii.gz"
    seg_path = case_dir / "segmentation.nii.gz"

    ct_vol = load_nifti_as_dhw(img_path, dtype=np.float32)
    mask_vol = load_nifti_as_dhw(seg_path, dtype=np.uint8)

    if ct_vol.shape != mask_vol.shape:
        raise ValueError(
            f"Shape mismatch in {case_id}: "
            f"CT shape {ct_vol.shape}, mask shape {mask_vol.shape}"
        )

    d, h, w = ct_vol.shape

    rows = []

    for slice_idx in range(d):
        ct_slice = ct_vol[slice_idx]
        mask_slice = mask_vol[slice_idx].astype(np.uint8)

        ct_preprocessed = preprocess_ct_slice(
            ct_slice,
            level=level,
            width=width,
            target_size=target_size,
        )

        mask_resized = resize_mask_nearest(
            mask_slice,
            target_size=target_size,
        )

        # Safety check
        unique_labels = np.unique(mask_resized)
        invalid_labels = [int(x) for x in unique_labels if x not in [0, 1, 2, 3]]

        if len(invalid_labels) > 0:
            raise ValueError(
                f"Invalid labels after resizing in {case_id}, slice {slice_idx}: {invalid_labels}"
            )

        # Compute statistics after resizing
        has_kidney = bool(np.any(mask_resized == 1))
        has_tumor = bool(np.any(mask_resized == 2))
        has_cyst = bool(np.any(mask_resized == 3))
        has_foreground = bool(np.any(mask_resized > 0))

        out_name = f"{case_id}_slice_{slice_idx:05d}.npz"

        image_path = out_split_dir / "images" / out_name
        label_path = out_split_dir / "labels" / out_name

        save_npz(
            image_path=image_path,
            label_path=label_path,
            ct=ct_preprocessed,
            mask=mask_resized,
            compressed=compressed,
        )

        rows.append({
            "split": split_name,
            "case_id": case_id,
            "slice_idx": slice_idx,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "height": h,
            "width": w,
            "has_foreground": has_foreground,
            "has_kidney": has_kidney,
            "has_tumor": has_tumor,
            "has_cyst": has_cyst,
            "num_kidney_pixels": int(np.sum(mask_resized == 1)),
            "num_tumor_pixels": int(np.sum(mask_resized == 2)),
            "num_cyst_pixels": int(np.sum(mask_resized == 3)),
        })

    return rows


CLASS_RGB = np.array([
    [0, 0, 0],        # background: black
    [255, 255, 0],    # kidney: yellow
    [0, 0, 255],      # tumor: blue
    [0, 255, 0],      # cyst: green
], dtype=np.uint8)


def colour_code_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    Convert label mask to RGB image.

    mask:
        shape (H, W)
        values:
            0 = background
            1 = kidney
            2 = tumor
            3 = cyst

    return:
        RGB image, shape (H, W, 3)
    """
    mask = mask.astype(np.int64)

    if mask.min() < 0 or mask.max() > 3:
        raise ValueError(f"Invalid mask values: {np.unique(mask)}")

    return CLASS_RGB[mask]


def overlay_mask_on_image(image_2d: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Create overlay image for checking alignment.

    image_2d:
        shape (H, W), range [0, 1]

    mask:
        shape (H, W), values 0,1,2,3
    """
    image_2d = np.clip(image_2d, 0.0, 1.0)

    image_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
    image_rgb = (image_rgb * 255).astype(np.uint8)

    mask_rgb = colour_code_segmentation(mask)

    overlay = image_rgb.copy()
    foreground = mask > 0

    overlay[foreground] = (
        (1.0 - alpha) * image_rgb[foreground]
        + alpha * mask_rgb[foreground]
    ).astype(np.uint8)

    return overlay


def save_preprocessing_preview(
    metadata: pd.DataFrame,
    out_root: Path,
    split_name: str = "train",
    num_samples: int = 20,
):
    """
    Save image-label-overlay previews after preprocessing.

    It prioritizes:
        1. cyst slices
        2. tumor slices
        3. kidney slices
        4. background-only slices

    Output:
        out_root / previews / split_name / preview_xxx.png
    """
    preview_dir = out_root / "previews" / split_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    df = metadata[metadata["split"] == split_name].copy()

    if len(df) == 0:
        print(f"[WARNING] No rows found for split={split_name}. Skip preview.")
        return

    selected_rows = []

    # Priority 1: cyst slices
    cyst_df = df[df["has_cyst"] == True]
    selected_rows.append(cyst_df.head(num_samples // 4 + 1))

    # Priority 2: tumor slices without cyst
    tumor_df = df[(df["has_tumor"] == True) & (df["has_cyst"] == False)]
    selected_rows.append(tumor_df.head(num_samples // 4 + 1))

    # Priority 3: kidney slices without tumor/cyst
    kidney_df = df[
        (df["has_kidney"] == True)
        & (df["has_tumor"] == False)
        & (df["has_cyst"] == False)
    ]
    selected_rows.append(kidney_df.head(num_samples // 4 + 1))

    # Priority 4: background-only slices
    bg_df = df[df["has_foreground"] == False]
    selected_rows.append(bg_df.head(num_samples // 4 + 1))

    selected = pd.concat(selected_rows, axis=0).drop_duplicates()
    selected = selected.head(num_samples)

    print(f"\nSaving preprocessing previews to: {preview_dir}")
    print(f"Number of preview samples: {len(selected)}")

    for preview_idx, (_, row) in enumerate(selected.iterrows()):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        image = np.load(image_path)["data"].astype(np.float32)  # (1, H, W)
        label = np.load(label_path)["data"].astype(np.uint8)    # (H, W)

        if image.ndim == 3:
            image_2d = image.squeeze(0)
        else:
            image_2d = image

        label_rgb = colour_code_segmentation(label)
        overlay = overlay_mask_on_image(image_2d, label, alpha=0.45)

        unique_labels = np.unique(label)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title(
            f"Preprocessed CT\n"
            f"{row['case_id']} slice {row['slice_idx']}"
        )
        axes[0].axis("off")

        axes[1].imshow(label_rgb)
        axes[1].set_title(
            f"Label mask\n"
            f"labels={unique_labels.tolist()}"
        )
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay\n"
            f"kidney={row['has_kidney']}, tumor={row['has_tumor']}, cyst={row['has_cyst']}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        save_path = preview_dir / (
            f"preview_{preview_idx:03d}_"
            f"{row['case_id']}_slice_{int(row['slice_idx']):05d}_"
            f"tumor_{int(row['has_tumor'])}_"
            f"cyst_{int(row['has_cyst'])}.png"
        )

        plt.savefig(save_path, dpi=150)
        plt.close()

    print("Preview saving finished.")


def main(args):
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    print(f"Raw KiTS23 root: {raw_root}")
    print(f"Output root:      {out_root}")

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = get_case_dirs(raw_root)

    if len(case_dirs) == 0:
        raise RuntimeError(f"No valid case folders found in {raw_root}")

    print(f"Found {len(case_dirs)} valid cases.")

    split_dict = make_case_level_split(
        case_dirs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    split_path = out_root / "splits.json"
    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=4)

    print(f"Saved split file to: {split_path}")

    case_dir_map = {p.name: p for p in case_dirs}

    all_rows = []

    for split_name, case_names in split_dict.items():
        print(f"\nProcessing split: {split_name} | cases: {len(case_names)}")

        out_split_dir = out_root / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        for case_name in tqdm(case_names):
            case_dir = case_dir_map[case_name]

            rows = preprocess_one_case(
                case_dir=case_dir,
                out_split_dir=out_split_dir,
                split_name=split_name,
                level=args.level,
                width=args.width,
                clip_limit=args.clip_limit,
                tile_grid_size=(args.tile_grid_h, args.tile_grid_w),
                target_size=(512, 512),
                compressed=args.compressed,
            )

            all_rows.extend(rows)

    metadata = pd.DataFrame(all_rows)
    metadata_path = out_root / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    if args.save_preview:
        save_preprocessing_preview(
            metadata=metadata,
            out_root=out_root,
            split_name=args.preview_split,
            num_samples=args.num_preview,
        )

    print("\nDone.")
    print(f"Saved metadata to: {metadata_path}")

    print("\nSummary:")
    print(metadata.groupby("split").agg(
        num_slices=("image_path", "count"),
        num_cases=("case_id", "nunique"),
        foreground_slices=("has_foreground", "sum"),
        kidney_slices=("has_kidney", "sum"),
        tumor_slices=("has_tumor", "sum"),
        cyst_slices=("has_cyst", "sum"),
    ))

    print("\nImportant:")
    print("All slices were saved, including background-only slices.")
    print("CT image was saved as float32 [0, 1].")
    print("Mask was saved as uint8 with labels 0, 1, 2, 3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_root",
        type=str,
        default="dataset",
        help="Path to raw KiTS23 dataset folder containing case_00000, case_00001, ...",
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="kits23_preprocessed_2d_all_slices",
        help="Output folder for preprocessed 2D NPZ slices.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # Paper preprocessing parameters
    parser.add_argument("--level", type=float, default=60.0)
    parser.add_argument("--width", type=float, default=200.0)
    parser.add_argument("--clip_limit", type=float, default=2.0)
    parser.add_argument("--tile_grid_h", type=int, default=8)
    parser.add_argument("--tile_grid_w", type=int, default=8)

    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Smaller files but slower training loading.",
    )

    parser.add_argument(
        "--save_preview",
        action="store_true",
        help="Save image-label-overlay preview images after preprocessing.",
    )

    parser.add_argument(
        "--num_preview",
        type=int,
        default=20,
        help="Number of preprocessing preview images to save.",
    )

    parser.add_argument(
        "--preview_split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Which split to use for preview images.",
    )

    args = parser.parse_args()
    main(args)