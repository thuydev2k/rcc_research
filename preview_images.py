from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASS_RGB = np.array([
    [0, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0],
], dtype=np.uint8)


def colour_code_segmentation(mask):
    mask = mask.astype(np.int64)
    return CLASS_RGB[mask]


def overlay_mask_on_image(image_2d, mask, alpha=0.45):
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
    metadata,
    out_root,
    split_name="train",
    num_samples=30,
):
    out_root = Path(out_root)
    preview_dir = out_root / "previews" / split_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    df = metadata[metadata["split"] == split_name].copy()

    if len(df) == 0:
        print(f"No rows found for split={split_name}")
        return

    selected_rows = []

    cyst_df = df[df["has_cyst"] == True]
    selected_rows.append(cyst_df.head(num_samples // 4 + 1))

    tumor_df = df[(df["has_tumor"] == True) & (df["has_cyst"] == False)]
    selected_rows.append(tumor_df.head(num_samples // 4 + 1))

    kidney_df = df[
        (df["has_kidney"] == True)
        & (df["has_tumor"] == False)
        & (df["has_cyst"] == False)
    ]
    selected_rows.append(kidney_df.head(num_samples // 4 + 1))

    bg_df = df[df["has_foreground"] == False]
    selected_rows.append(bg_df.head(num_samples // 4 + 1))

    selected = pd.concat(selected_rows, axis=0).drop_duplicates()
    selected = selected.head(num_samples)

    print(f"Saving previews to: {preview_dir}")
    print(f"Number of previews: {len(selected)}")

    for preview_idx, (_, row) in enumerate(selected.iterrows()):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        image = np.load(image_path)["data"].astype(np.float32)
        label = np.load(label_path)["data"].astype(np.uint8)

        image_2d = image.squeeze(0)

        label_rgb = colour_code_segmentation(label)
        overlay = overlay_mask_on_image(image_2d, label)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title(f"CT\n{row['case_id']} slice {row['slice_idx']}")
        axes[0].axis("off")

        axes[1].imshow(label_rgb)
        axes[1].set_title(f"Mask\nlabels={np.unique(label).tolist()}")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay\nkidney={row['has_kidney']}, "
            f"tumor={row['has_tumor']}, cyst={row['has_cyst']}"
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

    print("Done.")


if __name__ == "__main__":
    out_root = Path("kits23_preprocessed_2d_all_slices")
    metadata = pd.read_csv(out_root / "metadata.csv")

    save_preprocessing_preview(
        metadata=metadata,
        out_root=out_root,
        split_name="train",
        num_samples=30,
    )