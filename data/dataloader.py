import math
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


class KiTS23SliceDataset2D(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_paths = sorted(self.image_dir.glob("*.npz"))
        self.label_paths = sorted(self.label_dir.glob("*.npz"))

        self.samples = []

        print(f"Initializing dataset from: {self.image_dir}")
        print(f"Total files: {len(self.image_paths)}")

        for img_path, lbl_path in tqdm(
            zip(self.image_paths, self.label_paths),
            total=len(self.image_paths),
            desc="Reading slice metadata",
        ):
            image = np.load(img_path)["data"].astype(np.float32)  # (1, 512, 512)
            label = np.load(lbl_path)["data"].astype(np.int8)  # (512, 512)

            # Resize image: (1, H, W) -> (H, W) -> resize -> (1, 224, 224)
            image = image.squeeze(0)

            image = cv2.resize(
                image, (224, 224),
                interpolation=cv2.INTER_LINEAR)

            image = image[None, :, :].astype(np.float32)

            label = cv2.resize(
                label, (224, 224),
                interpolation=cv2.INTER_NEAREST).astype(np.int64)
                
            has_kidney = bool(np.any(label == 1))
            has_tumor = bool(np.any(label == 2))
            has_cyst = bool(np.any(label == 3))

            self.samples.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "image": image,
                    "label": label,
                    "has_kidney": has_kidney,
                    "has_tumor": has_tumor,
                    "has_cyst": has_cyst,
                }
            )

        print("Dataset statistics:")
        print(f"Total slices: {len(self.samples)}")
        print(f"Kidney slices: {sum(s['has_kidney'] for s in self.samples)}")
        print(f"Tumor slices: {sum(s['has_tumor'] for s in self.samples)}")
        print(f"Cyst slices: {sum(s['has_cyst'] for s in self.samples)}")
        print(
            f"Background-only slices: "
            f"{sum((not s['has_kidney']) and (not s['has_tumor']) and (not s['has_cyst']) for s in self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        image = torch.from_numpy(s["image"])
        label = torch.from_numpy(s["label"])

        return image, label


class TumorCystBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        tumor_ratio=0.25,
        cyst_ratio=0.25,
        num_batches=None,
        seed=42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tumor_ratio = tumor_ratio
        self.cyst_ratio = cyst_ratio
        self.seed = seed
        self.epoch = 0

        # Priority:
        # cyst slice = contains cyst, even if it also contains tumor
        # tumor slice = contains tumor but not cyst
        # other slice = no tumor and no cyst
        self.cyst_indices = [
            i for i, s in enumerate(dataset.samples)
            if s["has_cyst"]
        ]

        self.tumor_indices = [
            i for i, s in enumerate(dataset.samples)
            if s["has_tumor"] and not s["has_cyst"]
        ]

        self.other_indices = [
            i for i, s in enumerate(dataset.samples)
            if not s["has_tumor"] and not s["has_cyst"]
        ]

        self.num_cyst_per_batch = max(1, int(batch_size * cyst_ratio))
        self.num_tumor_per_batch = max(1, int(batch_size * tumor_ratio))
        self.num_other_per_batch = (
            batch_size - self.num_cyst_per_batch - self.num_tumor_per_batch
        )

        if self.num_other_per_batch < 0:
            raise ValueError(
                "Invalid batch composition. Reduce tumor_ratio or cyst_ratio."
            )

        if len(self.cyst_indices) == 0:
            raise RuntimeError("No cyst slices found in dataset.")

        if len(self.tumor_indices) == 0:
            raise RuntimeError("No tumor-only slices found in dataset.")

        if len(self.other_indices) == 0:
            raise RuntimeError("No other slices found in dataset.")

        if num_batches is None:
            self.num_batches = math.ceil(len(dataset) / batch_size)
        else:
            self.num_batches = num_batches

        print("\nBatchSampler statistics:")
        print(f"Total slices: {len(dataset)}")
        print(
            f"Cyst slices: {len(self.cyst_indices)} "
            f"({len(self.cyst_indices) / len(dataset) * 100:.2f}%)"
        )
        print(
            f"Tumor-only slices: {len(self.tumor_indices)} "
            f"({len(self.tumor_indices) / len(dataset) * 100:.2f}%)"
        )
        print(
            f"Other slices: {len(self.other_indices)} "
            f"({len(self.other_indices) / len(dataset) * 100:.2f}%)"
        )

        print("\nEach batch:")
        print(f"Cyst slices: {self.num_cyst_per_batch}")
        print(f"Tumor-only slices: {self.num_tumor_per_batch}")
        print(f"Other slices: {self.num_other_per_batch}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.num_batches):
            batch = []

            cyst_batch = rng.choice(
                self.cyst_indices,
                size=self.num_cyst_per_batch,
                replace=True,
            )

            tumor_batch = rng.choice(
                self.tumor_indices,
                size=self.num_tumor_per_batch,
                replace=True,
            )

            if self.num_other_per_batch > 0:
                other_batch = rng.choice(
                    self.other_indices,
                    size=self.num_other_per_batch,
                    replace=True,
                )
                batch.extend(other_batch.tolist())

            batch.extend(cyst_batch.tolist())
            batch.extend(tumor_batch.tolist())

            rng.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches