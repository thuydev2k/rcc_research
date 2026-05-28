import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SegmentationDataset2D(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.samples = []

        for img_path, lbl_path in zip(image_paths, label_paths):
            img = np.load(img_path)['data']     # (D, H, W)
            lbl = np.load(lbl_path)['data']     # (D, H, W)

            assert img.shape == lbl.shape, f"Shape mismatch: {img.shape} vs {lbl.shape}"

            for d in range(img.shape[0]):
                has_kidney = np.any(lbl[d] == 1)
                has_tumor = np.any(lbl[d] == 2)
                has_cyst = np.any(lbl[d] == 3)

                self.samples.append({
                    "image": img[d],
                    "label": lbl[d],
                    "has_kidney": has_kidney,
                    "has_tumor": has_tumor,
                    "has_cyst": has_cyst,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        image = torch.from_numpy(s["image"]).float().unsqueeze(0) / 255.0
        label = torch.from_numpy(s["label"]).long()

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
            raise ValueError("Invalid batch composition. Reduce tumor_ratio or cyst_ratio.")

        if num_batches is None:
            self.num_batches = math.ceil(len(dataset) / batch_size)
        else:
            self.num_batches = num_batches

        print("BatchSampler statistics:")
        print(f"Total slices: {len(dataset)}")
        print(f"Cyst slices: {len(self.cyst_indices)} {len(self.cyst_indices)/len(dataset)*100:.2f}%")
        print(f"Tumor slices: {len(self.tumor_indices)} {len(self.tumor_indices)/len(dataset)*100:.2f}%")
        print(f"Other slices: {len(self.other_indices)} {len(self.other_indices)/len(dataset)*100:.2f}%")
        print()
        print("Each batch:")
        print(f"Cyst slices: {self.num_cyst_per_batch}")
        print(f"Tumor slices: {self.num_tumor_per_batch}")
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