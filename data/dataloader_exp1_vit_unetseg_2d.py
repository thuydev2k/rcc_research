import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationSliceDataset2DViT(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        case_ids,
        img_size: Tuple[int, int] = (512, 512),
        image_key: str = 'data',
        label_key: str = 'data',
        slice_mode: str = 'non_empty',
        min_foreground_pixels: int = 1,
        return_metadata: bool = False,
    ):
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.case_ids = list(case_ids) if case_ids is not None else [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
        self.img_size = tuple(img_size)
        self.image_key = image_key
        self.label_key = label_key
        self.slice_mode = slice_mode
        self.min_foreground_pixels = min_foreground_pixels
        self.return_metadata = return_metadata

        for s in self.img_size:
            if s % 16 != 0:
                raise ValueError(f"img_size must be divisible by patch_size=16. Got img_size={self.img_size}")

        self.samples = self._build_slice_index()
        print(f"2D slice dataset mode='{slice_mode}': {len(self.samples)} slices")

    def _load_npz_array(self, path, key):
        data = np.load(path)

        if key is not None:
            return data[key]

        return data[data.files[0]]

    def _build_slice_index(self):
        samples = []

        for vol_idx, label_path in enumerate(self.label_paths):
            label = self._load_npz_array(label_path, self.label_key)
            label = np.asarray(label)

            if label.ndim == 4 and label.shape[0] == 1:
                label = label[0]
            if label.ndim != 3:
                raise ValueError(f"Expected label shape [D,H,W], got {label.shape} from {label_path}")

            if self.slice_mode == 'all':
                slice_indices = list(range(label.shape[0]))
            else:
                fg_per_slice = (label > 0).sum(axis=(1, 2))
                slice_indices = np.where(fg_per_slice >= self.min_foreground_pixels)[0].tolist()

                # Safety fallback: if a case has no foreground slices, keep the middle slice.
                if len(slice_indices) == 0:
                    slice_indices = [label.shape[0] // 2]

            for z in slice_indices:
                samples.append((vol_idx, int(z)))

        return samples

    def __len__(self):
        return len(self.samples)

    def _prepare_volume_label(self, image, label):
        image = np.asarray(image)
        label = np.asarray(label)

        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
        image = image.astype(np.float32)
        label = label.astype(np.int64)

        return image, label

    def _foreground_center_2d(self, label_slice):
        foreground = label_slice > 0
        h, w = label_slice.shape

        if foreground.sum() == 0:
            return np.array([h // 2, w // 2], dtype=np.int64)

        coords = np.array(np.where(foreground))
        y_min, x_min = coords.min(axis=1)
        y_max, x_max = coords.max(axis=1)
        return np.array([(y_min + y_max) // 2, (x_min + x_max) // 2], dtype=np.int64)

    def _fixed_crop_or_pad_2d(self, image_slice, label_slice, center):
        h, w = image_slice.shape
        roi_h, roi_w = self.img_size

        starts = center - np.array([roi_h, roi_w]) // 2
        ends = starts + np.array([roi_h, roi_w])
        shape = np.array([h, w])

        for axis in range(2):
            if starts[axis] < 0:
                ends[axis] -= starts[axis]
                starts[axis] = 0

            if ends[axis] > shape[axis]:
                shift = ends[axis] - shape[axis]
                starts[axis] -= shift
                ends[axis] = shape[axis]

            if starts[axis] < 0:
                starts[axis] = 0

        y0, x0 = starts.astype(int).tolist()
        y1, x1 = ends.astype(int).tolist()

        image_crop = image_slice[y0:y1, x0:x1]
        label_crop = label_slice[y0:y1, x0:x1]

        pad_h = roi_h - image_crop.shape[0]
        pad_w = roi_w - image_crop.shape[1]

        if pad_h > 0 or pad_w > 0:
            pad_width = ((0, max(pad_h, 0)), (0, max(pad_w, 0)))
            image_crop = np.pad(image_crop, pad_width, mode='constant', constant_values=0)
            label_crop = np.pad(label_crop, pad_width, mode='constant', constant_values=0)

        if image_crop.shape != self.img_size:
            raise RuntimeError(f"Expected crop shape {self.img_size}, got {image_crop.shape}")

        return image_crop, label_crop

    def __getitem__(self, index):
        vol_idx, slice_idx = self.samples[index]

        image = self._load_npz_array(self.image_paths[vol_idx], self.image_key)
        label = self._load_npz_array(self.label_paths[vol_idx], self.label_key)
        image, label = self._prepare_volume_label(image, label)

        image_slice = image[slice_idx]
        label_slice = label[slice_idx]

        center = self._foreground_center_2d(label_slice)
        image_slice, label_slice = self._fixed_crop_or_pad_2d(image_slice, label_slice, center)

        image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_slice, dtype=torch.long)

        if self.return_metadata:
            case_id = self.case_ids[vol_idx]
            return image_tensor, label_tensor, case_id, torch.tensor(slice_idx, dtype=torch.long)

        return image_tensor, label_tensor
