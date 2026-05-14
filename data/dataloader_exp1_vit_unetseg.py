import os
import numpy as np
import torch
from torch.utils.data import Dataset


def collect_paths(seg_data_dir, split):
    image_dir = os.path.join(seg_data_dir, split, 'images')
    label_dir = os.path.join(seg_data_dir, split, 'labels')

    volume_paths = []
    label_paths = []

    cases = sorted([f for f in os.listdir(image_dir) if f.endswith('.npz')])

    for case in cases:
        case_key = os.path.splitext(case)[0]
        volume_path = os.path.join(image_dir, f'{case_key}.npz')
        label_path = os.path.join(label_dir, f'{case_key}.npz')

        if not os.path.exists(label_path):
            print(f"[Warning] Missing label for {case_key}: {label_path}")
            continue

        volume_paths.append(volume_path)
        label_paths.append(label_path)

    print(f"{split}: found {len(volume_paths)} cases")
    return volume_paths, label_paths


class SegmentationDataset3DViT(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        roi_size=(128, 256, 256),
        image_key='data',
        label_key='data',
        normalize=False,
        window_min=-79.0,
        window_max=304.0,
        image_pad_value=None,
        check_label_values=True,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.roi_size = tuple(roi_size)
        self.image_key = image_key
        self.label_key = label_key
        self.normalize = normalize
        self.window_min = window_min
        self.window_max = window_max
        self.image_pad_value = image_pad_value
        self.check_label_values = check_label_values

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(
                f"image_paths and label_paths length mismatch: "
                f"{len(self.image_paths)} vs {len(self.label_paths)}"
            )

        for s in self.roi_size:
            if s % 16 != 0:
                raise ValueError(
                    f"roi_size must be divisible by 16 for patch_size=16. Got roi_size={self.roi_size}"
                )

    def __len__(self):
        return len(self.image_paths)

    def _load_npz_array(self, path, key):
        data = np.load(path)

        if key is not None:
            if key not in data.files:
                raise KeyError(f"{path} does not contain key='{key}'. Available keys={data.files}")
            return data[key]

        if len(data.files) != 1:
            raise ValueError(
                f"{path} contains multiple arrays: {data.files}. "
                f"Set image_key/label_key explicitly."
            )

        return data[data.files[0]]

    def _prepare_image_label(self, image, label):
        image = np.asarray(image)
        label = np.asarray(label)

        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]

        if image.ndim != 3:
            raise ValueError(f"Expected image shape [D,H,W], got {image.shape}")
        if label.ndim != 3:
            raise ValueError(f"Expected label shape [D,H,W], got {label.shape}")
        if image.shape != label.shape:
            raise ValueError(f"Image/label shape mismatch: {image.shape} vs {label.shape}")

        image = image.astype(np.float32)
        label = label.astype(np.int64)

        if self.normalize:
            image = np.clip(image, self.window_min, self.window_max)
            image = (image - self.window_min) / (self.window_max - self.window_min + 1e-8)
            image = image.astype(np.float32)

        if self.check_label_values:
            unique = np.unique(label)
            invalid = unique[(unique < 0) | (unique > 3)]
            if invalid.size > 0:
                raise ValueError(
                    f"Invalid label values: {invalid.tolist()}. "
                    f"Expected KiTS23 labels: 0,1,2,3."
                )

        return image, label

    def _foreground_center(self, label):
        foreground = label > 0

        if foreground.sum() == 0:
            d, h, w = label.shape
            return np.array([d // 2, h // 2, w // 2], dtype=np.int64)

        coords = np.array(np.where(foreground))
        z_min, y_min, x_min = coords.min(axis=1)
        z_max, y_max, x_max = coords.max(axis=1)

        return np.array(
            [
                (z_min + z_max) // 2,
                (y_min + y_max) // 2,
                (x_min + x_max) // 2,
            ],
            dtype=np.int64,
        )

    def _fixed_crop_or_pad(self, image, label, center):
        d, h, w = image.shape
        roi_d, roi_h, roi_w = self.roi_size

        starts = center - np.array([roi_d, roi_h, roi_w]) // 2
        ends = starts + np.array([roi_d, roi_h, roi_w])

        shape = np.array([d, h, w])

        for axis in range(3):
            if starts[axis] < 0:
                ends[axis] -= starts[axis]
                starts[axis] = 0

            if ends[axis] > shape[axis]:
                shift = ends[axis] - shape[axis]
                starts[axis] -= shift
                ends[axis] = shape[axis]

            if starts[axis] < 0:
                starts[axis] = 0

        z0, y0, x0 = starts.astype(int).tolist()
        z1, y1, x1 = ends.astype(int).tolist()

        image_crop = image[z0:z1, y0:y1, x0:x1]
        label_crop = label[z0:z1, y0:y1, x0:x1]

        pad_d = roi_d - image_crop.shape[0]
        pad_h = roi_h - image_crop.shape[1]
        pad_w = roi_w - image_crop.shape[2]

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_width = (
                (0, max(pad_d, 0)),
                (0, max(pad_h, 0)),
                (0, max(pad_w, 0)),
            )
            if self.image_pad_value is not None:
                image_pad_value = self.image_pad_value
            else:
                image_pad_value = 0.0 if self.normalize else self.window_min
            image_crop = np.pad(image_crop, pad_width, mode='constant', constant_values=image_pad_value)
            label_crop = np.pad(label_crop, pad_width, mode='constant', constant_values=0)

        if image_crop.shape != self.roi_size:
            raise RuntimeError(f"Expected crop shape {self.roi_size}, got {image_crop.shape}")

        return image_crop, label_crop

    def __getitem__(self, index):
        image = self._load_npz_array(self.image_paths[index], self.image_key)
        label = self._load_npz_array(self.label_paths[index], self.label_key)

        image, label = self._prepare_image_label(image, label)

        center = self._foreground_center(label)
        image, label = self._fixed_crop_or_pad(image, label, center)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
