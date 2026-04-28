import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset3D(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        margin=(8, 32, 32),
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.margin = margin  # (margin_D, margin_H, margin_W)

    def __len__(self):
        return len(self.image_paths)

    def _get_foreground_bbox(self, label):
        """
        label: [D, H, W]
        foreground = kidney/tumor/cyst = label > 0
        """

        foreground = label > 0

        # If the case doesn't include foreground, return full volume
        if foreground.sum() == 0:
            d, h, w = label.shape
            return 0, d, 0, h, 0, w

        coords = np.where(foreground)

        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()

        return z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1

    def _apply_margin(self, bbox, volume_shape):
        """
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
        volume_shape: [D, H, W]
        """

        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        d, h, w = volume_shape
        mz, my, mx = self.margin

        z_min = max(z_min - mz, 0)
        z_max = min(z_max + mz, d)

        y_min = max(y_min - my, 0)
        y_max = min(y_max + my, h)

        x_min = max(x_min - mx, 0)
        x_max = min(x_max + mx, w)

        return z_min, z_max, y_min, y_max, x_min, x_max

    def _crop_roi(self, image, label):
        bbox = self._get_foreground_bbox(label)
        bbox = self._apply_margin(bbox, image.shape)

        z_min, z_max, y_min, y_max, x_min, x_max = bbox

        image_roi = image[
            z_min:z_max,
            y_min:y_max,
            x_min:x_max,
        ]

        label_roi = label[
            z_min:z_max,
            y_min:y_max,
            x_min:x_max,
        ]

        return image_roi, label_roi

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])["data"]  # [D, H, W]
        label = np.load(self.label_paths[index])["data"]  # [D, H, W]

        image = image.astype(np.float32)
        label = label.astype(np.int64)

        image, label = self._crop_roi(image, label)

        # image: [D, H, W] -> [1, D, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # label: [D, H, W]
        label = torch.tensor(label, dtype=torch.long)

        return image, label