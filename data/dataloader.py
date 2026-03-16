import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

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

            for d in range(img.shape[0]):
                img_slice_resized = cv2.resize(
                    img[d], (224, 224),
                    interpolation=cv2.INTER_LINEAR)

                lbl_slice_resized = cv2.resize(
                    lbl[d], (224, 224),
                    interpolation=cv2.INTER_LINEAR)
                
                self.samples.append({
                    "image": img_slice_resized,
                    "label": lbl_slice_resized,
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        s = self.samples[index]

        image = torch.tensor(s["image"], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(s["label"], dtype=torch.long)

        return image, label