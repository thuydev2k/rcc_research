import torch
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset2D(Dataset):
    def __init__(
            self,
            image_paths,
            label_paths,
            clinical_data_list,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.clinical_data = clinical_data_list

        self.samples = []

        for img_path, lbl_path, clinical in zip(image_paths, label_paths, clinical_data_list):
            img = np.load(img_path)['data']     # (D, H, W)
            lbl = np.load(lbl_path)['data']     # (D, H, W)

            for d in range(img.shape[0]):
                self.samples.append({
                    "image": img[d],
                    "label": lbl[d],
                    "clinical": clinical
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        s = self.samples[index]

        image = torch.tensor(s["image"], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(s["label"], dtype=torch.long)

        clinical = s["clinical"]

        return image, label, clinical