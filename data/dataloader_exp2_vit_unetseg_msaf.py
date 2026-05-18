import os
import torch

from data.dataloader_exp1_vit_unetseg import (
    SegmentationDataset3DViT,
    collect_paths as collect_exp1_paths,
)


def collect_paths(seg_data_dir, split):
    """
    Same as Exp1 collect_paths, but also returns case_ids.

    Returns:
        volume_paths, label_paths, case_ids
    """
    volume_paths, label_paths = collect_exp1_paths(seg_data_dir, split)
    case_ids = [os.path.splitext(os.path.basename(p))[0] for p in volume_paths]
    return volume_paths, label_paths, case_ids


class SegmentationClinicalDataset3DViT(SegmentationDataset3DViT):
    """
    Exp2 dataset = Exp1 dataset + clinical vector.

    Exp1 returns:
        image, label

    Exp2 returns:
        image, label, clinical
    """

    def __init__(
        self,
        image_paths,
        label_paths,
        case_ids,
        clinical_dict,
        *args,
        **kwargs,
    ):
        super().__init__(image_paths, label_paths, *args, **kwargs)

        if len(case_ids) != len(image_paths):
            raise ValueError(
                f"case_ids length must match image_paths length. "
                f"Got {len(case_ids)} and {len(image_paths)}."
            )

        self.case_ids = [str(x) for x in case_ids]
        self.clinical_dict = clinical_dict

        missing = [case_id for case_id in self.case_ids if case_id not in self.clinical_dict]
        if missing:
            raise KeyError(f"Missing clinical vectors for {len(missing)} cases. Example: {missing[:5]}")

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        case_id = self.case_ids[index]
        clinical = torch.tensor(self.clinical_dict[case_id], dtype=torch.float32)

        return image, label, clinical
