import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SAFE_NUMERICAL_KEYS = [
    "age_at_nephrectomy",
    "bmi",
    "last_preop_egfr.value",
]

# Optional. This is pre-operative/radiology-derived, but it can be considered
# image-derived information. Use it only in a separate ablation experiment.
OPTIONAL_RADIOLOGY_KEYS = [
    "radiographic_size",
]

COMORBIDITY_KEYS = [
    "chronic_kidney_disease",
]

CATEGORICAL_MAPS = {
    "gender": {
        "__missing__": 0,
        "male": 1,
        "female": 2,
        "transgender_male_to_female": 3,
    },
}


def get_nested(case: Dict, key: str):
    """Read nested keys like 'last_preop_egfr.value'."""
    value = case
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def get_training_stats(
    json_path: str,
    train_case_ids: Iterable[str],
    include_radiographic_size: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean/std only from training cases to avoid validation/test leakage.
    """
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    data = {item["case_id"]: item for item in raw_data}
    numerical_keys = list(SAFE_NUMERICAL_KEYS)
    if include_radiographic_size:
        numerical_keys += OPTIONAL_RADIOLOGY_KEYS

    stats = {}
    for key in numerical_keys:
        values = []
        for case_id in train_case_ids:
            case = data.get(case_id)
            if case is None:
                continue
            value = get_nested(case, key)
            if value is not None:
                values.append(float(value))

        if len(values) == 0:
            stats[key] = (0.0, 1.0)
            continue

        values = np.asarray(values, dtype=np.float32)
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0
        stats[key] = (mean, std)

    return stats


class KiTS23PreopClinicalDataset(Dataset):
    """
    Clinical dataset using only pre-operative/patient-history variables.

    Returned dict:
        numerical:          FloatTensor [N]
        numerical_missing:  FloatTensor [N], 1 = missing, 0 = present
        comorbidities:      FloatTensor [1]
        gender:             LongTensor scalar
        smoking_history:    LongTensor scalar
        chewing_tobacco_use:LongTensor scalar
        alcohol_use:        LongTensor scalar
    """
    def __init__(
        self,
        json_path: str,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        include_radiographic_size: bool = True,
    ):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.data = {item["case_id"]: item for item in raw_data}
        self.cases = list(self.data.keys())
        self.stats = stats or {}
        self.include_radiographic_size = include_radiographic_size

        self.numerical_keys = list(SAFE_NUMERICAL_KEYS)
        if include_radiographic_size:
            self.numerical_keys += OPTIONAL_RADIOLOGY_KEYS

    def __len__(self):
        return len(self.cases)

    def normalize_value(self, case: Dict, key: str) -> Tuple[float, float]:
        value = get_nested(case, key)
        is_missing = value is None
        if is_missing:
            return 0.0, 1.0

        mean, std = self.stats.get(key, (0.0, 1.0))
        if std < 1e-6:
            std = 1.0
        return float((float(value) - mean) / std), 0.0

    def map_category(self, case: Dict, key: str) -> int:
        mapping = CATEGORICAL_MAPS[key]
        value = case.get(key)
        if value is None:
            return mapping["__missing__"]
        return mapping.get(value, mapping["__missing__"])

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        case = self.data[case_id]

        numerical_values = []
        missing_values = []
        for key in self.numerical_keys:
            value, missing = self.normalize_value(case, key)
            numerical_values.append(value)
            missing_values.append(missing)

        comorbidities = [
            float(case.get("comorbidities", {}).get(key, False) is True)
            for key in COMORBIDITY_KEYS
        ]

        clinical_data = {
            "case_id": case_id,
            "numerical": torch.tensor(numerical_values, dtype=torch.float32),
            "numerical_missing": torch.tensor(missing_values, dtype=torch.float32),
            "comorbidities": torch.tensor(comorbidities, dtype=torch.float32),
            "gender": torch.tensor(self.map_category(case, "gender"), dtype=torch.long),
        }

        return clinical_data
