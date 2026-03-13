import json
import torch
from torch.utils.data import Dataset
import numpy as np

class KiTS23ClinicalDataset(Dataset):
    def __init__(self, json_path, stats=None):

        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.data = {item['case_id']: item for item in raw_data}
        self.cases = list(self.data.keys())

        self.stats = stats

        # category mappings
        self.gender_map = {"male": 0, "female": 1, "transgender_male_to_female": 2}
        self.smoking_history_map = {
            "never_smoked": 0,
            "not_found_in_emr": 1,
            "previous_smoker": 2,
            "current_smoker": 3
        }
        self.surgery_type_map = {
            "robotic": 0,
            "open": 1,
            "laparoscopic": 2,
            "percutaneous": 3
        }
        self.surgical_approach_map = {
            "transperitoneal": 0,
            "retroperitoneal": 1,
            "trans_to_retro": 2,
            None: 3
        }
        self.tumor_histologic_subtype_map = {
            "clear_cell_rcc": 0,
            "papillary_rcc": 1,
            "chromophobe_rcc": 2,
            "oncocytoma": 3,
            "multilocular_cystic_rcc": 4,
            "clear_cell_papillary": 5,
            "rcc_unclassified": 6,
            "transitional_cell_carcinoma": 7,
            "spindle_cell_neoplasm": 8,
            "angiomyolipoma": 9,
            "wilms_tumor": 10,
            "cyst": 11,
            "mest": 12,
            "other": 13,
            None: 14
        }
        self.pathology_t_stage_map = {
            "1a": 0, "1b": 1, "2a": 2, "2b": 3, "3": 4, "4": 5, "na": 6, None: 7
        }

    def __len__(self):
        return len(self.cases)

    def normalize(self, x, key):
        if self.stats is None or x is None:
            return 0.0

        mean, std = self.stats[key]

        if std < 1e-6:
            return 0.0

        return (x - mean) / std

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        
        case = self.data[case_id]

        # ---- numerical ----
        numericals = np.array([
            self.normalize(case.get("bmi"), "bmi"),
            self.normalize(case.get("age_at_nephrectomy"), "age_at_nephrectomy"),
            self.normalize(case.get("pathologic_size"), "pathologic_size"),
            self.normalize(case.get("radiographic_size"), "radiographic_size"),
            self.normalize(case.get("hospitalization"), "hospitalization"),
        ], dtype=np.float32)

        # ---- comorbidities (binary vector) ----

        SELECTED_COMORBIDITIES = [
            "myocardial_infarction",
            "localized_solid_tumor",
            "congestive_heart_failure",
            "uncomplicated_diabetes_mellitus",
            "metastatic_solid_tumor",
            "mild_liver_disease",
        ]

        comorbidities = np.array(
            [float(case.get("comorbidities", {}).get(k, False)) for k in SELECTED_COMORBIDITIES],
            dtype=np.float32
        )

        # ---- categorical ----
        gender = self.gender_map[case.get("gender")]
        smoking_history = self.smoking_history_map[case.get("smoking_history")]
        surgery_type = self.surgery_type_map[case.get("surgery_type")]
        surgical_approach = self.surgical_approach_map[case.get("surgical_approach")]
        tumor_histologic_subtype = self.tumor_histologic_subtype_map[case.get("tumor_histologic_subtype")]

        # ---- ordinal ----
        pathology_t_stage = self.pathology_t_stage_map[case.get("pathology_t_stage")]

        clinical_data = {
            "numerical": torch.tensor(numericals),
            "comorbidities": torch.tensor(comorbidities),
            "gender": torch.tensor(gender),
            "smoking_history": torch.tensor(smoking_history),
            "surgery_type": torch.tensor(surgery_type),
            "surgical_approach": torch.tensor(surgical_approach),
            "tumor_histologic_subtype": torch.tensor(tumor_histologic_subtype),
            "pathology_t_stage": torch.tensor(pathology_t_stage, dtype=torch.long)
        }

        return clinical_data