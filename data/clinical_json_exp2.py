import json
import numpy as np


# Use only pre-op / scan-time available variables to reduce leakage risk.
DEFAULT_NUMERIC_FIELDS = [
    "age_at_nephrectomy",
    "bmi",
    "pack_years",
    "radiographic_size",
    "last_preop_egfr.value",
    "last_preop_egfr.days_before_nephrectomy",
]

DEFAULT_CATEGORICAL_FIELDS = [
    "gender",
    "smoking_history",
    "chewing_tobacco_use",
    "alcohol_use",
]

COMORBIDITY_PARENT = "comorbidities"


def get_nested_value(obj, path, default=None):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float_or_nan(value):
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_category(value):
    if value is None:
        return "__MISSING__"
    value = str(value).strip()
    if value == "" or value.lower() in ["nan", "none", "null"]:
        return "__MISSING__"
    return value


class KiTS23ClinicalJSONEncoder:
    """
    Encode kits23.json into numeric vectors for Exp2.

    Fit on train cases only:
        - numeric medians/means/stds
        - categorical vocabularies

    Then transform all cases using train-fitted statistics.
    """

    def __init__(
        self,
        json_path,
        numeric_fields=None,
        categorical_fields=None,
        use_comorbidities=True,
    ):
        self.json_path = json_path
        self.numeric_fields = numeric_fields or DEFAULT_NUMERIC_FIELDS
        self.categorical_fields = categorical_fields or DEFAULT_CATEGORICAL_FIELDS
        self.use_comorbidities = use_comorbidities

        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError("kits23.json must be a list of case dictionaries.")

        self.records = records
        self.case_map = {}
        for case in records:
            case_id = str(case["case_id"])
            self.case_map[case_id] = case

        self.comorbidity_fields = self._collect_comorbidity_fields()

        self.numeric_median = {}
        self.numeric_mean = {}
        self.numeric_std = {}
        self.category_vocab = {}
        self.feature_names = None
        self.is_fitted = False

    def _collect_comorbidity_fields(self):
        fields = set()
        for case in self.records:
            comorbidities = case.get(COMORBIDITY_PARENT, {})
            if isinstance(comorbidities, dict):
                fields.update(comorbidities.keys())
        return sorted(fields)

    def fit(self, train_case_ids):
        train_case_ids = [str(x) for x in train_case_ids]
        missing = [cid for cid in train_case_ids if cid not in self.case_map]
        if missing:
            raise KeyError(f"Missing clinical rows for {len(missing)} train cases. Example: {missing[:5]}")

        for field in self.numeric_fields:
            values = []
            for case_id in train_case_ids:
                case = self.case_map[case_id]
                values.append(to_float_or_nan(get_nested_value(case, field)))

            values = np.asarray(values, dtype=np.float32)

            if np.all(np.isnan(values)):
                median = 0.0
                mean = 0.0
                std = 1.0
            else:
                median = float(np.nanmedian(values))
                filled = np.where(np.isnan(values), median, values)
                mean = float(filled.mean())
                std = float(filled.std())
                if std < 1e-8:
                    std = 1.0

            self.numeric_median[field] = median
            self.numeric_mean[field] = mean
            self.numeric_std[field] = std

        for field in self.categorical_fields:
            values = []
            for case_id in train_case_ids:
                case = self.case_map[case_id]
                values.append(normalize_category(get_nested_value(case, field)))

            vocab = ["__MISSING__", "__UNK__"]
            for value in sorted(set(values)):
                if value not in vocab:
                    vocab.append(value)
            self.category_vocab[field] = vocab

        self.feature_names = self._make_feature_names()
        self.is_fitted = True
        return self

    def _make_feature_names(self):
        names = []

        for field in self.numeric_fields:
            names.append(field)

        if self.use_comorbidities:
            for field in self.comorbidity_fields:
                names.append(f"comorbidities.{field}")

        for field in self.categorical_fields:
            for category in self.category_vocab[field]:
                names.append(f"{field}={category}")

        return names

    def transform_case(self, case_id):
        if not self.is_fitted:
            raise RuntimeError("Call encoder.fit(train_case_ids) before transform_case().")

        case_id = str(case_id)
        if case_id not in self.case_map:
            raise KeyError(f"Case {case_id} not found in clinical JSON.")

        case = self.case_map[case_id]
        features = []

        # Numeric: fill missing by train median, then z-score using train stats.
        for field in self.numeric_fields:
            value = to_float_or_nan(get_nested_value(case, field))
            if np.isnan(value):
                value = self.numeric_median[field]
            value = (value - self.numeric_mean[field]) / (self.numeric_std[field] + 1e-8)
            features.append(float(value))

        # Comorbidities: boolean -> 0/1.
        if self.use_comorbidities:
            comorbidities = case.get(COMORBIDITY_PARENT, {})
            if not isinstance(comorbidities, dict):
                comorbidities = {}
            for field in self.comorbidity_fields:
                features.append(1.0 if bool(comorbidities.get(field, False)) else 0.0)

        # Categorical: one-hot using train vocab. Unknown valid/test category -> __UNK__.
        for field in self.categorical_fields:
            value = normalize_category(get_nested_value(case, field))
            vocab = self.category_vocab[field]
            if value not in vocab:
                value = "__UNK__"

            one_hot = [0.0] * len(vocab)
            one_hot[vocab.index(value)] = 1.0
            features.extend(one_hot)

        return np.asarray(features, dtype=np.float32)

    def transform_all(self):
        if not self.is_fitted:
            raise RuntimeError("Call encoder.fit(train_case_ids) before transform_all().")

        clinical_dict = {}
        for case_id in self.case_map.keys():
            clinical_dict[case_id] = self.transform_case(case_id)
        return clinical_dict
