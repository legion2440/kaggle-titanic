from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.features import F00_CORE, RAW_TABULAR


NUMERIC_FEATURES = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "FareLog",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Pclass",
    "Embarked",
    "Title",
    "AgeBin",
    "FamilySizeBucket",
]

# Binary 0/1 flags — no imputation needed (never NaN), no scaling (already bounded).
# Passed through as-is in both scaled_linear and unscaled_tree modes.
# Scripts that create local binary-like features (e.g. 11/12 AgeBucket) must register
# them dynamically before calling make_preprocessor, following the pattern in scripts 17/18.
BINARY_FEATURES = [
    "AgeMissing",
    "IsChild12",
    "CabinKnown",
]

PREPROCESSING_MODES = ["scaled_linear", "unscaled_tree"]

FEATURE_SETS = {
    "f00_core": F00_CORE,
    "raw_tabular": RAW_TABULAR,
    "raw_plus_title": [*RAW_TABULAR, "Title"],
    "raw_plus_agemissing": [*RAW_TABULAR, "AgeMissing"],
    "raw_plus_agebin_child12": [*RAW_TABULAR, "AgeBin", "IsChild12"],
    "raw_plus_family": [*RAW_TABULAR, "FamilySizeBucket"],
    "raw_plus_farelog": [*RAW_TABULAR, "FareLog"],
    "raw_plus_cabinknown": [*RAW_TABULAR, "CabinKnown"],
}

FEATURE_LAYERS = {
    "first_order": {
        "f00_core": F00_CORE,
        "raw_tabular": RAW_TABULAR,
    },
    "second_order": {
        "raw_plus_title": FEATURE_SETS["raw_plus_title"],
        "raw_plus_agemissing": FEATURE_SETS["raw_plus_agemissing"],
        "raw_plus_agebin_child12": FEATURE_SETS["raw_plus_agebin_child12"],
        "raw_plus_family": FEATURE_SETS["raw_plus_family"],
        "raw_plus_farelog": FEATURE_SETS["raw_plus_farelog"],
        "raw_plus_cabinknown": FEATURE_SETS["raw_plus_cabinknown"],
    },
}


def get_feature_layers() -> dict[str, dict[str, list[str]]]:
    return {
        layer_name: {set_name: list(features) for set_name, features in layer_sets.items()}
        for layer_name, layer_sets in FEATURE_LAYERS.items()
    }


def get_feature_sets() -> dict[str, list[str]]:
    return {name: list(features) for name, features in FEATURE_SETS.items()}


def split_feature_types(
    feature_names: list[str],
) -> tuple[list[str], list[str], list[str]]:
    numeric_set = set(NUMERIC_FEATURES)
    categorical_set = set(CATEGORICAL_FEATURES)
    binary_set = set(BINARY_FEATURES)
    known = numeric_set | categorical_set | binary_set

    unknown = [f for f in feature_names if f not in known]
    if unknown:
        raise ValueError(f"Unknown feature type for: {unknown}")

    numeric = [f for f in feature_names if f in numeric_set]
    categorical = [f for f in feature_names if f in categorical_set]
    binary = [f for f in feature_names if f in binary_set]
    return numeric, categorical, binary


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mode: str, feature_names: list[str]) -> ColumnTransformer:
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    numeric_features, categorical_features, binary_features = split_feature_types(feature_names)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if mode == "scaled_linear":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ]

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_features))
    if binary_features:
        # Binary 0/1 flags are always finite and never NaN — pass through unchanged.
        transformers.append(("binary", "passthrough", binary_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
