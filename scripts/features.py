import re

import numpy as np
import pandas as pd


F00_CORE = ["Sex", "Pclass", "Embarked"]
RAW_TABULAR = ["Sex", "Pclass", "Embarked", "Age", "SibSp", "Parch", "Fare"]

CLEAN_DERIVED = [
    "Title",
    "AgeMissing",
    "AgeBin",
    "IsChild12",
    "FamilySize",
    "FamilySizeBucket",
    "FareLog",
    "CabinKnown",
]

SECOND_ORDER_FEATURE_ORDER = [
    "Title",
    "AgeMissing",
    "AgeBin",
    "IsChild12",
    "FamilySize",
    "FamilySizeBucket",
    "FareLog",
    "CabinKnown",
]

DEFERRED_FEATURES = [
    "Deck",
    "TicketPrefix",
    "FarePerPerson",
    "WomanOrChild",
    "IsAdultMale",
    "SexAgeBin",
    "PclassSex",
    "PclassSexAgeBin",
]

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
COMMON_TITLES = {"Mr", "Miss", "Mrs", "Master"}


def extract_title(name: object) -> str:
    if pd.isna(name):
        return "Unknown"

    match = re.search(r",\s*([^\.]+)\.", str(name))
    if not match:
        return "Unknown"

    raw_title = match.group(1).strip()
    normalized = TITLE_MAP.get(raw_title, raw_title)
    if normalized in COMMON_TITLES:
        return normalized
    return "Rare"


def bucket_family_size(size: object) -> str:
    if pd.isna(size):
        return "unknown"

    value = int(size)
    if value == 1:
        return "alone"
    if value <= 4:
        return "small"
    if value <= 6:
        return "medium"
    return "large"


def age_bin_12(age: object) -> str:
    if pd.isna(age):
        return "missing"

    value = float(age)
    if value < 12:
        return "child_lt_12"
    if value < 18:
        return "teen_12_17"
    return "adult_18_plus"


def add_clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    out["Title"] = out["Name"].map(extract_title)
    out["AgeMissing"] = out["Age"].isna().astype(int)
    out["AgeBin"] = out["Age"].map(age_bin_12)
    out["IsChild12"] = ((out["Age"].notna()) & (out["Age"] < 12)).astype(int)

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["FamilySizeBucket"] = out["FamilySize"].map(bucket_family_size)

    out["FareLog"] = np.log1p(out["Fare"])
    out["CabinKnown"] = out["Cabin"].notna().astype(int)

    return out
