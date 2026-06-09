from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.config import REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import (
    CLEAN_DERIVED,
    DEFERRED_FEATURES,
    F00_CORE,
    RAW_TABULAR,
    SECOND_ORDER_FEATURE_ORDER,
    add_clean_features,
)


REPORT_PATH = REPORTS_DIR / "check001_feature_contract.md"
TARGET_DERIVED_MARKERS = [
    "SurvivalRate",
    "TargetRate",
    "SurvivedRate",
    "GroupSurvival",
    "TargetEncoded",
]


def _without_target(columns: list[str]) -> list[str]:
    return [column for column in columns if column != TARGET]


def _missing(columns: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [column for column in columns if column not in available_set]


def _present(columns: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [column for column in columns if column in available_set]


def _bool_mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _markdown_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- `{item}`" for item in items)


def build_report(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, dict[str, object]]:
    train_features = add_clean_features(train)
    test_features = add_clean_features(test)

    train_compare_columns = _without_target(list(train_features.columns))
    test_compare_columns = list(test_features.columns)
    same_column_order = train_compare_columns == test_compare_columns

    required_first_order = list(dict.fromkeys(F00_CORE + RAW_TABULAR))
    required_second_order = CLEAN_DERIVED
    train_columns = list(train_features.columns)
    test_columns = list(test_features.columns)

    missing_first_train = _missing(required_first_order, train_columns)
    missing_first_test = _missing(required_first_order, test_columns)
    missing_second_train = _missing(required_second_order, train_columns)
    missing_second_test = _missing(required_second_order, test_columns)
    deferred_present_train = _present(DEFERRED_FEATURES, train_columns)
    deferred_present_test = _present(DEFERRED_FEATURES, test_columns)
    target_derived_present = [
        column
        for column in train_columns
        if any(marker.lower() in column.lower() for marker in TARGET_DERIVED_MARKERS)
    ]

    checks = {
        "same_column_order_excluding_target": same_column_order,
        "first_order_present_train": not missing_first_train,
        "first_order_present_test": not missing_first_test,
        "second_order_present_train": not missing_second_train,
        "second_order_present_test": not missing_second_test,
        "deferred_absent_train": not deferred_present_train,
        "deferred_absent_test": not deferred_present_test,
        "target_derived_absent": not target_derived_present,
    }
    overall_pass = all(checks.values())

    lines = [
        "# CHECK001 Feature Contract",
        "",
        "This report records the first clean script layer for feature order and preprocessing contract.",
        "It does not run model training, CV, OOF, holdout, benchmarking, or submission generation.",
        "",
        "## Contract Check Status",
        "",
        f"- overall: `{_bool_mark(overall_pass)}`",
        f"- train shape raw: `{train.shape}`",
        f"- test shape raw: `{test.shape}`",
        f"- train shape after `add_clean_features`: `{train_features.shape}`",
        f"- test shape after `add_clean_features`: `{test_features.shape}`",
        f"- train/test same column order after feature engineering, excluding `{TARGET}`: `{_bool_mark(same_column_order)}`",
        "",
        "## Feature Check Order",
        "",
        "### First-Order",
        "",
        f"- F00_CORE = {', '.join(F00_CORE)}",
        f"- RAW_TABULAR = {', '.join(RAW_TABULAR)}",
        "",
        "### Second-Order",
        "",
        "- + Title",
        "- + AgeMissing",
        "- + AgeBin / IsChild12",
        "- + FamilySize / FamilySizeBucket",
        "- + FareLog",
        "- + CabinKnown",
        "",
        "Second-order feature order:",
        "",
        _markdown_list(SECOND_ORDER_FEATURE_ORDER),
        "",
        "### Deferred / Not In Current Step",
        "",
        "- Deck",
        "- TicketPrefix",
        "- FarePerPerson",
        "- WomanOrChild",
        "- IsAdultMale",
        "- SexAgeBin",
        "- PclassSex",
        "- PclassSexAgeBin",
        "- target-derived logic",
        "",
        "## Presence / Absence Checks",
        "",
        f"- first-order present in train: `{_bool_mark(not missing_first_train)}`",
        f"- first-order present in test: `{_bool_mark(not missing_first_test)}`",
        f"- second-order present in train: `{_bool_mark(not missing_second_train)}`",
        f"- second-order present in test: `{_bool_mark(not missing_second_test)}`",
        f"- deferred features absent in train: `{_bool_mark(not deferred_present_train)}`",
        f"- deferred features absent in test: `{_bool_mark(not deferred_present_test)}`",
        f"- target-derived markers absent: `{_bool_mark(not target_derived_present)}`",
        "",
        "Missing first-order train:",
        "",
        _markdown_list(missing_first_train),
        "",
        "Missing first-order test:",
        "",
        _markdown_list(missing_first_test),
        "",
        "Missing second-order train:",
        "",
        _markdown_list(missing_second_train),
        "",
        "Missing second-order test:",
        "",
        _markdown_list(missing_second_test),
        "",
        "Deferred features unexpectedly present in train:",
        "",
        _markdown_list(deferred_present_train),
        "",
        "Deferred features unexpectedly present in test:",
        "",
        _markdown_list(deferred_present_test),
        "",
        "Target-derived markers unexpectedly present:",
        "",
        _markdown_list(target_derived_present),
        "",
        "## Preprocessing Contract",
        "",
        "- numeric features -> median imputation",
        "- categorical features -> most_frequent imputation",
        "- one-hot encoding only for models that need it",
        "- scaling only for linear / SVM / KNN",
        "- no scaling as default contract for tree models",
        "- all fit operations only inside train folds later",
        "- no target-derived features",
        "- no `gender_submission.csv` as truth",
        "- no test-side lookup logic",
        "",
        "## Scope Boundary",
        "",
        "- no model training",
        "- no benchmark models",
        "- no CV / OOF code",
        "- no holdout logic",
        "- no `train.py` / `predict.py`",
        "- no F01",
        "- no BASE_1",
        "- no submission generation",
    ]

    return "\n".join(lines) + "\n", checks


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    report, checks = build_report(train, test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"wrote {REPORT_PATH.relative_to(Path.cwd())}")
    for name, passed in checks.items():
        print(f"{name}: {_bool_mark(bool(passed))}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
