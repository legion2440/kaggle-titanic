from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import REPORTS_DIR, TEST_PATH, TRAIN_PATH
from scripts.features import DEFERRED_FEATURES, add_clean_features
from scripts.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREPROCESSING_MODES,
    get_feature_sets,
    make_preprocessor,
    split_feature_types,
)


REPORT_PATH = REPORTS_DIR / "check002_preprocessing_layer.md"


def _bool_mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _markdown_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- `{item}`" for item in items)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _to_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def _finite(values: object) -> bool:
    array = _to_array(values)
    return bool(np.isfinite(array).all())


def _output_feature_names(preprocessor: object) -> list[str] | None:
    if not hasattr(preprocessor, "get_feature_names_out"):
        return None
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, ValueError):
        return None


def _present(feature_names: list[str], columns: list[str]) -> bool:
    column_set = set(columns)
    return all(feature in column_set for feature in feature_names)


def _missing(feature_names: list[str], columns: list[str]) -> list[str]:
    column_set = set(columns)
    return [feature for feature in feature_names if feature not in column_set]


def build_report(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, bool]:
    train_features = add_clean_features(train)
    test_features = add_clean_features(test)
    feature_sets = get_feature_sets()

    feature_set_rows = []
    mode_rows = []
    all_passed = True

    for set_name, feature_names in feature_sets.items():
        train_present = _present(feature_names, list(train_features.columns))
        test_present = _present(feature_names, list(test_features.columns))
        numeric_features, categorical_features = split_feature_types(feature_names)
        deferred_present = [
            feature for feature in feature_names if feature in set(DEFERRED_FEATURES)
        ]

        feature_set_passed = train_present and test_present and not deferred_present
        all_passed = all_passed and feature_set_passed

        feature_set_rows.append(
            {
                "feature_set": set_name,
                "features": ", ".join(feature_names),
                "numeric": ", ".join(numeric_features) or "none",
                "categorical": ", ".join(categorical_features) or "none",
                "presence_train": _bool_mark(train_present),
                "presence_test": _bool_mark(test_present),
                "deferred_absent": _bool_mark(not deferred_present),
            }
        )

        if not train_present or not test_present:
            for mode in PREPROCESSING_MODES:
                mode_rows.append(
                    {
                        "feature_set": set_name,
                        "mode": mode,
                        "fit_transform": "SKIP",
                        "train_shape": "n/a",
                        "test_shape": "n/a",
                        "same_output_columns": "n/a",
                        "finite_train": "n/a",
                        "finite_test": "n/a",
                    }
                )
            continue

        train_x = train_features[feature_names]
        test_x = test_features[feature_names]

        for mode in PREPROCESSING_MODES:
            try:
                preprocessor = make_preprocessor(mode, feature_names)
                train_out = preprocessor.fit_transform(train_x)
                train_output_names = _output_feature_names(preprocessor)
                test_out = preprocessor.transform(test_x)
                test_output_names = _output_feature_names(preprocessor)

                train_array = _to_array(train_out)
                test_array = _to_array(test_out)
                if train_output_names is not None and test_output_names is not None:
                    same_output_columns = (
                        train_output_names == test_output_names
                        and len(train_output_names) == train_array.shape[1]
                        and len(test_output_names) == test_array.shape[1]
                    )
                else:
                    same_output_columns = train_array.shape[1] == test_array.shape[1]
                row_counts_ok = (
                    train_array.shape[0] == len(train_x)
                    and test_array.shape[0] == len(test_x)
                )
                finite_train = _finite(train_array)
                finite_test = _finite(test_array)
                fit_transform_passed = same_output_columns and row_counts_ok and finite_train and finite_test
            except Exception as exc:
                all_passed = False
                mode_rows.append(
                    {
                        "feature_set": set_name,
                        "mode": mode,
                        "fit_transform": f"FAIL: {type(exc).__name__}",
                        "train_shape": "n/a",
                        "test_shape": "n/a",
                        "same_output_columns": "FAIL",
                        "finite_train": "FAIL",
                        "finite_test": "FAIL",
                    }
                )
                continue

            all_passed = all_passed and fit_transform_passed
            mode_rows.append(
                {
                    "feature_set": set_name,
                    "mode": mode,
                    "fit_transform": _bool_mark(fit_transform_passed),
                    "train_shape": train_array.shape,
                    "test_shape": test_array.shape,
                    "same_output_columns": _bool_mark(same_output_columns),
                    "finite_train": _bool_mark(finite_train),
                    "finite_test": _bool_mark(finite_test),
                }
            )

    lines = [
        "# CHECK002 Preprocessing Layer",
        "",
        "This report records the first clean preprocessing layer implementation.",
        "It does not run model training, benchmarking, CV, OOF, holdout, F01, BASE_1, or submission generation.",
        "",
        "## Overall Status",
        "",
        f"- overall: `{_bool_mark(all_passed)}`",
        f"- train shape raw: `{train.shape}`",
        f"- test shape raw: `{test.shape}`",
        f"- train shape after `add_clean_features`: `{train_features.shape}`",
        f"- test shape after `add_clean_features`: `{test_features.shape}`",
        "",
        "## Preprocessing Modes",
        "",
        "- `scaled_linear`: numeric median imputation + scaling; categorical most_frequent imputation + one-hot encoding",
        "- `unscaled_tree`: numeric median imputation without scaling; categorical most_frequent imputation + one-hot encoding",
        "",
        "## Current Feature Typing",
        "",
        "Numeric features:",
        "",
        _markdown_list(NUMERIC_FEATURES),
        "",
        "Categorical features:",
        "",
        _markdown_list(CATEGORICAL_FEATURES),
        "",
        "## Agreed Feature Sets For Next Step",
        "",
        _markdown_table(
            feature_set_rows,
            [
                "feature_set",
                "features",
                "numeric",
                "categorical",
                "presence_train",
                "presence_test",
                "deferred_absent",
            ],
        ),
        "",
        "## Preprocessing Checks",
        "",
        _markdown_table(
            mode_rows,
            [
                "feature_set",
                "mode",
                "fit_transform",
                "train_shape",
                "test_shape",
                "same_output_columns",
                "finite_train",
                "finite_test",
            ],
        ),
        "",
        "## Scope Boundary",
        "",
        "- no model training",
        "- no benchmark",
        "- no CV / OOF",
        "- no holdout",
        "- no F01",
        "- no BASE_1",
        "- no submissions",
    ]

    return "\n".join(lines) + "\n", all_passed


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    report, passed = build_report(train, test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"wrote {REPORT_PATH.relative_to(Path.cwd())}")
    print(f"overall: {_bool_mark(passed)}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
