from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, REPORTS_DIR, TARGET, TRAIN_PATH
from scripts.features import F00_CORE, RAW_TABULAR
from scripts.preprocessing import make_preprocessor


warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

REPORT_PATH = REPORTS_DIR / "04_baseline.md"
CSV_PATH = REPORTS_DIR / "04_baseline.csv"

FEATURE_SETS = {
    "f00_core": F00_CORE,
    "raw_tabular": RAW_TABULAR,
}

RESULT_COLUMNS = [
    "status",
    "feature_set",
    "model",
    "preprocessing_mode",
    "cv_mean",
    "cv_std",
    "cv_min",
    "cv_max",
    "fold_scores",
    "error",
]

PAIRED_COLUMNS = [
    "model",
    "f00_core_cv_mean",
    "raw_tabular_cv_mean",
    "delta_raw_minus_f00",
    "result",
]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    package: str
    version_package: str
    module: str
    class_name: str
    preprocessing_mode: str
    explicit_params: dict[str, Any]


MODEL_SPECS = [
    ModelSpec(
        model="DummyClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.dummy",
        class_name="DummyClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"strategy": "most_frequent"},
    ),
    ModelSpec(
        model="LogisticRegression",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.linear_model",
        class_name="LogisticRegression",
        preprocessing_mode="scaled_linear",
        explicit_params={"max_iter": 1000, "random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="GaussianNB",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.naive_bayes",
        class_name="GaussianNB",
        preprocessing_mode="unscaled_tree",
        explicit_params={},
    ),
    ModelSpec(
        model="KNeighborsClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.neighbors",
        class_name="KNeighborsClassifier",
        preprocessing_mode="scaled_linear",
        explicit_params={},
    ),
    ModelSpec(
        model="LinearSVC",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.svm",
        class_name="LinearSVC",
        preprocessing_mode="scaled_linear",
        explicit_params={"max_iter": 5000, "random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="SVC",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.svm",
        class_name="SVC",
        preprocessing_mode="scaled_linear",
        explicit_params={"random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="DecisionTreeClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.tree",
        class_name="DecisionTreeClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="RandomForestClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.ensemble",
        class_name="RandomForestClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE, "n_jobs": 1},
    ),
    ModelSpec(
        model="ExtraTreesClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.ensemble",
        class_name="ExtraTreesClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE, "n_jobs": 1},
    ),
    ModelSpec(
        model="AdaBoostClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.ensemble",
        class_name="AdaBoostClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="GradientBoostingClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.ensemble",
        class_name="GradientBoostingClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="HistGradientBoostingClassifier",
        package="scikit-learn",
        version_package="scikit-learn",
        module="sklearn.ensemble",
        class_name="HistGradientBoostingClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={"random_state": RANDOM_STATE},
    ),
    ModelSpec(
        model="XGBClassifier",
        package="xgboost",
        version_package="xgboost",
        module="xgboost",
        class_name="XGBClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
    ),
    ModelSpec(
        model="LGBMClassifier",
        package="lightgbm",
        version_package="lightgbm",
        module="lightgbm",
        class_name="LGBMClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        },
    ),
    ModelSpec(
        model="CatBoostClassifier",
        package="catboost",
        version_package="catboost",
        module="catboost",
        class_name="CatBoostClassifier",
        preprocessing_mode="unscaled_tree",
        explicit_params={
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
        },
    ),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=True)


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _markdown_escape(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(_markdown_escape(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _unsupported_kwarg(error: TypeError) -> str | None:
    marker = "unexpected keyword argument "
    message = str(error)
    if marker not in message:
        return None
    return message.split(marker, 1)[1].strip().strip("'\"")


def _build_model(spec: ModelSpec) -> tuple[object | None, dict[str, Any], str]:
    try:
        module = importlib.import_module(spec.module)
        model_class = getattr(module, spec.class_name)
    except Exception as exc:
        return None, {}, f"{type(exc).__name__}: {exc}"

    params = dict(spec.explicit_params)
    removed_params = []
    while True:
        try:
            model = model_class(**params)
            if removed_params:
                adjustment = "removed unsupported technical params: " + ", ".join(removed_params)
            else:
                adjustment = ""
            return model, params, adjustment
        except TypeError as exc:
            unsupported = _unsupported_kwarg(exc)
            if unsupported and unsupported in params:
                removed_params.append(unsupported)
                params.pop(unsupported)
                continue
            return None, params, f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return None, params, f"{type(exc).__name__}: {exc}"


def _model_panel_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, Any]]]:
    rows = []
    resolved: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        model, used_params, error_or_adjustment = _build_model(spec)
        package_version = _package_version(spec.version_package)
        if model is not None and hasattr(model, "get_params"):
            actual_params = model.get_params(deep=False)
        elif model is not None:
            actual_params = "get_params_unavailable"
        else:
            actual_params = "model_unavailable"

        is_error = model is None
        adjustment = "" if is_error else error_or_adjustment
        error = error_or_adjustment if is_error else ""
        rows.append(
            {
                "model_class": spec.class_name,
                "package": spec.package,
                "package_version": package_version,
                "preprocessing_mode": spec.preprocessing_mode,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": adjustment,
                "error": error,
            }
        )
        resolved[spec.model] = {
            "spec": spec,
            "used_params": used_params,
            "model_available": model is not None,
            "model_error": error,
        }

    return rows, resolved


def _evaluate(
    train: pd.DataFrame,
    resolved_models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, object]], bool]:
    missing_target = TARGET not in train.columns
    y = train[TARGET] if not missing_target else pd.Series(dtype=int)
    splits = []
    if not missing_target:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(np.zeros(len(train)), y))

    rows = []
    all_passed = True

    for feature_set_name, feature_names in FEATURE_SETS.items():
        missing_features = [feature for feature in feature_names if feature not in train.columns]
        if missing_target:
            feature_set_error = f"missing target column: {TARGET}"
        elif missing_features:
            feature_set_error = "missing feature columns: " + ", ".join(missing_features)
        else:
            feature_set_error = ""

        for spec in MODEL_SPECS:
            resolved = resolved_models[spec.model]
            base_row = {
                "feature_set": feature_set_name,
                "model": spec.model,
                "preprocessing_mode": spec.preprocessing_mode,
            }

            if feature_set_error:
                all_passed = False
                rows.append(
                    {
                        "status": "fail",
                        **base_row,
                        "cv_mean": "",
                        "cv_std": "",
                        "cv_min": "",
                        "cv_max": "",
                        "fold_scores": "[]",
                        "error": feature_set_error,
                    }
                )
                continue

            if not resolved["model_available"]:
                all_passed = False
                rows.append(
                    {
                        "status": "fail",
                        **base_row,
                        "cv_mean": "",
                        "cv_std": "",
                        "cv_min": "",
                        "cv_max": "",
                        "fold_scores": "[]",
                        "error": resolved["model_error"],
                    }
                )
                continue

            try:
                model, _, build_error = _build_model(spec)
                if model is None:
                    raise RuntimeError(build_error)
                estimator = Pipeline(
                    steps=[
                        ("preprocess", make_preprocessor(spec.preprocessing_mode, list(feature_names))),
                        ("model", model),
                    ]
                )
                scores = cross_val_score(
                    estimator,
                    train[list(feature_names)],
                    y,
                    cv=splits,
                    scoring="accuracy",
                    error_score="raise",
                )
                rounded_scores = [_round_float(score) for score in scores]
                rows.append(
                    {
                        "status": "ok",
                        **base_row,
                        "cv_mean": _round_float(np.mean(scores)),
                        "cv_std": _round_float(np.std(scores)),
                        "cv_min": _round_float(np.min(scores)),
                        "cv_max": _round_float(np.max(scores)),
                        "fold_scores": _json_dumps(rounded_scores),
                        "error": "",
                    }
                )
            except Exception as exc:
                all_passed = False
                rows.append(
                    {
                        "status": "fail",
                        **base_row,
                        "cv_mean": "",
                        "cv_std": "",
                        "cv_min": "",
                        "cv_max": "",
                        "fold_scores": "[]",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return rows, all_passed


def _best_row(rows: list[dict[str, object]], feature_set: str | None = None) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if row["status"] == "ok"
        and row["cv_mean"] != ""
        and (feature_set is None or row["feature_set"] == feature_set)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["cv_mean"]))


def _best_summary(best: dict[str, object] | None) -> str:
    if best is None:
        return "n/a"
    return f"{best['model']} ({best['cv_mean']})"


def _comparison_summary(rows: list[dict[str, object]]) -> list[str]:
    best_f00 = _best_row(rows, "f00_core")
    best_raw = _best_row(rows, "raw_tabular")
    lines = [
        f"- best model on `f00_core`: {_best_summary(best_f00)}",
        f"- best model on `raw_tabular`: {_best_summary(best_raw)}",
    ]
    if best_f00 is None or best_raw is None:
        lines.append("- whether `raw_tabular` improves over `f00_core`: n/a")
        return lines

    delta = float(best_raw["cv_mean"]) - float(best_f00["cv_mean"])
    if delta > 0:
        status = "yes"
    elif delta < 0:
        status = "no"
    else:
        status = "tie"
    lines.append(
        f"- whether `raw_tabular` improves over `f00_core`: {status} "
        f"(delta={_round_float(delta)})"
    )
    return lines


def _paired_feature_set_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_model_feature_set = {
        (row["model"], row["feature_set"]): row
        for row in rows
        if row["status"] == "ok" and row["cv_mean"] != ""
    }
    paired_rows = []

    for spec in MODEL_SPECS:
        f00 = rows_by_model_feature_set.get((spec.model, "f00_core"))
        raw = rows_by_model_feature_set.get((spec.model, "raw_tabular"))
        if f00 is None or raw is None:
            paired_rows.append(
                {
                    "model": spec.model,
                    "f00_core_cv_mean": "" if f00 is None else f00["cv_mean"],
                    "raw_tabular_cv_mean": "" if raw is None else raw["cv_mean"],
                    "delta_raw_minus_f00": "",
                    "result": "n/a",
                }
            )
            continue

        delta = _round_float(float(raw["cv_mean"]) - float(f00["cv_mean"]))
        if delta > 0:
            result = "improved"
        elif delta < 0:
            result = "worsened"
        else:
            result = "tied"
        paired_rows.append(
            {
                "model": spec.model,
                "f00_core_cv_mean": f00["cv_mean"],
                "raw_tabular_cv_mean": raw["cv_mean"],
                "delta_raw_minus_f00": delta,
                "result": result,
            }
        )

    return paired_rows


def _paired_feature_set_summary(rows: list[dict[str, object]]) -> list[str]:
    improved = [row for row in rows if row["result"] == "improved"]
    worsened = [row for row in rows if row["result"] == "worsened"]
    tied = [row for row in rows if row["result"] == "tied"]
    positive = [
        row for row in rows
        if row["delta_raw_minus_f00"] != "" and float(row["delta_raw_minus_f00"]) > 0
    ]
    negative = [
        row for row in rows
        if row["delta_raw_minus_f00"] != "" and float(row["delta_raw_minus_f00"]) < 0
    ]
    best_positive = max(positive, key=lambda row: float(row["delta_raw_minus_f00"]), default=None)
    worst_negative = min(negative, key=lambda row: float(row["delta_raw_minus_f00"]), default=None)

    return [
        f"- improved: `{len(improved)}`",
        f"- worsened: `{len(worsened)}`",
        f"- tied: `{len(tied)}`",
        "- best positive delta: "
        + (
            "`n/a`"
            if best_positive is None
            else f"`{best_positive['model']} ({best_positive['delta_raw_minus_f00']})`"
        ),
        "- worst negative delta: "
        + (
            "`n/a`"
            if worst_negative is None
            else f"`{worst_negative['model']} ({worst_negative['delta_raw_minus_f00']})`"
        ),
    ]


def _build_report(
    train: pd.DataFrame,
    panel_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    best = _best_row(result_rows)
    best_rows = [best] if best is not None else []
    paired_rows = _paired_feature_set_rows(result_rows)
    status = "PASS" if all_passed else "FAIL"

    lines = [
        "# 04 Baseline",
        "",
        "## Scope Boundary",
        "",
        "- train-side CV only",
        "- `train.csv` only",
        "- `Survived` is used only as the target",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- no submission generation",
        "- no Kaggle/public leaderboard use",
        "- no `test.csv` scoring",
        "- no test labels or row-level correctness checks",
        "- no `gender_submission.csv` as truth",
        "- no feature engineering",
        "- no hyperparameter tuning",
        "- no threshold tuning",
        "- no final model selection",
        "",
        "## Feature Sets",
        "",
        _markdown_table(
            [
                {"feature_set": name, "features": ", ".join(features)}
                for name, features in FEATURE_SETS.items()
            ],
            ["feature_set", "features"],
        ),
        "",
        "## CV Protocol",
        "",
        f"- splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})`",
        "- metric: `accuracy`",
        "- identical precomputed CV split indices are reused for every model and feature set",
        "- preprocessing is fitted inside each train fold through an sklearn `Pipeline`",
        f"- rows: `{len(train)}` from `train.csv`",
        "",
        "## Model Panel",
        "",
        _markdown_table(
            panel_rows,
            [
                "model_class",
                "package",
                "package_version",
                "preprocessing_mode",
                "explicit_technical_params",
                "actual_resolved_params",
                "parameter_adjustments",
                "error",
            ],
        ),
        "",
        "## All Results",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(result_rows, RESULT_COLUMNS),
        "",
        "## Best row by `cv_mean`",
        "",
        _markdown_table(best_rows, RESULT_COLUMNS) if best_rows else "No successful rows.",
        "",
        "## Paired feature-set comparison",
        "",
        "This compares `raw_tabular` against `f00_core` within each model class.",
        "",
        _markdown_table(paired_rows, PAIRED_COLUMNS),
        "",
        *(_paired_feature_set_summary(paired_rows)),
        "",
        "## Comparison summary",
        "",
        *(_comparison_summary(result_rows)),
        "",
        "## Short interpretation",
        "",
        "- This is baseline evidence only.",
        "- This is not final model selection.",
        "- No feature engineering or tuning was used.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    panel_rows, resolved_models = _model_panel_rows()
    result_rows, all_passed = _evaluate(train, resolved_models)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(result_rows)
    REPORT_PATH.write_text(
        _build_report(train, panel_rows, result_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
