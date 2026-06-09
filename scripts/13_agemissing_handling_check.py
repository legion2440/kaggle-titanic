from __future__ import annotations

import csv
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, REPORTS_DIR, TARGET, TRAIN_PATH
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "13_agemissing_handling_check.md"
CSV_PATH = REPORTS_DIR / "13_agemissing_handling_check.csv"

RAW_TABULAR = ["Sex", "Pclass", "Embarked", "Age", "SibSp", "Parch", "Fare"]
RAW_PLUS_AGEMISSING = [
    "Sex",
    "Pclass",
    "Embarked",
    "Age",
    "AgeMissing",
    "SibSp",
    "Parch",
    "Fare",
]
HISTGB_CATEGORICAL_FEATURES = ["Sex", "Pclass", "Embarked"]
CLOSE_TOLERANCE = 0.0025

CSV_COLUMNS = [
    "model_name",
    "variant",
    "feature_set",
    "preprocessing_strategy",
    "cv_mean",
    "cv_std",
    "oof_accuracy",
    "pred_1_rate",
    "base_variant",
    "changed_predictions_vs_base",
    "changed_pct_vs_base",
    "rescue_count",
    "kill_count",
    "net_correct_delta",
    "status",
]

MODEL_PANEL_COLUMNS = [
    "variant",
    "model_class",
    "package",
    "package_version",
    "preprocessing_strategy",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

VARIANT_COLUMNS = [
    "variant",
    "model_name",
    "feature_set",
    "features",
    "preprocessing_strategy",
    "purpose",
]


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    model_name: str
    feature_set: str
    features: list[str]
    preprocessing_strategy: str
    base_variant: str
    purpose: str


VARIANTS = [
    VariantSpec(
        variant="gb_raw_tabular_median",
        model_name="GradientBoostingClassifier",
        feature_set="raw_tabular",
        features=RAW_TABULAR,
        preprocessing_strategy=(
            "existing unscaled_tree preprocessing: numeric median imputation; "
            "categorical most_frequent + one-hot"
        ),
        base_variant="gb_raw_tabular_median",
        purpose="baseline reference for current raw_tabular GB lane",
    ),
    VariantSpec(
        variant="gb_raw_plus_agemissing_median",
        model_name="GradientBoostingClassifier",
        feature_set="raw_plus_agemissing",
        features=RAW_PLUS_AGEMISSING,
        preprocessing_strategy=(
            "AgeMissing created before imputation; Age remains median-imputed; "
            "categorical handling matches baseline"
        ),
        base_variant="gb_raw_tabular_median",
        purpose="check explicit missingness signal on top of median-imputed Age",
    ),
    VariantSpec(
        variant="gb_raw_age_sentinel_plus_agemissing",
        model_name="GradientBoostingClassifier",
        feature_set="raw_plus_agemissing",
        features=RAW_PLUS_AGEMISSING,
        preprocessing_strategy=(
            "AgeMissing created before fill; missing Age replaced with sentinel -1; "
            "other numeric missing handling remains safe"
        ),
        base_variant="gb_raw_tabular_median",
        purpose="check whether median pseudo-age is hurting GB while preserving known Age",
    ),
    VariantSpec(
        variant="histgb_raw_tabular_nan_native",
        model_name="HistGradientBoostingClassifier",
        feature_set="raw_tabular",
        features=RAW_TABULAR,
        preprocessing_strategy=(
            "native NaN lane: Age kept as NaN; pandas categorical dtypes with "
            'categorical_features="from_dtype"'
        ),
        base_variant="histgb_raw_tabular_nan_native",
        purpose="diagnostic native-NaN reference without explicit AgeMissing",
    ),
    VariantSpec(
        variant="histgb_raw_plus_agemissing_nan_native",
        model_name="HistGradientBoostingClassifier",
        feature_set="raw_plus_agemissing",
        features=RAW_PLUS_AGEMISSING,
        preprocessing_strategy=(
            "native NaN lane: Age kept as NaN; AgeMissing added; categorical handling "
            "matches HistGB native-NaN reference"
        ),
        base_variant="histgb_raw_tabular_nan_native",
        purpose="diagnostic check for AgeMissing on top of native NaN handling",
    ),
]


def _load_baseline04() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "04_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline04", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline04"] = module
    spec.loader.exec_module(module)
    return module


baseline04 = _load_baseline04()
MODEL_SPECS_BY_NAME = {spec.model: spec for spec in baseline04.MODEL_SPECS}


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


class AgeSentinelTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y: object = None) -> "AgeSentinelTransformer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        out["Age"] = out["Age"].fillna(-1)
        return out


class HistGBNativeNanPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], categorical_features: list[str]) -> None:
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.categories_: dict[str, list[object]] = {}

    def fit(self, x: pd.DataFrame, y: object = None) -> "HistGBNativeNanPreprocessor":
        self.categories_ = {}
        for feature in self.categorical_features:
            if feature in self.features:
                self.categories_[feature] = sorted(
                    value for value in x[feature].dropna().unique().tolist()
                )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x[self.features].copy()
        for feature in self.categorical_features:
            if feature in out.columns:
                out[feature] = pd.Categorical(out[feature], categories=self.categories_[feature])
        return out


class HistGBOrdinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], categorical_features: list[str]) -> None:
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.maps_: dict[str, dict[object, int]] = {}

    def fit(self, x: pd.DataFrame, y: object = None) -> "HistGBOrdinalPreprocessor":
        self.maps_ = {}
        for feature in self.categorical_features:
            if feature in self.features:
                categories = sorted(value for value in x[feature].dropna().unique().tolist())
                self.maps_[feature] = {category: code for code, category in enumerate(categories)}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x[self.features].copy()
        for feature in self.categorical_features:
            if feature in out.columns:
                out[feature] = out[feature].map(self.maps_[feature]).fillna(-1).astype(float)
        return out


def _build_base_model(model_name: str) -> tuple[object | None, dict[str, Any], str]:
    spec = MODEL_SPECS_BY_NAME[model_name]
    return baseline04._build_model(spec)


def _histgb_categorical_mask(features: list[str]) -> list[bool]:
    categorical_set = set(HISTGB_CATEGORICAL_FEATURES)
    return [feature in categorical_set for feature in features]


def _build_estimator(
    variant: VariantSpec,
    histgb_strategy: str,
) -> tuple[Pipeline | None, dict[str, Any], str]:
    model, used_params, adjustment = _build_base_model(variant.model_name)
    if model is None:
        return None, used_params, adjustment

    if variant.model_name == "GradientBoostingClassifier":
        steps: list[tuple[str, object]] = []
        if variant.variant == "gb_raw_age_sentinel_plus_agemissing":
            steps.append(("age_sentinel", AgeSentinelTransformer()))
        steps.extend(
            [
                ("preprocess", make_preprocessor("unscaled_tree", list(variant.features))),
                ("model", model),
            ]
        )
        return Pipeline(steps=steps), used_params, adjustment

    if variant.model_name != "HistGradientBoostingClassifier":
        return None, used_params, f"Unsupported model for this check: {variant.model_name}"

    if histgb_strategy == "from_dtype":
        model.set_params(categorical_features="from_dtype")
        used_params = {**used_params, "categorical_features": "from_dtype"}
        return (
            Pipeline(
                steps=[
                    (
                        "preprocess",
                        HistGBNativeNanPreprocessor(
                            list(variant.features),
                            HISTGB_CATEGORICAL_FEATURES,
                        ),
                    ),
                    ("model", model),
                ]
            ),
            used_params,
            adjustment,
        )

    model.set_params(categorical_features=_histgb_categorical_mask(variant.features))
    used_params = {
        **used_params,
        "categorical_features": _histgb_categorical_mask(variant.features),
    }
    return (
        Pipeline(
            steps=[
                (
                    "preprocess",
                    HistGBOrdinalPreprocessor(list(variant.features), HISTGB_CATEGORICAL_FEATURES),
                ),
                ("model", model),
            ]
        ),
        used_params,
        (adjustment + "; " if adjustment else "")
        + 'fallback categorical handling: ordinal codes; missing/unknown encoded as -1',
    )


def _add_agemissing(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["AgeMissing"] = out["Age"].isna().astype(int)
    return out


def _variant_by_name(name: str) -> VariantSpec:
    for variant in VARIANTS:
        if variant.variant == name:
            return variant
    raise KeyError(name)


def _evaluate_variant(
    variant: VariantSpec,
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
    histgb_strategy: str,
) -> tuple[dict[str, object], str]:
    missing_features = [feature for feature in variant.features if feature not in train.columns]
    if missing_features:
        return (
            {
                "variant": variant.variant,
                "status": "fail",
                "error": "missing feature columns: " + ", ".join(missing_features),
            },
            histgb_strategy,
        )

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)
    effective_strategy = histgb_strategy

    try:
        for i, (train_idx, valid_idx) in enumerate(splits):
            estimator, _, build_error = _build_estimator(variant, effective_strategy)
            if estimator is None:
                raise RuntimeError(build_error)
            x = train[variant.features]
            estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
            fold_pred = estimator.predict(x.iloc[valid_idx]).astype(int)
            if i < 5:
                oof[valid_idx] = fold_pred
            fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))
    except Exception as exc:
        if variant.model_name == "HistGradientBoostingClassifier" and effective_strategy == "from_dtype":
            effective_strategy = "ordinal_fallback"
            fold_scores = []
            oof = np.full(len(train), -1, dtype=int)
            try:
                for i, (train_idx, valid_idx) in enumerate(splits):
                    estimator, _, build_error = _build_estimator(variant, effective_strategy)
                    if estimator is None:
                        raise RuntimeError(build_error)
                    x = train[variant.features]
                    estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
                    fold_pred = estimator.predict(x.iloc[valid_idx]).astype(int)
                    if i < 5:
                        oof[valid_idx] = fold_pred
                    fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))
            except Exception as fallback_exc:
                return (
                    {
                        "variant": variant.variant,
                        "status": "fail",
                        "error": (
                            f"from_dtype failed with {type(exc).__name__}: {exc}; "
                            f"ordinal fallback failed with {type(fallback_exc).__name__}: {fallback_exc}"
                        ),
                    },
                    effective_strategy,
                )
        else:
            return (
                {
                    "variant": variant.variant,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                effective_strategy,
            )

    if (oof < 0).any():
        return (
            {
                "variant": variant.variant,
                "status": "fail",
                "error": "OOF prediction assignment incomplete",
            },
            effective_strategy,
        )

    return (
        {
            "variant": variant.variant,
            "status": "ok",
            "fold_scores": fold_scores,
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "oof_accuracy": float((oof == y.to_numpy()).mean()),
            "pred_1_rate": float((oof == 1).mean()),
            "oof": oof,
            "error": "",
        },
        effective_strategy,
    )


def _evaluate_variants(
    train: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], dict[str, str], bool]:
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]
    results: dict[str, dict[str, object]] = {}
    histgb_strategies: dict[str, str] = {}
    all_passed = True
    preferred_histgb_strategy = "from_dtype"

    for variant in VARIANTS:
        result, effective_strategy = _evaluate_variant(
            variant,
            train,
            splits,
            y,
            preferred_histgb_strategy,
        )
        results[variant.variant] = result
        if variant.model_name == "HistGradientBoostingClassifier":
            preferred_histgb_strategy = effective_strategy
            histgb_strategies[variant.variant] = effective_strategy
        if result["status"] != "ok":
            all_passed = False

    return results, histgb_strategies, all_passed


def _status_for_comparison(
    candidate_variant: str,
    base_variant: str,
    cv_delta: float,
    oof_delta: float,
    net_correct_delta: int,
    changed_predictions: int,
) -> str:
    if candidate_variant == "gb_raw_tabular_median" and base_variant == candidate_variant:
        return "BASELINE_REFERENCE"
    if candidate_variant == "histgb_raw_tabular_nan_native" and base_variant == candidate_variant:
        return "DIAGNOSTIC_BASELINE_REFERENCE"
    if base_variant == "gb_raw_tabular_median" and candidate_variant.startswith("histgb_"):
        return "CROSS_REFERENCE_ONLY"
    if (
        candidate_variant == "histgb_raw_plus_agemissing_nan_native"
        and changed_predictions == 0
        and cv_delta == 0
        and oof_delta == 0
        and net_correct_delta == 0
    ):
        return "DIAGNOSTIC_ONLY"
    if cv_delta > 0 and oof_delta > 0 and net_correct_delta > 0:
        return "KEEP_CANDIDATE"
    if cv_delta < 0 and oof_delta < 0 and net_correct_delta < 0:
        return "REJECTED"
    if abs(cv_delta) <= CLOSE_TOLERANCE:
        return "DEFERRED"
    return "DEFERRED"


def _comparison_row(
    variant: VariantSpec,
    base_variant_name: str,
    results: dict[str, dict[str, object]],
    y: np.ndarray,
) -> dict[str, object]:
    candidate = results[variant.variant]
    base = results[base_variant_name]
    fold_scores = candidate["fold_scores"]
    candidate_oof = candidate["oof"]
    base_oof = base["oof"]
    changed = candidate_oof != base_oof
    base_correct = base_oof == y
    candidate_correct = candidate_oof == y
    rescue_count = int((~base_correct & candidate_correct).sum())
    kill_count = int((base_correct & ~candidate_correct).sum())
    cv_delta = float(candidate["cv_mean"]) - float(base["cv_mean"])
    oof_delta = float(candidate["oof_accuracy"]) - float(base["oof_accuracy"])
    net_correct_delta = rescue_count - kill_count
    status = _status_for_comparison(
        variant.variant,
        base_variant_name,
        cv_delta,
        oof_delta,
        net_correct_delta,
        int(changed.sum()),
    )

    return {
        "model_name": variant.model_name,
        "variant": variant.variant,
        "feature_set": variant.feature_set,
        "preprocessing_strategy": variant.preprocessing_strategy,
        "cv_mean": _round_float(candidate["cv_mean"]),
        "cv_std": _round_float(candidate["cv_std"]),
        "oof_accuracy": _round_float(candidate["oof_accuracy"]),
        "pred_1_rate": _round_float(candidate["pred_1_rate"]),
        "base_variant": base_variant_name,
        "changed_predictions_vs_base": int(changed.sum()),
        "changed_pct_vs_base": _round_float(float(changed.mean() * 100)),
        "rescue_count": rescue_count,
        "kill_count": kill_count,
        "net_correct_delta": net_correct_delta,
        "status": status,
    }


def _comparison_rows(
    results: dict[str, dict[str, object]],
    train: pd.DataFrame,
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    rows = [_comparison_row(variant, variant.base_variant, results, y) for variant in VARIANTS]

    hist_variants = [
        row
        for row in rows
        if row["model_name"] == "HistGradientBoostingClassifier"
        and row["base_variant"] != "gb_raw_tabular_median"
    ]
    best_hist_row = max(hist_variants, key=lambda row: float(row["cv_mean"]))
    best_hist_variant = _variant_by_name(str(best_hist_row["variant"]))
    rows.append(_comparison_row(best_hist_variant, "gb_raw_tabular_median", results, y))
    return rows


def _failed_rows(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        result = results[variant.variant]
        if result["status"] == "ok":
            continue
        rows.append(
            {
                "model_name": variant.model_name,
                "variant": variant.variant,
                "feature_set": variant.feature_set,
                "preprocessing_strategy": variant.preprocessing_strategy,
                "cv_mean": "",
                "cv_std": "",
                "oof_accuracy": "",
                "pred_1_rate": "",
                "base_variant": variant.base_variant,
                "changed_predictions_vs_base": "",
                "changed_pct_vs_base": "",
                "rescue_count": "",
                "kill_count": "",
                "net_correct_delta": "",
                "status": "FAIL: " + str(result["error"]),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)


def _variant_rows() -> list[dict[str, object]]:
    return [
        {
            "variant": variant.variant,
            "model_name": variant.model_name,
            "feature_set": variant.feature_set,
            "features": ", ".join(variant.features),
            "preprocessing_strategy": variant.preprocessing_strategy,
            "purpose": variant.purpose,
        }
        for variant in VARIANTS
    ]


def _model_panel_rows(histgb_strategies: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        estimator, used_params, build_error = _build_estimator(
            variant,
            histgb_strategies.get(variant.variant, "from_dtype"),
        )
        spec = MODEL_SPECS_BY_NAME[variant.model_name]
        package_version = baseline04._package_version(spec.version_package)
        if estimator is None:
            actual_params: object = "model_unavailable"
            adjustment = ""
            error = build_error
        else:
            model = estimator.named_steps["model"]
            actual_params = model.get_params(deep=False) if hasattr(model, "get_params") else "n/a"
            if (
                variant.model_name == "HistGradientBoostingClassifier"
                and histgb_strategies.get(variant.variant) == "ordinal_fallback"
            ):
                adjustment = (
                    build_error
                    or "fallback categorical handling: ordinal codes; missing/unknown encoded as -1"
                )
            else:
                adjustment = build_error
            error = ""

        rows.append(
            {
                "variant": variant.variant,
                "model_class": variant.model_name,
                "package": spec.package,
                "package_version": package_version,
                "preprocessing_strategy": variant.preprocessing_strategy,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": adjustment,
                "error": error,
            }
        )
    return rows


def _best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique_variant_rows = [
        row
        for row in rows
        if row["status"] not in {"CROSS_REFERENCE_ONLY"}
        and row["base_variant"] != "gb_raw_tabular_median"
        or row["variant"].startswith("gb_")
    ]
    by_variant: dict[str, dict[str, object]] = {}
    for row in unique_variant_rows:
        by_variant.setdefault(str(row["variant"]), row)
    return sorted(by_variant.values(), key=lambda row: float(row["cv_mean"]), reverse=True)


def _gb_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if str(row["variant"]).startswith("gb_")]


def _histgb_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["variant"]).startswith("histgb_")
        and row["status"] != "CROSS_REFERENCE_ONLY"
    ]


def _cross_reference_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["status"] == "CROSS_REFERENCE_ONLY"]


def _decision_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["variant"]
        in {
            "gb_raw_plus_agemissing_median",
            "gb_raw_age_sentinel_plus_agemissing",
            "histgb_raw_plus_agemissing_nan_native",
        }
        and row["status"] != "CROSS_REFERENCE_ONLY"
    ]


def _next_step_recommendation(decision_rows: list[dict[str, object]]) -> str:
    gb_candidate_rows = [
        row
        for row in decision_rows
        if str(row["variant"]).startswith("gb_") and row["status"] == "KEEP_CANDIDATE"
    ]
    if gb_candidate_rows:
        best = max(gb_candidate_rows, key=lambda row: float(row["cv_mean"]))
        return (
            f"Review `{best['variant']}` as the only possible GB missing-handling candidate. "
            "Do not create a submission or checkpoint without an explicit frozen-checkpoint decision."
        )
    return (
        "Do not checkpoint Age missing handling from this report alone. "
        "The primary GB missing-handling variants did not produce a clear keep signal."
    )


def _build_report(
    train: pd.DataFrame,
    panel_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    decision_rows = _decision_rows(comparison_rows)
    recommendation = _next_step_recommendation(decision_rows) if all_passed else "Fix failed rows first."

    lines = [
        "# 13 AgeMissing Handling Check",
        "",
        "## Scope",
        "",
        "- controlled train-side CV/OOF check only",
        "- only `train.csv` is read",
        "- no submission was created",
        "- no public score or Kaggle leaderboard use",
        "- no `gender_submission.csv` as truth",
        "- no test target or row-level test correctness",
        "",
        "## Method boundary",
        "",
        "- This is not feature acceptance.",
        "- No post-score tuning.",
        "- Frozen checkpoint reports are not changed.",
        "- AgeBucket v1 is not being reopened here.",
        "- Broad `Title` remains closed.",
        "- No AgeBucket, Master, Old, Mrs/Miss, Surname, or PassengerId corrections are added.",
        "- CatBoost / LightGBM / XGBoost are not reopened in this protocol.",
        "- The main question is whether median-imputed `Age` is hurting GB.",
        "",
        "## Why this follows AgeBucket v1",
        "",
        "- AgeBucket v1 was checked honestly and improved the SVC lane on public transfer, but it did not beat the primary GB public baseline.",
        "- Step 11 showed that removing raw `Age` made the model lanes noticeably worse.",
        "- This check narrows the hypothesis: keep real known `Age`, then test whether explicit missingness or sentinel missing handling improves the GB lane.",
        "",
        "## Variants and preprocessing strategy",
        "",
        _markdown_table(_variant_rows(), VARIANT_COLUMNS),
        "",
        "## Model panel",
        "",
        _markdown_table(panel_rows, MODEL_PANEL_COLUMNS),
        "",
        "## CV/OOF summary table",
        "",
        f"- overall status: `{status}`",
        f"- rows: `{len(train)}`",
        f"- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state={RANDOM_STATE})`",
        "- metric: `accuracy`",
        "",
        _markdown_table(comparison_rows, CSV_COLUMNS),
        "",
        "## Best rows",
        "",
        _markdown_table(_best_rows(comparison_rows), CSV_COLUMNS),
        "",
        "## Diagnostics",
        "",
        "### GradientBoostingClassifier primary lane",
        "",
        _markdown_table(_gb_rows(comparison_rows), CSV_COLUMNS),
        "",
        "### HistGradientBoostingClassifier native-missing diagnostic lane",
        "",
        _markdown_table(_histgb_rows(comparison_rows), CSV_COLUMNS),
        "",
        "### HistGB best vs GB median baseline cross-reference",
        "",
        _markdown_table(_cross_reference_rows(comparison_rows), CSV_COLUMNS),
        "",
        "The cross-reference row is diagnostic only and is not automatic acceptance.",
        "",
        "## Decision",
        "",
        _markdown_table(decision_rows, CSV_COLUMNS),
        "",
        "Decision status definitions:",
        "",
        "- `BASELINE_REFERENCE`: current comparison anchor.",
        "- `KEEP_CANDIDATE`: beats its base by CV/OOF and has positive net OOF correctness delta.",
        "- `DEFERRED`: close or mixed diagnostics, not enough for acceptance.",
        "- `REJECTED`: CV and OOF diagnostics are worse than the base.",
        "- `DIAGNOSTIC_ONLY`: native-missing diagnostic evidence, not a primary GB decision.",
        "",
        "## Next step recommendation",
        "",
        f"- {recommendation}",
        "- No submission should be created from this step.",
        "- If a later frozen checkpoint is considered, it needs explicit review and a fixed candidate list before any public score.",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    train = _add_agemissing(pd.read_csv(TRAIN_PATH))
    results, histgb_strategies, all_passed = _evaluate_variants(train)
    if all_passed:
        comparison_rows = _comparison_rows(results, train)
    else:
        comparison_rows = _failed_rows(results)

    panel_rows = _model_panel_rows(histgb_strategies)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_rows)
    REPORT_PATH.write_text(
        _build_report(train, panel_rows, comparison_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if all_passed:
        for row in _best_rows(comparison_rows)[:3]:
            print(
                f"{row['variant']}: cv_mean={row['cv_mean']} "
                f"status={row['status']} net={row['net_correct_delta']}"
            )
    else:
        for row in comparison_rows:
            print(f"{row['variant']}: {row['status']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
