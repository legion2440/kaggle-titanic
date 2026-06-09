from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, REPORTS_DIR, TARGET, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import CATEGORICAL_FEATURES, make_preprocessor

# AgeBucket is constructed locally for this check only — register it as categorical
# so make_preprocessor can route it to OHE (same pattern as scripts 17/18).
if "AgeBucket" not in CATEGORICAL_FEATURES:
    CATEGORICAL_FEATURES.append("AgeBucket")


REPORT_PATH = REPORTS_DIR / "11_agebucket_feature_check.md"
CSV_PATH = REPORTS_DIR / "11_agebucket_feature_check.csv"

ACTIVE_MODELS = [
    "GradientBoostingClassifier",
    "SVC",
    "CatBoostClassifier",
]

RAW_NO_AGE = ["Sex", "Pclass", "Embarked", "SibSp", "Parch", "Fare"]
RAW_NO_AGE_NO_SEX_PLUS_AGEBUCKET_V1 = [
    "Pclass",
    "Embarked",
    "SibSp",
    "Parch",
    "Fare",
    "AgeBucket",
]

FEATURE_SETS = {
    "raw_tabular": list(RAW_TABULAR),
    "raw_no_age": RAW_NO_AGE,
    "raw_no_age_no_sex_plus_agebucket_v1": RAW_NO_AGE_NO_SEX_PLUS_AGEBUCKET_V1,
}

CSV_COLUMNS = [
    "model_name",
    "feature_set",
    "cv_mean",
    "cv_std",
    "oof_accuracy",
    "pred_1_rate",
    "base_feature_set",
    "changed_predictions_vs_base",
    "changed_pct_vs_base",
    "rescue_count",
    "kill_count",
    "net_correct_delta",
    "status",
]

MODEL_COLUMNS = [
    "model",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

ALMOST_NO_WORSE_TOLERANCE = 0.0025


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
MODEL_SPECS = [spec for spec in baseline04.MODEL_SPECS if spec.model in ACTIVE_MODELS]


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _build_model(spec: object) -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(spec)


def _add_agebucket_v1(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age_missing = out["Age"].isna()
    female = out["Sex"].eq("female")
    male = out["Sex"].eq("male")

    bucket = pd.Series("AdultMale", index=out.index, dtype="object")
    bucket.loc[female] = "AdultFemale"
    bucket.loc[age_missing & female] = "AgeMissingFemale"
    bucket.loc[age_missing & male] = "AgeMissingMale"
    bucket.loc[out["Age"].notna() & female & out["Age"].lt(14)] = "ChildFemale"
    bucket.loc[out["Age"].notna() & male & out["Age"].lt(14)] = "ChildMale"
    out["AgeBucket"] = bucket
    return out


def _model_panel_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, Any]]]:
    rows = []
    resolved: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        model, used_params, error_or_adjustment = _build_model(spec)
        package_version = baseline04._package_version(spec.version_package)
        if model is not None and hasattr(model, "get_params"):
            actual_params = model.get_params(deep=False)
        elif model is not None:
            actual_params = "get_params_unavailable"
        else:
            actual_params = "model_unavailable"

        is_error = model is None
        rows.append(
            {
                "model": spec.model,
                "package": spec.package,
                "package_version": package_version,
                "preprocessing_mode": spec.preprocessing_mode,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": "" if is_error else error_or_adjustment,
                "error": error_or_adjustment if is_error else "",
            }
        )
        resolved[spec.model] = {
            "model_available": model is not None,
            "model_error": error_or_adjustment if is_error else "",
        }

    return rows, resolved


def _evaluate_feature_sets(
    train: pd.DataFrame,
    resolved_models: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, object]], bool]:
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]
    results: dict[tuple[str, str], dict[str, object]] = {}
    all_passed = True

    for spec in MODEL_SPECS:
        resolved = resolved_models[spec.model]
        for feature_set_name, feature_names in FEATURE_SETS.items():
            missing_features = [feature for feature in feature_names if feature not in train.columns]
            if missing_features:
                all_passed = False
                results[(spec.model, feature_set_name)] = {
                    "status": "fail",
                    "error": "missing feature columns: " + ", ".join(missing_features),
                }
                continue
            if not resolved["model_available"]:
                all_passed = False
                results[(spec.model, feature_set_name)] = {
                    "status": "fail",
                    "error": resolved["model_error"],
                }
                continue

            try:
                fold_scores = []
                oof = np.full(len(train), -1, dtype=int)
                x = train[feature_names]

                for i, (train_idx, valid_idx) in enumerate(splits):
                    model, _, build_error = _build_model(spec)
                    if model is None:
                        raise RuntimeError(build_error)
                    estimator = Pipeline(
                        steps=[
                            ("preprocess", make_preprocessor(spec.preprocessing_mode, list(feature_names))),
                            ("model", model),
                        ]
                    )
                    estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
                    fold_pred = estimator.predict(x.iloc[valid_idx]).astype(int)
                    if i < 5:
                        oof[valid_idx] = fold_pred
                    fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

                if (oof < 0).any():
                    raise RuntimeError("OOF prediction assignment incomplete")

                results[(spec.model, feature_set_name)] = {
                    "status": "ok",
                    "fold_scores": fold_scores,
                    "cv_mean": float(np.mean(fold_scores)),
                    "cv_std": float(np.std(fold_scores)),
                    "oof_accuracy": float((oof == y.to_numpy()).mean()),
                    "pred_1_rate": float((oof == 1).mean()),
                    "oof": oof,
                    "error": "",
                }
            except Exception as exc:
                all_passed = False
                results[(spec.model, feature_set_name)] = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return results, all_passed


def _raw_no_age_status(raw_no_age_mean: float, raw_tabular_mean: float) -> str:
    if raw_no_age_mean >= raw_tabular_mean - ALMOST_NO_WORSE_TOLERANCE:
        return "RAW_AGE_NO_STABLE_BENEFIT"
    return "RAW_AGE_REMOVAL_HURTS"


def _agebucket_status(agebucket_mean: float, raw_tabular_mean: float, raw_no_age_mean: float) -> str:
    if agebucket_mean > raw_tabular_mean and agebucket_mean > raw_no_age_mean:
        return "KEEP_CANDIDATE"
    if agebucket_mean > raw_no_age_mean and agebucket_mean < raw_tabular_mean:
        return "DEFERRED"
    if agebucket_mean < raw_tabular_mean and agebucket_mean < raw_no_age_mean:
        return "REJECTED"
    return "DIAGNOSTIC_TIE"


def _comparison_row(
    model_name: str,
    feature_set: str,
    base_feature_set: str,
    status: str,
    results: dict[tuple[str, str], dict[str, object]],
    y: np.ndarray,
) -> dict[str, object]:
    candidate = results[(model_name, feature_set)]
    base = results[(model_name, base_feature_set)]
    fold_scores = candidate["fold_scores"]
    candidate_oof = candidate["oof"]
    base_oof = base["oof"]
    changed = candidate_oof != base_oof
    base_correct = base_oof == y
    candidate_correct = candidate_oof == y
    rescue_count = int((~base_correct & candidate_correct).sum())
    kill_count = int((base_correct & ~candidate_correct).sum())

    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "cv_mean": _round_float(candidate["cv_mean"]),
        "cv_std": _round_float(candidate["cv_std"]),
        "oof_accuracy": _round_float(candidate["oof_accuracy"]),
        "pred_1_rate": _round_float(candidate["pred_1_rate"]),
        "base_feature_set": base_feature_set,
        "changed_predictions_vs_base": int(changed.sum()),
        "changed_pct_vs_base": _round_float(float(changed.mean() * 100)),
        "rescue_count": rescue_count,
        "kill_count": kill_count,
        "net_correct_delta": rescue_count - kill_count,
        "status": status,
    }


def _comparison_rows(
    results: dict[tuple[str, str], dict[str, object]],
    train: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    y = train[TARGET].astype(int).to_numpy()

    for spec in MODEL_SPECS:
        model_name = spec.model
        raw_mean = float(results[(model_name, "raw_tabular")]["cv_mean"])
        no_age_mean = float(results[(model_name, "raw_no_age")]["cv_mean"])
        agebucket_mean = float(results[(model_name, "raw_no_age_no_sex_plus_agebucket_v1")]["cv_mean"])
        raw_no_age_status = _raw_no_age_status(no_age_mean, raw_mean)
        agebucket_status = _agebucket_status(agebucket_mean, raw_mean, no_age_mean)

        rows.append(
            _comparison_row(
                model_name,
                "raw_tabular",
                "raw_tabular",
                "BASELINE_REFERENCE",
                results,
                y,
            )
        )
        rows.append(
            _comparison_row(
                model_name,
                "raw_no_age",
                "raw_tabular",
                raw_no_age_status,
                results,
                y,
            )
        )
        rows.append(
            _comparison_row(
                model_name,
                "raw_no_age_no_sex_plus_agebucket_v1",
                "raw_tabular",
                agebucket_status,
                results,
                y,
            )
        )
        rows.append(
            _comparison_row(
                model_name,
                "raw_no_age_no_sex_plus_agebucket_v1",
                "raw_no_age",
                agebucket_status,
                results,
                y,
            )
        )

    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)


def _best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_pair = {}
    for row in rows:
        key = (row["model_name"], row["feature_set"])
        if key not in by_pair:
            by_pair[key] = row
    return sorted(by_pair.values(), key=lambda row: float(row["cv_mean"]), reverse=True)[:6]


def _diagnostic_rows_for_model(rows: list[dict[str, object]], model_name: str) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["model_name"] == model_name
        and not (row["feature_set"] == "raw_tabular" and row["base_feature_set"] == "raw_tabular")
    ]


def _decision_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["feature_set"] == "raw_no_age_no_sex_plus_agebucket_v1"
        and row["base_feature_set"] == "raw_tabular"
    ]


def _build_report(
    train: pd.DataFrame,
    panel_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    decision_rows = _decision_rows(comparison_rows)
    primary_decision = next(row for row in decision_rows if row["model_name"] == "GradientBoostingClassifier")
    keep_count = sum(1 for row in decision_rows if row["status"] == "KEEP_CANDIDATE")
    deferred_count = sum(1 for row in decision_rows if row["status"] == "DEFERRED")
    rejected_count = sum(1 for row in decision_rows if row["status"] == "REJECTED")

    if primary_decision["status"] == "KEEP_CANDIDATE":
        recommendation = (
            "AgeBucket v1 is a clear primary-lane candidate for review. "
            "A frozen checkpoint is possible only after explicit review."
        )
    elif primary_decision["status"] == "DEFERRED":
        recommendation = (
            "AgeBucket v1 improves the raw-no-age branch but does not beat raw_tabular in the primary lane. "
            "Do not checkpoint without further review."
        )
    else:
        recommendation = (
            "AgeBucket v1 is not a clear primary-lane candidate from this check. "
            "No frozen checkpoint is recommended."
        )

    lines = [
        "# 11 AgeBucket Feature Check",
        "",
        "## Scope",
        "",
        "- train-side CV/OOF check only",
        "- only `train.csv` is read",
        "- no submission generation",
        "- no Kaggle/public leaderboard use",
        "- no `gender_submission.csv` as truth",
        "- no test target or row-level test correctness",
        "- no target-derived family/group survival",
        "",
        "## Method boundary",
        "",
        "- This is not feature acceptance.",
        "- Broad Title remains closed.",
        "- Master fallback was intentionally skipped.",
        "- Old buckets were intentionally skipped.",
        "- `AgeBucket` v1 is built locally for this controlled check only.",
        "- `scripts/preprocessing.py` only recognizes `AgeBucket` as a categorical column for this check.",
        "",
        "## Feature sets",
        "",
        _markdown_table(
            [{"feature_set": name, "features": ", ".join(features)} for name, features in FEATURE_SETS.items()],
            ["feature_set", "features"],
        ),
        "",
        "## AgeBucket v1 mapping",
        "",
        "```python",
        "if Age is missing and Sex == \"female\":",
        "    AgeBucket = \"AgeMissingFemale\"",
        "elif Age is missing and Sex == \"male\":",
        "    AgeBucket = \"AgeMissingMale\"",
        "elif Sex == \"female\" and Age < 14:",
        "    AgeBucket = \"ChildFemale\"",
        "elif Sex == \"male\" and Age < 14:",
        "    AgeBucket = \"ChildMale\"",
        "elif Sex == \"female\":",
        "    AgeBucket = \"AdultFemale\"",
        "else:",
        "    AgeBucket = \"AdultMale\"",
        "```",
        "",
        "Skipped by design: `OldFemale`, `OldMale`, `Master`, `Mrs`, `Miss`, `Surname`, broad `Title`, and PassengerId corrections.",
        "",
        "## Model panel",
        "",
        _markdown_table(panel_rows, MODEL_COLUMNS),
        "",
        "## CV/OOF summary table",
        "",
        f"- overall status: `{status}`",
        f"- rows: `{len(train)}`",
        f"- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state={RANDOM_STATE})`",
        "",
        _markdown_table(comparison_rows, CSV_COLUMNS),
        "",
        "## Best rows by CV mean",
        "",
        _markdown_table(_best_rows(comparison_rows), CSV_COLUMNS),
        "",
        "## Diagnostics by model",
    ]

    for spec in MODEL_SPECS:
        lines.extend(
            [
                "",
                f"### {spec.model}",
                "",
                _markdown_table(_diagnostic_rows_for_model(comparison_rows, spec.model), CSV_COLUMNS),
            ]
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            _markdown_table(decision_rows, CSV_COLUMNS),
            "",
            f"- `KEEP_CANDIDATE` lanes: `{keep_count}`",
            f"- `DEFERRED` lanes: `{deferred_count}`",
            f"- `REJECTED` lanes: `{rejected_count}`",
            f"- Primary lane status: `{primary_decision['status']}`",
            "",
            "## Next step recommendation",
            "",
            f"- {recommendation}",
            "- Do not mark AgeBucket v1 as accepted from this report alone.",
            "- Next possible step after review: frozen checkpoint only if AgeBucket v1 becomes a clear candidate.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    train = _add_agebucket_v1(train_raw)
    panel_rows, resolved_models = _model_panel_rows()
    results, all_passed = _evaluate_feature_sets(train, resolved_models)

    if all_passed:
        comparison_rows = _comparison_rows(results, train)
    else:
        comparison_rows = []

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_rows)
    REPORT_PATH.write_text(
        _build_report(train, panel_rows, comparison_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
