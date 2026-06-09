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
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, REPORTS_DIR, TARGET, TRAIN_PATH
from scripts.features import F00_CORE, RAW_TABULAR, add_clean_features
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "06_title_feature_check.md"
CSV_PATH = REPORTS_DIR / "06_title_feature_check.csv"

ACTIVE_MODELS = [
    "GradientBoostingClassifier",
    "SVC",
    "CatBoostClassifier",
]

EXCLUDED_MODEL_REASONS = {
    "RandomForestClassifier": "rejected by `05_baseline_frozen_checkpoint` public transfer",
    "LGBMClassifier": "rejected by `05_baseline_frozen_checkpoint` public transfer",
    "HistGradientBoostingClassifier": "rejected by `05_baseline_frozen_checkpoint` public transfer",
    "XGBClassifier": "not active after baseline checkpoint; no raw_tabular checkpoint lane",
    "ExtraTreesClassifier": "excluded before baseline checkpoint or not active baseline lanes",
    "DecisionTreeClassifier": "excluded before baseline checkpoint or not active baseline lanes",
    "AdaBoostClassifier": "excluded before baseline checkpoint or not active baseline lanes",
    "LinearSVC": "excluded before baseline checkpoint or not active baseline lanes",
    "KNeighborsClassifier": "excluded before baseline checkpoint or not active baseline lanes",
    "GaussianNB": "excluded before baseline checkpoint or not active baseline lanes",
    "DummyClassifier": "excluded before baseline checkpoint or not active baseline lanes",
}

FEATURE_SETS = {
    "f00_core": list(F00_CORE),
    "f00_core_plus_title": [*F00_CORE, "Title"],
    "raw_tabular": list(RAW_TABULAR),
    "raw_plus_title": [*RAW_TABULAR, "Title"],
}

COMPARISONS = [
    ("f00_core", "f00_core_plus_title"),
    ("raw_tabular", "raw_plus_title"),
]

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

PANEL_COLUMNS = [
    "model",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

PAIRED_COLUMNS = [
    "model",
    "base_feature_set",
    "candidate_feature_set",
    "base_cv_mean",
    "candidate_cv_mean",
    "delta_candidate_minus_base",
    "result",
]

OOF_COLUMNS = [
    "model",
    "base_feature_set",
    "candidate_feature_set",
    "changed_predictions",
    "changed_pct",
    "base_pred_1_count",
    "candidate_pred_1_count",
    "base_pred_1_rate",
    "candidate_pred_1_rate",
    "delta_pred_1_rate",
    "rescue_count",
    "kill_count",
    "net_correct_delta",
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
MODEL_SPECS = [spec for spec in baseline04.MODEL_SPECS if spec.model in ACTIVE_MODELS]


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _build_model(spec: object) -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(spec)


def _prepare_title_frame(train_raw: pd.DataFrame) -> pd.DataFrame:
    clean = add_clean_features(train_raw)
    train = train_raw.copy()
    train["Title"] = clean["Title"]
    return train


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
        adjustment = "" if is_error else error_or_adjustment
        error = error_or_adjustment if is_error else ""
        rows.append(
            {
                "model": spec.model,
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
            "used_params": used_params,
            "model_available": model is not None,
            "model_error": error,
        }

    return rows, resolved


def _evaluate(
    train: pd.DataFrame,
    resolved_models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray], bool]:
    missing_target = TARGET not in train.columns
    y = train[TARGET] if not missing_target else pd.Series(dtype=int)
    splits = []
    if not missing_target:
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
        oof_splits = splits[:5]

    rows = []
    oof_predictions: dict[tuple[str, str], np.ndarray] = {}
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
                fold_scores = []
                oof = np.full(len(train), -1, dtype=int)
                x = train[list(feature_names)]

                for i, (train_idx, valid_idx) in enumerate(splits):
                    fold_estimator = clone(estimator)
                    fold_estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
                    fold_pred = fold_estimator.predict(x.iloc[valid_idx]).astype(int)
                    if i < 5:
                        oof[valid_idx] = fold_pred
                    fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

                if (oof < 0).any():
                    raise RuntimeError("OOF prediction assignment incomplete")

                oof_predictions[(spec.model, feature_set_name)] = oof
                rows.append(
                    {
                        "status": "ok",
                        **base_row,
                        "cv_mean": _round_float(np.mean(fold_scores)),
                        "cv_std": _round_float(np.std(fold_scores)),
                        "cv_min": _round_float(np.min(fold_scores)),
                        "cv_max": _round_float(np.max(fold_scores)),
                        "fold_scores": _json_dumps([_round_float(score) for score in fold_scores]),
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

    return rows, oof_predictions, all_passed


def _paired_rows(
    rows: list[dict[str, object]],
    base_feature_set: str,
    candidate_feature_set: str,
) -> list[dict[str, object]]:
    rows_by_model_feature_set = {
        (row["model"], row["feature_set"]): row
        for row in rows
        if row["status"] == "ok" and row["cv_mean"] != ""
    }
    paired = []

    for spec in MODEL_SPECS:
        base = rows_by_model_feature_set.get((spec.model, base_feature_set))
        candidate = rows_by_model_feature_set.get((spec.model, candidate_feature_set))
        if base is None or candidate is None:
            paired.append(
                {
                    "model": spec.model,
                    "base_feature_set": base_feature_set,
                    "candidate_feature_set": candidate_feature_set,
                    "base_cv_mean": "" if base is None else base["cv_mean"],
                    "candidate_cv_mean": "" if candidate is None else candidate["cv_mean"],
                    "delta_candidate_minus_base": "",
                    "result": "n/a",
                }
            )
            continue

        delta = _round_float(float(candidate["cv_mean"]) - float(base["cv_mean"]))
        if delta > 0:
            result = "improved"
        elif delta < 0:
            result = "worsened"
        else:
            result = "tied"
        paired.append(
            {
                "model": spec.model,
                "base_feature_set": base_feature_set,
                "candidate_feature_set": candidate_feature_set,
                "base_cv_mean": base["cv_mean"],
                "candidate_cv_mean": candidate["cv_mean"],
                "delta_candidate_minus_base": delta,
                "result": result,
            }
        )

    return paired


def _oof_rows(
    train: pd.DataFrame,
    oof_predictions: dict[tuple[str, str], np.ndarray],
    base_feature_set: str,
    candidate_feature_set: str,
) -> list[dict[str, object]]:
    rows = []
    y = train[TARGET].to_numpy()

    for spec in MODEL_SPECS:
        base_pred = oof_predictions.get((spec.model, base_feature_set))
        candidate_pred = oof_predictions.get((spec.model, candidate_feature_set))
        if base_pred is None or candidate_pred is None:
            rows.append(
                {
                    "model": spec.model,
                    "base_feature_set": base_feature_set,
                    "candidate_feature_set": candidate_feature_set,
                    "changed_predictions": "n/a",
                    "changed_pct": "n/a",
                    "base_pred_1_count": "n/a",
                    "candidate_pred_1_count": "n/a",
                    "base_pred_1_rate": "n/a",
                    "candidate_pred_1_rate": "n/a",
                    "delta_pred_1_rate": "n/a",
                    "rescue_count": "n/a",
                    "kill_count": "n/a",
                    "net_correct_delta": "n/a",
                }
            )
            continue

        changed = base_pred != candidate_pred
        base_correct = base_pred == y
        candidate_correct = candidate_pred == y
        rescue_count = int((~base_correct & candidate_correct).sum())
        kill_count = int((base_correct & ~candidate_correct).sum())
        base_pred_1_rate = float((base_pred == 1).mean())
        candidate_pred_1_rate = float((candidate_pred == 1).mean())

        rows.append(
            {
                "model": spec.model,
                "base_feature_set": base_feature_set,
                "candidate_feature_set": candidate_feature_set,
                "changed_predictions": int(changed.sum()),
                "changed_pct": _round_float(float(changed.mean() * 100)),
                "base_pred_1_count": int((base_pred == 1).sum()),
                "candidate_pred_1_count": int((candidate_pred == 1).sum()),
                "base_pred_1_rate": _round_float(base_pred_1_rate),
                "candidate_pred_1_rate": _round_float(candidate_pred_1_rate),
                "delta_pred_1_rate": _round_float(candidate_pred_1_rate - base_pred_1_rate),
                "rescue_count": rescue_count,
                "kill_count": kill_count,
                "net_correct_delta": rescue_count - kill_count,
            }
        )

    return rows


def _paired_summary(rows: list[dict[str, object]]) -> list[str]:
    improved = [row for row in rows if row["result"] == "improved"]
    worsened = [row for row in rows if row["result"] == "worsened"]
    tied = [row for row in rows if row["result"] == "tied"]
    positive = [
        row for row in rows
        if row["delta_candidate_minus_base"] != ""
        and float(row["delta_candidate_minus_base"]) > 0
    ]
    negative = [
        row for row in rows
        if row["delta_candidate_minus_base"] != ""
        and float(row["delta_candidate_minus_base"]) < 0
    ]
    best_positive = max(positive, key=lambda row: float(row["delta_candidate_minus_base"]), default=None)
    worst_negative = min(negative, key=lambda row: float(row["delta_candidate_minus_base"]), default=None)

    return [
        f"- improved: `{len(improved)}`",
        f"- worsened: `{len(worsened)}`",
        f"- tied: `{len(tied)}`",
        "- best positive delta: "
        + (
            "`n/a`"
            if best_positive is None
            else f"`{best_positive['model']} ({best_positive['delta_candidate_minus_base']})`"
        ),
        "- worst negative delta: "
        + (
            "`n/a`"
            if worst_negative is None
            else f"`{worst_negative['model']} ({worst_negative['delta_candidate_minus_base']})`"
        ),
    ]


def _oof_summary(rows: list[dict[str, object]]) -> list[str]:
    numeric_rows = [row for row in rows if row["changed_pct"] != "n/a"]
    if not numeric_rows:
        return [
            "- OOF changed prediction range: `n/a`",
            "- best net_correct_delta: `n/a`",
        ]

    min_changed = min(float(row["changed_pct"]) for row in numeric_rows)
    max_changed = max(float(row["changed_pct"]) for row in numeric_rows)
    best_net = max(numeric_rows, key=lambda row: int(row["net_correct_delta"]))
    return [
        f"- OOF changed prediction range: `{_round_float(min_changed)}%` to `{_round_float(max_changed)}%`",
        f"- best net_correct_delta: `{best_net['model']} ({best_net['net_correct_delta']})`",
    ]


def _effect_label(oof_rows: list[dict[str, object]]) -> str:
    numeric = [float(row["changed_pct"]) for row in oof_rows if row["changed_pct"] != "n/a"]
    if not numeric:
        return "unknown"
    max_changed = max(numeric)
    if max_changed < 2:
        return "small"
    if max_changed < 10:
        return "moderate"
    return "large"


def _comparison_effect(rows: list[dict[str, object]], label: str) -> str:
    improved = sum(1 for row in rows if row["result"] == "improved")
    worsened = sum(1 for row in rows if row["result"] == "worsened")
    tied = sum(1 for row in rows if row["result"] == "tied")
    if improved > worsened:
        direction = "helps"
    elif worsened > improved:
        direction = "does not help overall"
    else:
        direction = "is mixed"
    return (
        f"- On the {label}, `Title` {direction} in this controlled check "
        f"({improved} improved, {worsened} worsened, {tied} tied)."
    )


def _excluded_rows() -> list[dict[str, object]]:
    return [
        {"model": model, "reason": reason}
        for model, reason in EXCLUDED_MODEL_REASONS.items()
    ]


def _build_report(
    train: pd.DataFrame,
    panel_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    oof_predictions: dict[tuple[str, str], np.ndarray],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    core_paired = _paired_rows(result_rows, "f00_core", "f00_core_plus_title")
    raw_paired = _paired_rows(result_rows, "raw_tabular", "raw_plus_title")
    core_oof = _oof_rows(train, oof_predictions, "f00_core", "f00_core_plus_title")
    raw_oof = _oof_rows(train, oof_predictions, "raw_tabular", "raw_plus_title")
    oof_effect = _effect_label([*core_oof, *raw_oof])

    lines = [
        "# 06 Title Feature Check",
        "",
        "## Scope Boundary",
        "",
        "- train-side CV only",
        "- `train.csv` only",
        "- `Survived` is used only as the target",
        "- `add_clean_features()` is used only to create `Title`",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- same CV protocol and technical model parameters as `04_baseline`",
        "- OOF diagnostics are train-side only",
        "- no submission generation",
        "- no Kaggle/public leaderboard use",
        "- no `test.csv` scoring",
        "- no test labels or row-level correctness checks",
        "- no `gender_submission.csv` as truth",
        "- no hyperparameter tuning",
        "- no threshold tuning",
        "- no final model selection",
        "- no derived features other than `Title` are included in any feature set",
        "- no gating, probability threshold changes, PassengerId overrides, or manual correction rules",
        "",
        "## Active Model Lanes",
        "",
        _markdown_table(
            [{"model": model, "reason": "active/deferred after `05_baseline_frozen_checkpoint`"} for model in ACTIVE_MODELS],
            ["model", "reason"],
        ),
        "",
        "## Excluded Models",
        "",
        _markdown_table(_excluded_rows(), ["model", "reason"]),
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
        f"- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state={RANDOM_STATE})`",
        "- metric: `accuracy`",
        "- identical precomputed CV split indices are reused for every model and feature set",
        "- preprocessing is fitted inside each train fold through an sklearn `Pipeline`",
        f"- rows: `{len(train)}` from `train.csv`",
        "",
        "## Model Panel Used",
        "",
        _markdown_table(panel_rows, PANEL_COLUMNS),
        "",
        "## All Results",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(result_rows, RESULT_COLUMNS),
        "",
        "## Paired comparison: `f00_core_plus_title` vs `f00_core`",
        "",
        _markdown_table(core_paired, PAIRED_COLUMNS),
        "",
        "## Paired comparison: `raw_plus_title` vs `raw_tabular`",
        "",
        _markdown_table(raw_paired, PAIRED_COLUMNS),
        "",
        "## OOF diagnostics: `f00_core_plus_title` vs `f00_core`",
        "",
        _markdown_table(core_oof, OOF_COLUMNS),
        "",
        "## OOF diagnostics: `raw_plus_title` vs `raw_tabular`",
        "",
        _markdown_table(raw_oof, OOF_COLUMNS),
        "",
        "## Summary",
        "",
        "`f00_core_plus_title` vs `f00_core`:",
        "",
        *(_paired_summary(core_paired)),
        *(_oof_summary(core_oof)),
        "",
        "`raw_plus_title` vs `raw_tabular`:",
        "",
        *(_paired_summary(raw_paired)),
        *(_oof_summary(raw_oof)),
        "",
        "## Short interpretation",
        "",
        _comparison_effect(core_paired, "core layer"),
        _comparison_effect(raw_paired, "raw_tabular layer"),
        f"- The OOF effect looks {oof_effect} by changed-prediction share across active lanes.",
        "- This is feature-check and OOF diagnostic evidence only.",
        "- This is not final model selection.",
        "- No gating was applied.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows)


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    train = _prepare_title_frame(train_raw)
    panel_rows, resolved_models = _model_panel_rows()
    result_rows, oof_predictions, all_passed = _evaluate(train, resolved_models)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(result_rows)
    REPORT_PATH.write_text(
        _build_report(train, panel_rows, result_rows, oof_predictions, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
