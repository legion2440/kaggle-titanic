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
from scripts.features import RAW_TABULAR, add_clean_features
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "08_title_gating_check.md"
CSV_PATH = REPORTS_DIR / "08_title_gating_check.csv"

MODEL_NAME = "GradientBoostingClassifier"
BASE_FEATURE_SET = "raw_tabular"
TITLE_FEATURE_SET = "raw_plus_title"
BASE_FEATURES = list(RAW_TABULAR)
TITLE_FEATURES = [*RAW_TABULAR, "Title"]

WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.00]

RESULT_COLUMNS = [
    "weight",
    "accuracy",
    "accuracy_delta_vs_base",
    "changed_predictions_vs_base",
    "changed_pct_vs_base",
    "base_pred_1_count",
    "blend_pred_1_count",
    "base_pred_1_rate",
    "blend_pred_1_rate",
    "delta_pred_1_rate",
    "rescue_count",
    "kill_count",
    "net_correct_delta",
    "flip_0_to_1",
    "flip_1_to_0",
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
MODEL_SPEC = next(spec for spec in baseline04.MODEL_SPECS if spec.model == MODEL_NAME)


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPEC)


def _prepare_title_frame(raw: pd.DataFrame) -> pd.DataFrame:
    clean = add_clean_features(raw)
    frame = raw.copy()
    frame["Title"] = clean["Title"]
    return frame


def _class1_proba(estimator: Pipeline, x_valid: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(x_valid)
    model = estimator.named_steps["model"]
    classes = getattr(model, "classes_", getattr(estimator, "classes_", None))
    if classes is None:
        raise RuntimeError("model classes_ unavailable for probability alignment")
    class_values = list(classes)
    if 1 not in class_values:
        raise RuntimeError(f"class 1 missing from fitted classes: {class_values}")
    return probabilities[:, class_values.index(1)]


def _model_row() -> tuple[dict[str, object], bool]:
    model, used_params, error_or_adjustment = _build_model()
    package_version = baseline04._package_version(MODEL_SPEC.version_package)
    if model is not None and hasattr(model, "get_params"):
        actual_params = model.get_params(deep=False)
    elif model is not None:
        actual_params = "get_params_unavailable"
    else:
        actual_params = "model_unavailable"

    row = {
        "model": MODEL_SPEC.model,
        "package": MODEL_SPEC.package,
        "package_version": package_version,
        "preprocessing_mode": MODEL_SPEC.preprocessing_mode,
        "explicit_technical_params": _json_dumps(used_params),
        "actual_resolved_params": _json_dumps(actual_params),
        "parameter_adjustments": "" if model is not None else "model unavailable",
        "error": "" if model is not None else error_or_adjustment,
    }
    return row, model is not None


def _compute_oof_probabilities(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing_base = [feature for feature in BASE_FEATURES if feature not in train.columns]
    missing_title = [feature for feature in TITLE_FEATURES if feature not in train.columns]
    if TARGET not in train.columns:
        raise ValueError(f"missing target column: {TARGET}")
    if missing_base:
        raise ValueError("missing base feature columns: " + ", ".join(missing_base))
    if missing_title:
        raise ValueError("missing title feature columns: " + ", ".join(missing_title))

    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]
    p_base = np.full(len(train), np.nan, dtype=float)
    p_title = np.full(len(train), np.nan, dtype=float)

    for i, (train_idx, valid_idx) in enumerate(splits):
        base_model, _, base_error = _build_model()
        title_model, _, title_error = _build_model()
        if base_model is None:
            raise RuntimeError(base_error)
        if title_model is None:
            raise RuntimeError(title_error)

        base_pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(MODEL_SPEC.preprocessing_mode, BASE_FEATURES)),
                ("model", base_model),
            ]
        )
        title_pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(MODEL_SPEC.preprocessing_mode, TITLE_FEATURES)),
                ("model", title_model),
            ]
        )

        x_base_train = train.iloc[train_idx][BASE_FEATURES]
        x_base_valid = train.iloc[valid_idx][BASE_FEATURES]
        x_title_train = train.iloc[train_idx][TITLE_FEATURES]
        x_title_valid = train.iloc[valid_idx][TITLE_FEATURES]

        base_pipeline.fit(x_base_train, y.iloc[train_idx])
        title_pipeline.fit(x_title_train, y.iloc[train_idx])
        if i < 5:
            p_base[valid_idx] = _class1_proba(base_pipeline, x_base_valid)
            p_title[valid_idx] = _class1_proba(title_pipeline, x_title_valid)

    if np.isnan(p_base).any() or np.isnan(p_title).any():
        raise RuntimeError("OOF probability assignment incomplete")

    return p_base, p_title


def _status_for_weight(
    weight: float,
    accuracy_delta: float,
    changed_predictions: int,
    net_correct_delta: int,
) -> str:
    if weight == 0.00:
        return "BASELINE_REFERENCE"
    if weight == 1.00 and changed_predictions != 0:
        return "REJECTED_FULL_STRENGTH"
    if accuracy_delta <= 0:
        return "REJECTED_NO_OOF_GAIN"
    if changed_predictions > 14:
        return "HIGH_TRANSFER_RISK"
    if 9 <= changed_predictions <= 14:
        return "CAUTION_ZONE"
    if accuracy_delta > 0 and net_correct_delta > 0 and changed_predictions <= 8:
        return "CONSERVATIVE_CANDIDATE"
    return "DIAGNOSTIC_ONLY"


def _result_rows(train: pd.DataFrame, p_base: np.ndarray, p_title: np.ndarray) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    base_pred = (p_base >= 0.5).astype(int)
    base_accuracy = float((base_pred == y).mean())
    base_pred_1_rate = float((base_pred == 1).mean())
    rows = []

    for weight in WEIGHTS:
        p_blend = (1 - weight) * p_base + weight * p_title
        blend_pred = (p_blend >= 0.5).astype(int)
        changed = base_pred != blend_pred
        base_correct = base_pred == y
        blend_correct = blend_pred == y
        rescue_count = int((~base_correct & blend_correct).sum())
        kill_count = int((base_correct & ~blend_correct).sum())
        accuracy = float(blend_correct.mean())
        accuracy_delta = accuracy - base_accuracy
        blend_pred_1_rate = float((blend_pred == 1).mean())
        net_correct_delta = rescue_count - kill_count
        flip_0_to_1 = int(((base_pred == 0) & (blend_pred == 1)).sum())
        flip_1_to_0 = int(((base_pred == 1) & (blend_pred == 0)).sum())
        changed_count = int(changed.sum())

        rows.append(
            {
                "weight": f"{weight:.2f}",
                "accuracy": _round_float(accuracy),
                "accuracy_delta_vs_base": _round_float(accuracy_delta),
                "changed_predictions_vs_base": changed_count,
                "changed_pct_vs_base": _round_float(float(changed.mean() * 100)),
                "base_pred_1_count": int((base_pred == 1).sum()),
                "blend_pred_1_count": int((blend_pred == 1).sum()),
                "base_pred_1_rate": _round_float(base_pred_1_rate),
                "blend_pred_1_rate": _round_float(blend_pred_1_rate),
                "delta_pred_1_rate": _round_float(blend_pred_1_rate - base_pred_1_rate),
                "rescue_count": rescue_count,
                "kill_count": kill_count,
                "net_correct_delta": net_correct_delta,
                "flip_0_to_1": flip_0_to_1,
                "flip_1_to_0": flip_1_to_0,
                "status": _status_for_weight(weight, accuracy_delta, changed_count, net_correct_delta),
            }
        )

    return rows


def _best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best = max(rows, key=lambda row: float(row["accuracy"]))
    return [best]


def _conservative_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["status"] == "CONSERVATIVE_CANDIDATE"]


def _interpretation(rows: list[dict[str, object]]) -> list[str]:
    conservative = _conservative_rows(rows)
    improved = [row for row in rows if float(row["accuracy_delta_vs_base"]) > 0]
    if conservative:
        best_conservative = max(conservative, key=lambda row: float(row["accuracy"]))
        conservative_changed = int(best_conservative["changed_predictions_vs_base"])
        if conservative_changed <= 8:
            conservative_flip_label = "small"
        elif conservative_changed <= 14:
            conservative_flip_label = "moderate"
        else:
            conservative_flip_label = "high"
        gain_line = (
            "- At least one conservative Title weight improves OOF over baseline; "
            f"the best conservative row is `w={best_conservative['weight']}` "
            f"with a {conservative_flip_label} flip count."
        )
        checkpoint_line = (
            "- A next frozen checkpoint is train-side justified only for the conservative "
            f"`w={best_conservative['weight']}` lane, subject to keeping it separate from final model selection."
        )
    else:
        gain_line = "- No conservative Title weight improves OOF over the pure raw_tabular baseline."
        checkpoint_line = "- A next frozen checkpoint is not justified by the conservative status rule in this check."

    if improved:
        best_improved = max(improved, key=lambda row: float(row["accuracy"]))
        best_overall_line = (
            "- The best OOF accuracy row is "
            f"`w={best_improved['weight']}`, with status `{best_improved['status']}`."
        )
    else:
        best_overall_line = "- No Title blending weight improves OOF over the pure raw_tabular baseline."

    return [
        gain_line,
        best_overall_line,
        checkpoint_line,
        "- This is not final model selection.",
    ]


def _build_report(
    train: pd.DataFrame,
    model_row: dict[str, object],
    rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    conservative = _conservative_rows(rows)
    lines = [
        "# 08 Title Gating Check",
        "",
        "## Scope Boundary",
        "",
        "- train-side OOF only",
        "- only `train.csv` is read",
        "- `Survived` is used only as the target",
        "- `add_clean_features()` is used only to create `Title`; only `Title` is copied from its output",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- same CV protocol and technical/default model parameters as `04_baseline`",
        "- `predict_proba()` is used only for OOF probability blending",
        "- fixed weight blending values only",
        "- no submission generation",
        "- no Kaggle/public leaderboard lookup",
        "- no `test.csv` scoring or inference",
        "- no test labels or row-level correctness checks",
        "- no `gender_submission.csv` as truth",
        "- no feature other than `Title` is added",
        "- no hyperparameter tuning, model parameter tuning, PassengerId overrides, manual correction rules, or target-derived features",
        "",
        "## Input Evidence",
        "",
        "- `raw_tabular / GradientBoostingClassifier` is the current clean public baseline leader from step 05.",
        "- unrestricted `raw_plus_title / GradientBoostingClassifier` was rejected for public transfer in step 07.",
        "- this step checks conservative Title blending train-side only.",
        "",
        "## CV / OOF Protocol",
        "",
        f"- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state={RANDOM_STATE})`",
        "- identical folds are used for base and title models",
        "- preprocessing is fitted inside each train fold through an sklearn `Pipeline`",
        "- base model: `raw_tabular / GradientBoostingClassifier`",
        "- title model: `raw_plus_title / GradientBoostingClassifier`",
        "- class-1 probabilities are aligned to `Survived == 1` before blending",
        f"- rows: `{len(train)}` from `train.csv`",
        "",
        "## Feature Sets",
        "",
        _markdown_table(
            [
                {"feature_set": BASE_FEATURE_SET, "features": ", ".join(BASE_FEATURES)},
                {"feature_set": TITLE_FEATURE_SET, "features": ", ".join(TITLE_FEATURES)},
            ],
            ["feature_set", "features"],
        ),
        "",
        "## Model Used",
        "",
        _markdown_table([model_row], MODEL_COLUMNS),
        "",
        "## Weight Grid",
        "",
        _markdown_table(
            [{"weight": f"{weight:.2f}"} for weight in WEIGHTS],
            ["weight"],
        ),
        "",
        "## OOF Results",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(rows, RESULT_COLUMNS),
        "",
        "## Best OOF row by `accuracy`",
        "",
        _markdown_table(_best_rows(rows), RESULT_COLUMNS),
        "",
        "## Conservative candidates",
        "",
        _markdown_table(conservative, RESULT_COLUMNS),
        "",
        "## Transfer-risk notes",
        "",
        "- This is train-side only.",
        "- Flip count is treated as a risk heuristic, not hard truth.",
        "- No submission was created.",
        "- No public score was used.",
        "",
        "## Short interpretation",
        "",
        *(_interpretation(rows)),
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
    model_row, model_available = _model_row()
    all_passed = model_available
    if not model_available:
        rows = []
    else:
        p_base, p_title = _compute_oof_probabilities(train)
        rows = _result_rows(train, p_base, p_title)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows)
    REPORT_PATH.write_text(
        _build_report(train, model_row, rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
