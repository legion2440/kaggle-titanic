from __future__ import annotations

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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "18b_surname_survival_gated_overlay_check.md"
OLD_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "18b_surname_survival_delta_fallback_neutral_check.py"
OLD_REPORT_PATH = REPORTS_DIR / "18b_surname_survival_delta_fallback_neutral_check.md"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
CANDIDATE_VARIANT = "raw_tabular_plus_surname_survival_gated_overlay"
SURNAME_FEATURE = "Surname"
SURNAME_SURVIVAL_RATE = "SurnameSurvivalRate"
SURNAME_COUNT_FROM_ENCODER = "SurnameCountFromEncoder"
SURNAME_OVERLAY_ACTIVE = "SurnameOverlayActive"
SURNAME_OVERLAY_DIRECTION = "SurnameOverlayDirection"
SURNAME_OVERLAY_REASON = "SurnameOverlayReason"

MIN_COUNT = 2
ALPHA = 5.0
CURRENT_PUBLIC_LEADER_SUBMISSION = "submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"
CURRENT_PUBLIC_LEADER_SCORE = "0.79904"

BASELINE_FEATURES = list(RAW_TABULAR)

METRIC_COLUMNS = [
    "variant",
    "model_or_candidate",
    "features",
    "cv_mean",
    "cv_std",
    "oof_accuracy",
    "oof_accuracy_delta_vs_raw_tabular",
    "oof_changed_rows",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "test_changed_rows",
    "test_0_to_1",
    "test_1_to_0",
    "test_pred_1_count",
    "test_pred_1_rate",
    "diagnostic_status",
]

MODEL_PANEL_COLUMNS = [
    "model_class",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

OOF_DIFF_COLUMNS = [
    "PassengerId",
    "Survived",
    "baseline_pred",
    "candidate_pred",
    "diff_direction",
    "rescue_or_kill",
    "Sex",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Surname",
    "SurnameSurvivalRate",
    "SurnameCountFromEncoder",
    "SurnameOverlayActive",
    "SurnameOverlayDirection",
    "SurnameOverlayReason",
]

TEST_DIFF_COLUMNS = [
    "PassengerId",
    "baseline_pred",
    "candidate_pred",
    "diff_direction",
    "Sex",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Surname",
    "SurnameSurvivalRate",
    "SurnameCountFromEncoder",
    "SurnameOverlayActive",
    "SurnameOverlayDirection",
    "SurnameOverlayReason",
]

ACTIVE_DIAGNOSTIC_COLUMNS = [
    "split",
    "active_changed_rows",
    "inactive_changed_rows",
    "active_0_to_1",
    "active_1_to_0",
    "active_rescue",
    "active_kill",
    "active_net",
]


@dataclass(frozen=True)
class SurnameEncoder:
    mapping: pd.Series
    counts: pd.Series
    global_survival_rate: float


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


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _csv_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    return value


def _extract_surname(frame: pd.DataFrame) -> pd.Series:
    return frame["Name"].str.extract(r"^([^,]+),", expand=False).str.strip().fillna("")


def _add_surname(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[SURNAME_FEATURE] = _extract_surname(out)
    return out


def _fit_surname_encoder(frame: pd.DataFrame) -> SurnameEncoder:
    global_survival_rate = float(frame[TARGET].astype(int).mean())
    grouped = frame.groupby(SURNAME_FEATURE, dropna=False)[TARGET].agg(["sum", "count"])
    mapping = (grouped["sum"] + ALPHA * global_survival_rate) / (grouped["count"] + ALPHA)
    return SurnameEncoder(
        mapping=mapping.astype(float),
        counts=grouped["count"].astype(int),
        global_survival_rate=global_survival_rate,
    )


def _add_surname_overlay_columns(frame: pd.DataFrame, encoder: SurnameEncoder) -> pd.DataFrame:
    out = frame.copy()
    counts = out[SURNAME_FEATURE].map(encoder.counts).fillna(0).astype(int)
    rates = out[SURNAME_FEATURE].map(encoder.mapping).fillna(encoder.global_survival_rate).astype(float)
    active = counts >= MIN_COUNT

    direction = np.full(len(out), "keep_baseline", dtype=object)
    reason = np.full(len(out), "inactive_count_lt_min", dtype=object)
    set_to_1 = active & (rates >= 0.5)
    set_to_0 = active & (rates <= encoder.global_survival_rate)
    neutral = active & ~(set_to_1 | set_to_0)

    direction[set_to_1.to_numpy()] = "set_to_1"
    reason[set_to_1.to_numpy()] = "active_rate_ge_0_5"
    direction[set_to_0.to_numpy()] = "set_to_0"
    reason[set_to_0.to_numpy()] = "active_rate_le_global"
    direction[neutral.to_numpy()] = "keep_baseline"
    reason[neutral.to_numpy()] = "active_neutral_middle_band"

    out[SURNAME_SURVIVAL_RATE] = rates
    out[SURNAME_COUNT_FROM_ENCODER] = counts
    out[SURNAME_OVERLAY_ACTIVE] = active.astype(int)
    out[SURNAME_OVERLAY_DIRECTION] = direction
    out[SURNAME_OVERLAY_REASON] = reason
    return out


def _overlay_predictions(baseline_pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    candidate = baseline_pred.copy()
    active = frame[SURNAME_OVERLAY_ACTIVE].to_numpy().astype(bool)
    set_to_1 = active & frame[SURNAME_OVERLAY_DIRECTION].eq("set_to_1").to_numpy()
    set_to_0 = active & frame[SURNAME_OVERLAY_DIRECTION].eq("set_to_0").to_numpy()
    candidate[set_to_1] = 1
    candidate[set_to_0] = 0
    return candidate.astype(int)


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPECS_BY_NAME[MODEL_NAME])


def _build_estimator() -> tuple[Pipeline | None, dict[str, Any], str]:
    model, used_params, adjustment = _build_model()
    if model is None:
        return None, used_params, adjustment
    return (
        Pipeline(
            steps=[
                ("preprocess", make_preprocessor("unscaled_tree", BASELINE_FEATURES)),
                ("model", model),
            ]
        ),
        used_params,
        adjustment,
    )


def _model_panel_rows() -> list[dict[str, object]]:
    model, used_params, adjustment = _build_model()
    spec = MODEL_SPECS_BY_NAME[MODEL_NAME]
    package_version = baseline04._package_version(spec.version_package)
    if model is not None and hasattr(model, "get_params"):
        actual_params: object = model.get_params(deep=False)
    elif model is not None:
        actual_params = "get_params_unavailable"
    else:
        actual_params = "model_unavailable"
    return [
        {
            "model_class": MODEL_NAME,
            "package": spec.package,
            "package_version": package_version,
            "preprocessing_mode": spec.preprocessing_mode,
            "explicit_technical_params": _json_dumps(used_params),
            "actual_resolved_params": _json_dumps(actual_params),
            "parameter_adjustments": "" if model is None else adjustment,
            "error": adjustment if model is None else "",
        }
    ]


def _evaluate_oof(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    baseline_oof = np.full(len(train), -1, dtype=int)
    candidate_oof = np.full(len(train), -1, dtype=int)
    baseline_fold_scores: list[float] = []
    candidate_fold_scores: list[float] = []
    overlay_parts: list[pd.DataFrame] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        estimator, _, build_error = _build_estimator()
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(fold_train[BASELINE_FEATURES], y.iloc[train_idx])
        fold_baseline = estimator.predict(fold_valid[BASELINE_FEATURES]).astype(int)

        encoder = _fit_surname_encoder(fold_train)
        fold_valid_overlay = _add_surname_overlay_columns(fold_valid, encoder)
        fold_candidate = _overlay_predictions(fold_baseline, fold_valid_overlay)

        if fold_id <= 5:
            baseline_oof[valid_idx] = fold_baseline
            candidate_oof[valid_idx] = fold_candidate
            fold_valid_overlay = fold_valid_overlay.copy()
            fold_valid_overlay["Fold"] = fold_id
            fold_valid_overlay["baseline_pred"] = fold_baseline
            fold_valid_overlay["candidate_pred"] = fold_candidate
            overlay_parts.append(fold_valid_overlay)
        baseline_fold_scores.append(float((fold_baseline == y.iloc[valid_idx].to_numpy()).mean()))
        candidate_fold_scores.append(float((fold_candidate == y.iloc[valid_idx].to_numpy()).mean()))

    if (baseline_oof < 0).any() or (candidate_oof < 0).any():
        raise RuntimeError("OOF prediction assignment incomplete")

    overlay_frame = pd.concat(overlay_parts).sort_index()
    return {
        "baseline_oof": baseline_oof,
        "candidate_oof": candidate_oof,
        "baseline_fold_scores": baseline_fold_scores,
        "candidate_fold_scores": candidate_fold_scores,
        "overlay_frame": overlay_frame,
    }


def _evaluate_test(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    estimator, _, build_error = _build_estimator()
    if estimator is None:
        raise RuntimeError(build_error)
    estimator.fit(train[BASELINE_FEATURES], train[TARGET].astype(int))
    baseline_pred = estimator.predict(test[BASELINE_FEATURES]).astype(int)

    encoder = _fit_surname_encoder(train)
    overlay_frame = _add_surname_overlay_columns(test, encoder)
    candidate_pred = _overlay_predictions(baseline_pred, overlay_frame)
    overlay_frame = overlay_frame.copy()
    overlay_frame["baseline_pred"] = baseline_pred
    overlay_frame["candidate_pred"] = candidate_pred
    return {
        "baseline_pred": baseline_pred,
        "candidate_pred": candidate_pred,
        "overlay_frame": overlay_frame,
    }


def _direction(baseline_pred: int, candidate_pred: int) -> str:
    if baseline_pred == 0 and candidate_pred == 1:
        return "upshift_0_to_1"
    if baseline_pred == 1 and candidate_pred == 0:
        return "downshift_1_to_0"
    return "unchanged"


def _train_diff_type(baseline_correct: bool, candidate_correct: bool) -> str:
    if not baseline_correct and candidate_correct:
        return "rescue"
    if baseline_correct and not candidate_correct:
        return "kill"
    return "changed_same_correctness"


def _diagnostic_status(candidate_accuracy: float, baseline_accuracy: float, net: int) -> str:
    if candidate_accuracy > baseline_accuracy and net > 0:
        return "OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    if candidate_accuracy < baseline_accuracy or net < 0:
        return "OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    return "OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN"


def _verify_inactive_invariant(overlay_frame: pd.DataFrame, split_name: str) -> int:
    changed = overlay_frame["baseline_pred"].astype(int).to_numpy() != overlay_frame["candidate_pred"].astype(int).to_numpy()
    inactive = overlay_frame[SURNAME_OVERLAY_ACTIVE].astype(int).to_numpy() == 0
    inactive_changed = int((changed & inactive).sum())
    if inactive_changed != 0:
        raise RuntimeError(f"{split_name}: inactive changed rows invariant failed: {inactive_changed}")
    return inactive_changed


def _metric_rows(
    train: pd.DataFrame,
    oof_result: dict[str, object],
    test_result: dict[str, object],
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    baseline_oof = oof_result["baseline_oof"]
    candidate_oof = oof_result["candidate_oof"]
    baseline_test = test_result["baseline_pred"]
    candidate_test = test_result["candidate_pred"]

    baseline_correct = baseline_oof == y
    candidate_correct = candidate_oof == y
    rescue = int((~baseline_correct & candidate_correct).sum())
    kill = int((baseline_correct & ~candidate_correct).sum())
    net = rescue - kill

    baseline_accuracy = float((baseline_oof == y).mean())
    candidate_accuracy = float((candidate_oof == y).mean())
    status = _diagnostic_status(candidate_accuracy, baseline_accuracy, net)

    rows = []
    specs = [
        {
            "variant": BASELINE_VARIANT,
            "model_or_candidate": MODEL_NAME,
            "features": ", ".join(BASELINE_FEATURES),
            "fold_scores": oof_result["baseline_fold_scores"],
            "oof": baseline_oof,
            "test_pred": baseline_test,
            "status": "",
        },
        {
            "variant": CANDIDATE_VARIANT,
            "model_or_candidate": "post-model gated overlay",
            "features": "raw_tabular baseline predictions + SurnameSurvival gated overlay",
            "fold_scores": oof_result["candidate_fold_scores"],
            "oof": candidate_oof,
            "test_pred": candidate_test,
            "status": status,
        },
    ]

    for spec in specs:
        oof = spec["oof"]
        test_pred = spec["test_pred"]
        changed = oof != baseline_oof
        test_changed = test_pred != baseline_test
        rows.append(
            {
                "variant": spec["variant"],
                "model_or_candidate": spec["model_or_candidate"],
                "features": spec["features"],
                "cv_mean": _round_float(float(np.mean(spec["fold_scores"]))),
                "cv_std": _round_float(float(np.std(spec["fold_scores"]))),
                "oof_accuracy": _round_float(float((oof == y).mean())),
                "oof_accuracy_delta_vs_raw_tabular": _round_float(float((oof == y).mean()) - baseline_accuracy),
                "oof_changed_rows": int(changed.sum()),
                "oof_0_to_1": int(((baseline_oof == 0) & (oof == 1)).sum()),
                "oof_1_to_0": int(((baseline_oof == 1) & (oof == 0)).sum()),
                "rescue": 0 if spec["variant"] == BASELINE_VARIANT else rescue,
                "kill": 0 if spec["variant"] == BASELINE_VARIANT else kill,
                "net": 0 if spec["variant"] == BASELINE_VARIANT else net,
                "test_changed_rows": int(test_changed.sum()),
                "test_0_to_1": int(((baseline_test == 0) & (test_pred == 1)).sum()),
                "test_1_to_0": int(((baseline_test == 1) & (test_pred == 0)).sum()),
                "test_pred_1_count": int((test_pred == 1).sum()),
                "test_pred_1_rate": _round_float(float((test_pred == 1).mean())),
                "diagnostic_status": spec["status"],
            }
        )

    return rows


def _oof_diff_rows(train: pd.DataFrame, oof_result: dict[str, object]) -> list[dict[str, object]]:
    overlay = oof_result["overlay_frame"]
    y = train[TARGET].astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    changed_idx = np.flatnonzero(
        overlay["baseline_pred"].astype(int).to_numpy() != overlay["candidate_pred"].astype(int).to_numpy()
    )
    for idx in changed_idx:
        row = overlay.iloc[int(idx)]
        baseline_pred = int(row["baseline_pred"])
        candidate_pred = int(row["candidate_pred"])
        baseline_correct = bool(baseline_pred == y[int(idx)])
        candidate_correct = bool(candidate_pred == y[int(idx)])
        rows.append(
            {
                "PassengerId": _csv_scalar(row[ID_COLUMN]),
                "Survived": int(y[int(idx)]),
                "baseline_pred": baseline_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(baseline_pred, candidate_pred),
                "rescue_or_kill": _train_diff_type(baseline_correct, candidate_correct),
                "Sex": _csv_scalar(row["Sex"]),
                "Pclass": _csv_scalar(row["Pclass"]),
                "Age": _csv_scalar(row["Age"]),
                "SibSp": _csv_scalar(row["SibSp"]),
                "Parch": _csv_scalar(row["Parch"]),
                "Fare": _csv_scalar(row["Fare"]),
                "Embarked": _csv_scalar(row["Embarked"]),
                "Surname": _csv_scalar(row[SURNAME_FEATURE]),
                "SurnameSurvivalRate": _round_float(float(row[SURNAME_SURVIVAL_RATE])),
                "SurnameCountFromEncoder": int(row[SURNAME_COUNT_FROM_ENCODER]),
                "SurnameOverlayActive": int(row[SURNAME_OVERLAY_ACTIVE]),
                "SurnameOverlayDirection": row[SURNAME_OVERLAY_DIRECTION],
                "SurnameOverlayReason": row[SURNAME_OVERLAY_REASON],
            }
        )
    return rows


def _test_diff_rows(test_result: dict[str, object]) -> list[dict[str, object]]:
    overlay = test_result["overlay_frame"]
    rows: list[dict[str, object]] = []
    changed_idx = np.flatnonzero(
        overlay["baseline_pred"].astype(int).to_numpy() != overlay["candidate_pred"].astype(int).to_numpy()
    )
    for idx in changed_idx:
        row = overlay.iloc[int(idx)]
        baseline_pred = int(row["baseline_pred"])
        candidate_pred = int(row["candidate_pred"])
        rows.append(
            {
                "PassengerId": _csv_scalar(row[ID_COLUMN]),
                "baseline_pred": baseline_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(baseline_pred, candidate_pred),
                "Sex": _csv_scalar(row["Sex"]),
                "Pclass": _csv_scalar(row["Pclass"]),
                "Age": _csv_scalar(row["Age"]),
                "SibSp": _csv_scalar(row["SibSp"]),
                "Parch": _csv_scalar(row["Parch"]),
                "Fare": _csv_scalar(row["Fare"]),
                "Embarked": _csv_scalar(row["Embarked"]),
                "Surname": _csv_scalar(row[SURNAME_FEATURE]),
                "SurnameSurvivalRate": _round_float(float(row[SURNAME_SURVIVAL_RATE])),
                "SurnameCountFromEncoder": int(row[SURNAME_COUNT_FROM_ENCODER]),
                "SurnameOverlayActive": int(row[SURNAME_OVERLAY_ACTIVE]),
                "SurnameOverlayDirection": row[SURNAME_OVERLAY_DIRECTION],
                "SurnameOverlayReason": row[SURNAME_OVERLAY_REASON],
            }
        )
    return rows


def _active_diagnostic_rows(train: pd.DataFrame, oof_result: dict[str, object], test_result: dict[str, object]) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    rows: list[dict[str, object]] = []

    for split, overlay, include_correctness in [
        ("OOF", oof_result["overlay_frame"], True),
        ("test", test_result["overlay_frame"], False),
    ]:
        baseline = overlay["baseline_pred"].astype(int).to_numpy()
        candidate = overlay["candidate_pred"].astype(int).to_numpy()
        active = overlay[SURNAME_OVERLAY_ACTIVE].astype(int).to_numpy() == 1
        changed = baseline != candidate
        active_changed = changed & active
        inactive_changed = changed & ~active

        if include_correctness:
            baseline_correct = baseline == y
            candidate_correct = candidate == y
            active_rescue = int((active_changed & ~baseline_correct & candidate_correct).sum())
            active_kill = int((active_changed & baseline_correct & ~candidate_correct).sum())
        else:
            active_rescue = ""
            active_kill = ""

        rows.append(
            {
                "split": split,
                "active_changed_rows": int(active_changed.sum()),
                "inactive_changed_rows": int(inactive_changed.sum()),
                "active_0_to_1": int((active_changed & (baseline == 0) & (candidate == 1)).sum()),
                "active_1_to_0": int((active_changed & (baseline == 1) & (candidate == 0)).sum()),
                "active_rescue": active_rescue,
                "active_kill": active_kill,
                "active_net": "" if not include_correctness else int(active_rescue) - int(active_kill),
            }
        )

    return rows


def _write_report(
    metric_rows: list[dict[str, object]],
    model_panel_rows: list[dict[str, object]],
    oof_diff_rows: list[dict[str, object]],
    test_diff_rows: list[dict[str, object]],
    active_rows: list[dict[str, object]],
    oof_inactive_changed: int,
    test_inactive_changed: int,
) -> None:
    baseline_row = next(row for row in metric_rows if row["variant"] == BASELINE_VARIANT)
    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    old_script_status = "absent / removed" if not OLD_SCRIPT_PATH.exists() else "still present"
    old_report_status = "absent / removed" if not OLD_REPORT_PATH.exists() else "still present"

    lines = [
        "# 18B SurnameSurvival Gated Overlay Check",
        "",
        "## Purpose",
        "",
        "This diagnostic replaces the old Step 18B delta feature check with a controlled post-model overlay. The baseline model remains the unchanged `raw_tabular / GradientBoostingClassifier`; the candidate copies baseline predictions and applies one surname rule only on active repeated-surname rows.",
        "",
        f"Current public leader remains frozen: `{CURRENT_PUBLIC_LEADER_SUBMISSION}`, public score `{CURRENT_PUBLIC_LEADER_SCORE}`.",
        "",
        "## Why old 18B was removed/replaced",
        "",
        "The old Step 18B `SurnameSurvivalDelta` fallback-neutral check still trained `GradientBoostingClassifier` with a target-derived surname feature. That did not isolate a surname correction path because the model could reshape its full decision surface. This replacement keeps GB unchanged and applies a bounded overlay after prediction.",
        "",
        "## Method boundary",
        "",
        "- This is diagnostic only and not public tuning.",
        "- The candidate is not a retrained model with extra features.",
        "- `raw_tabular` baseline features remain `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`.",
        "- `GradientBoostingClassifier` hyperparameters are unchanged.",
        "- Existing submissions and the frozen public leader are not altered.",
        "- PassengerId is not used as a rule, feature, lookup key, or tuning input.",
        "- Step 17 structural docs and closed structural lanes are not reopened.",
        "",
        "## Anti-leakage / fold-safe notes",
        "",
        "- For each CV fold, the baseline GB model is trained only on train-fold rows.",
        "- Validation-fold baseline predictions are OOF predictions from that fold model.",
        "- For each CV fold, the surname encoder is fitted only on train-fold rows.",
        "- Validation-fold labels are never used to build surname encoding.",
        "- Unknown surnames and train-fold surname counts below `min_count` are inactive and keep baseline predictions.",
        "- OOF `global_survival_rate` is fold-specific.",
        "- Test overlay is transformed from a full-train surname map after OOF validation.",
        "- Test labels are never used.",
        "",
        "## Overlay rule",
        "",
        "- `Surname` is the substring before comma in `Name`.",
        "- Smoothed surname survival rate: `(survived_sum + alpha * global_survival_rate) / (surname_count + alpha)`.",
        f"- Fixed parameters: `min_count={MIN_COUNT}`, `alpha={ALPHA}`.",
        "- `active = SurnameCountFromEncoder >= min_count`.",
        "- If inactive: `candidate_pred = baseline_pred`.",
        "- If active and `SurnameSurvivalRate >= 0.5`: `candidate_pred = 1`.",
        "- If active and `SurnameSurvivalRate <= global_survival_rate`: `candidate_pred = 0`.",
        "- Otherwise: `candidate_pred = baseline_pred`.",
        "- Thresholds are fixed and not tuned.",
        "",
        "## Baseline metrics",
        "",
        _markdown_table([baseline_row], METRIC_COLUMNS),
        "",
        "## Overlay candidate metrics",
        "",
        _markdown_table([candidate_row], METRIC_COLUMNS),
        "",
        "## Required metric summary",
        "",
        f"- baseline OOF accuracy: `{baseline_row['oof_accuracy']}`",
        f"- candidate OOF accuracy: `{candidate_row['oof_accuracy']}`",
        f"- delta vs raw_tabular: `{candidate_row['oof_accuracy_delta_vs_raw_tabular']}`",
        f"- OOF changed rows: `{candidate_row['oof_changed_rows']}`",
        f"- OOF 0 -> 1: `{candidate_row['oof_0_to_1']}`",
        f"- OOF 1 -> 0: `{candidate_row['oof_1_to_0']}`",
        f"- rescue / kill / net: `{candidate_row['rescue']}` / `{candidate_row['kill']}` / `{candidate_row['net']}`",
        f"- test changed rows: `{candidate_row['test_changed_rows']}`",
        f"- test 0 -> 1: `{candidate_row['test_0_to_1']}`",
        f"- test 1 -> 0: `{candidate_row['test_1_to_0']}`",
        f"- baseline test predicted survivors and rate: `{baseline_row['test_pred_1_count']}` / `{baseline_row['test_pred_1_rate']}`",
        f"- candidate test predicted survivors and rate: `{candidate_row['test_pred_1_count']}` / `{candidate_row['test_pred_1_rate']}`",
        "",
        "## Model panel",
        "",
        _markdown_table(model_panel_rows, MODEL_PANEL_COLUMNS),
        "",
        "## Safety invariant",
        "",
        f"- OOF changed rows where `SurnameOverlayActive == 0`: `{oof_inactive_changed}`",
        f"- test changed rows where `SurnameOverlayActive == 0`: `{test_inactive_changed}`",
        "- invariant: `No inactive row changed prediction.`",
        "",
        "## OOF diff audit",
        "",
        _markdown_table(oof_diff_rows, OOF_DIFF_COLUMNS),
        "",
        "## Test diff audit",
        "",
        _markdown_table(test_diff_rows, TEST_DIFF_COLUMNS),
        "",
        "## Diff-by-active diagnostic",
        "",
        _markdown_table(active_rows, ACTIVE_DIAGNOSTIC_COLUMNS),
        "",
        "## Decision wording",
        "",
        f"Diagnostic status: **{candidate_row['diagnostic_status']}**",
        "",
        "This does not claim public transfer, does not claim public score is worse or better, and does not auto-close the SurnameSurvival lane. Final project decision is not automatic.",
        "",
        "## Deleted old artifact note",
        "",
        f"- `{_relative(OLD_SCRIPT_PATH)}`: `{old_script_status}`",
        f"- `{_relative(OLD_REPORT_PATH)}`: `{old_report_status}`",
        "",
        "## Submission status",
        "",
        "No submission was created.",
        "",
        "## Output files",
        "",
        f"- report: `{_relative(REPORT_PATH)}`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train = _add_surname(pd.read_csv(TRAIN_PATH))
    test = _add_surname(pd.read_csv(TEST_PATH))
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]

    oof_result = _evaluate_oof(train, splits, y)
    test_result = _evaluate_test(train, test)
    oof_inactive_changed = _verify_inactive_invariant(oof_result["overlay_frame"], "OOF")
    test_inactive_changed = _verify_inactive_invariant(test_result["overlay_frame"], "test")

    metric_rows = _metric_rows(train, oof_result, test_result)
    model_panel_rows = _model_panel_rows()
    oof_rows = _oof_diff_rows(train, oof_result)
    test_rows = _test_diff_rows(test_result)
    active_rows = _active_diagnostic_rows(train, oof_result, test_result)

    _write_report(
        metric_rows,
        model_panel_rows,
        oof_rows,
        test_rows,
        active_rows,
        oof_inactive_changed,
        test_inactive_changed,
    )

    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    print(f"wrote {_relative(REPORT_PATH)}")
    print(f"diagnostic_status={candidate_row['diagnostic_status']}")
    print(
        "oof_delta={oof_delta} rescue={rescue} kill={kill} net={net} test_changed_rows={test_changed}".format(
            oof_delta=candidate_row["oof_accuracy_delta_vs_raw_tabular"],
            rescue=candidate_row["rescue"],
            kill=candidate_row["kill"],
            net=candidate_row["net"],
            test_changed=candidate_row["test_changed_rows"],
        )
    )
    print(f"inactive_changed_oof={oof_inactive_changed} inactive_changed_test={test_inactive_changed}")
    print("no submission was created")


if __name__ == "__main__":
    main()
