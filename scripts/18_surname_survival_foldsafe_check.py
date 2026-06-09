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
import scripts.preprocessing as preprocessing


REPORT_PATH = REPORTS_DIR / "18_surname_survival_foldsafe_check.md"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
CANDIDATE_VARIANT = "raw_tabular_plus_surname_survival"
SURNAME_FEATURE = "Surname"
SURNAME_SURVIVAL_FEATURE = "SurnameSurvival"

MIN_COUNT = 2
ALPHA = 5.0
CURRENT_PUBLIC_LEADER_SUBMISSION = "submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"
CURRENT_PUBLIC_LEADER_SCORE = "0.79904"

BASELINE_FEATURES = list(RAW_TABULAR)
CANDIDATE_FEATURES = [*RAW_TABULAR, SURNAME_SURVIVAL_FEATURE]


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    candidate_id: str
    feature_set: str
    features: list[str]
    purpose: str


@dataclass(frozen=True)
class SurnameEncoder:
    mapping: pd.Series
    counts: pd.Series
    global_survival_rate: float


VARIANTS = [
    VariantSpec(
        variant=BASELINE_VARIANT,
        candidate_id="raw_tabular__GradientBoostingClassifier",
        feature_set="raw_tabular",
        features=BASELINE_FEATURES,
        purpose="unchanged raw_tabular GB baseline",
    ),
    VariantSpec(
        variant=CANDIDATE_VARIANT,
        candidate_id="raw_tabular_plus_surname_survival__GradientBoostingClassifier",
        feature_set="raw_tabular_plus_surname_survival",
        features=CANDIDATE_FEATURES,
        purpose="one fixed fold-safe smoothed surname survival-rate check",
    ),
]

METRIC_COLUMNS = [
    "model_name",
    "variant",
    "candidate_id",
    "feature_set",
    "features",
    "cv_mean",
    "cv_std",
    "oof_accuracy",
    "oof_accuracy_delta_vs_raw_tabular",
    "oof_changed_rows",
    "oof_changed_pct",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "oof_pred_1_count",
    "oof_pred_1_rate",
    "oof_pred_1_rate_delta_vs_raw_tabular",
    "test_changed_rows",
    "test_changed_pct",
    "test_0_to_1",
    "test_1_to_0",
    "test_pred_1_count",
    "test_pred_1_rate",
    "test_pred_1_rate_delta_vs_raw_tabular",
    "train_survival_rate",
    "decision",
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

DIFF_ROW_COLUMNS = [
    "split",
    "PassengerId",
    "Survived",
    "raw_pred",
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
    "SurnameSurvival",
    "SurnameCountFromEncoder",
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


def _transform_surname_encoder(
    encoder: SurnameEncoder,
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    counts = frame[SURNAME_FEATURE].map(encoder.counts).fillna(0).astype(int)
    values = frame[SURNAME_FEATURE].map(encoder.mapping)
    values = values.where(counts >= MIN_COUNT, encoder.global_survival_rate)
    values = values.fillna(encoder.global_survival_rate).astype(float)
    return values, counts


def _add_surname_survival_from_encoder(
    frame: pd.DataFrame,
    encoder: SurnameEncoder,
) -> tuple[pd.DataFrame, pd.Series]:
    out = frame.copy()
    values, counts = _transform_surname_encoder(encoder, out)
    out[SURNAME_SURVIVAL_FEATURE] = values
    return out, counts


def _ensure_surname_preprocessor_support() -> None:
    if SURNAME_SURVIVAL_FEATURE not in preprocessing.NUMERIC_FEATURES:
        preprocessing.NUMERIC_FEATURES.append(SURNAME_SURVIVAL_FEATURE)


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPECS_BY_NAME[MODEL_NAME])


def _build_estimator(features: list[str]) -> tuple[Pipeline | None, dict[str, Any], str]:
    _ensure_surname_preprocessor_support()
    model, used_params, adjustment = _build_model()
    if model is None:
        return None, used_params, adjustment
    return (
        Pipeline(
            steps=[
                ("preprocess", preprocessing.make_preprocessor("unscaled_tree", list(features))),
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


def _evaluate_raw_tabular(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    missing_features = [feature for feature in BASELINE_FEATURES if feature not in train.columns]
    if missing_features:
        raise ValueError(f"{BASELINE_VARIANT}: missing feature columns: {missing_features}")

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)
    for i, (train_idx, valid_idx) in enumerate(splits):
        estimator, _, build_error = _build_estimator(BASELINE_FEATURES)
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(train[BASELINE_FEATURES].iloc[train_idx], y.iloc[train_idx])
        fold_pred = estimator.predict(train[BASELINE_FEATURES].iloc[valid_idx]).astype(int)
        if i < 5:
            oof[valid_idx] = fold_pred
        fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

    if (oof < 0).any():
        raise RuntimeError(f"{BASELINE_VARIANT}: OOF prediction assignment incomplete")

    return {
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_accuracy": float((oof == y.to_numpy()).mean()),
        "pred_1_count": int((oof == 1).sum()),
        "pred_1_rate": float((oof == 1).mean()),
        "oof": oof,
    }


def _evaluate_surname_survival(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    missing_features = [feature for feature in CANDIDATE_FEATURES if feature not in train.columns and feature != SURNAME_SURVIVAL_FEATURE]
    if missing_features:
        raise ValueError(f"{CANDIDATE_VARIANT}: missing feature columns: {missing_features}")

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)
    oof_surname_survival = np.full(len(train), np.nan, dtype=float)
    oof_surname_count = np.zeros(len(train), dtype=int)

    for i, (train_idx, valid_idx) in enumerate(splits):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()
        encoder = _fit_surname_encoder(fold_train)
        fold_train, _ = _add_surname_survival_from_encoder(fold_train, encoder)
        fold_valid, valid_counts = _add_surname_survival_from_encoder(fold_valid, encoder)

        estimator, _, build_error = _build_estimator(CANDIDATE_FEATURES)
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(fold_train[CANDIDATE_FEATURES], y.iloc[train_idx])
        fold_pred = estimator.predict(fold_valid[CANDIDATE_FEATURES]).astype(int)
        if i < 5:
            oof[valid_idx] = fold_pred
            oof_surname_survival[valid_idx] = fold_valid[SURNAME_SURVIVAL_FEATURE].to_numpy()
            oof_surname_count[valid_idx] = valid_counts.to_numpy()
        fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

    if (oof < 0).any():
        raise RuntimeError(f"{CANDIDATE_VARIANT}: OOF prediction assignment incomplete")
    if np.isnan(oof_surname_survival).any():
        raise RuntimeError(f"{CANDIDATE_VARIANT}: OOF surname encoding assignment incomplete")

    return {
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_accuracy": float((oof == y.to_numpy()).mean()),
        "pred_1_count": int((oof == 1).sum()),
        "pred_1_rate": float((oof == 1).mean()),
        "oof": oof,
        "oof_surname_survival": oof_surname_survival,
        "oof_surname_count": oof_surname_count,
    }


def _fit_full_predict_raw(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    estimator, _, build_error = _build_estimator(BASELINE_FEATURES)
    if estimator is None:
        raise RuntimeError(build_error)
    estimator.fit(train[BASELINE_FEATURES], train[TARGET].astype(int))
    return estimator.predict(test[BASELINE_FEATURES]).astype(int)


def _fit_full_predict_surname(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    encoder = _fit_surname_encoder(train)
    train_encoded, train_counts = _add_surname_survival_from_encoder(train, encoder)
    test_encoded, test_counts = _add_surname_survival_from_encoder(test, encoder)
    estimator, _, build_error = _build_estimator(CANDIDATE_FEATURES)
    if estimator is None:
        raise RuntimeError(build_error)
    estimator.fit(train_encoded[CANDIDATE_FEATURES], train_encoded[TARGET].astype(int))
    pred = estimator.predict(test_encoded[CANDIDATE_FEATURES]).astype(int)
    return pred, train_encoded, test_encoded, train_counts, test_counts


def _direction(raw_pred: int, candidate_pred: int) -> str:
    if raw_pred == 0 and candidate_pred == 1:
        return "upshift_0_to_1"
    if raw_pred == 1 and candidate_pred == 0:
        return "downshift_1_to_0"
    return "unchanged"


def _train_diff_type(raw_correct: bool, candidate_correct: bool) -> str:
    if not raw_correct and candidate_correct:
        return "rescue"
    if raw_correct and not candidate_correct:
        return "kill"
    return "changed_same_correctness"


def _base_row_fields(
    frame: pd.DataFrame,
    idx: int,
    surname_survival: float,
    surname_count: int,
) -> dict[str, object]:
    row = frame.iloc[idx]
    return {
        "PassengerId": _csv_scalar(row[ID_COLUMN]),
        "Sex": _csv_scalar(row["Sex"]),
        "Pclass": _csv_scalar(row["Pclass"]),
        "Age": _csv_scalar(row["Age"]),
        "SibSp": _csv_scalar(row["SibSp"]),
        "Parch": _csv_scalar(row["Parch"]),
        "Fare": _csv_scalar(row["Fare"]),
        "Embarked": _csv_scalar(row["Embarked"]),
        "Surname": _csv_scalar(row[SURNAME_FEATURE]),
        "SurnameSurvival": _round_float(surname_survival),
        "SurnameCountFromEncoder": int(surname_count),
    }


def _diff_rows(
    train: pd.DataFrame,
    test_encoded: pd.DataFrame,
    results: dict[str, dict[str, object]],
    raw_test_pred: np.ndarray,
    candidate_test_pred: np.ndarray,
    test_surname_counts: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_oof = results[BASELINE_VARIANT]["oof"]
    candidate_oof = results[CANDIDATE_VARIANT]["oof"]
    candidate_oof_values = results[CANDIDATE_VARIANT]["oof_surname_survival"]
    candidate_oof_counts = results[CANDIDATE_VARIANT]["oof_surname_count"]
    y = train[TARGET].astype(int).to_numpy()

    for idx in np.flatnonzero(candidate_oof != raw_oof):
        raw_pred = int(raw_oof[idx])
        candidate_pred = int(candidate_oof[idx])
        raw_correct = bool(raw_pred == y[idx])
        candidate_correct = bool(candidate_pred == y[idx])
        rows.append(
            {
                "split": "train_oof",
                "Survived": int(y[idx]),
                "raw_pred": raw_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(raw_pred, candidate_pred),
                "rescue_or_kill": _train_diff_type(raw_correct, candidate_correct),
                **_base_row_fields(
                    train,
                    int(idx),
                    float(candidate_oof_values[idx]),
                    int(candidate_oof_counts[idx]),
                ),
            }
        )

    for idx in np.flatnonzero(candidate_test_pred != raw_test_pred):
        raw_pred = int(raw_test_pred[idx])
        candidate_pred = int(candidate_test_pred[idx])
        rows.append(
            {
                "split": "test_full_fit",
                "Survived": "",
                "raw_pred": raw_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(raw_pred, candidate_pred),
                "rescue_or_kill": "test_changed",
                **_base_row_fields(
                    test_encoded,
                    int(idx),
                    float(test_encoded[SURNAME_SURVIVAL_FEATURE].iloc[idx]),
                    int(test_surname_counts.iloc[idx]),
                ),
            }
        )

    return rows


def _decision(candidate_row: dict[str, object]) -> str:
    oof_delta = float(candidate_row["oof_accuracy_delta_vs_raw_tabular"])
    net = int(candidate_row["net"])
    if oof_delta > 0 and net > 0:
        return "KEEP_TRAIN_SIDE"
    return "REJECT_TRAIN_SIDE"


def _metric_rows(
    train: pd.DataFrame,
    results: dict[str, dict[str, object]],
    raw_test_pred: np.ndarray,
    candidate_test_pred: np.ndarray,
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    base = results[BASELINE_VARIANT]
    base_oof = base["oof"]
    train_survival_rate = float(train[TARGET].mean())
    test_predictions = {
        BASELINE_VARIANT: raw_test_pred,
        CANDIDATE_VARIANT: candidate_test_pred,
    }

    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        result = results[variant.variant]
        oof = result["oof"]
        full_fit_pred = test_predictions[variant.variant]
        changed = oof != base_oof
        raw_correct = base_oof == y
        candidate_correct = oof == y
        rescue = int((~raw_correct & candidate_correct).sum())
        kill = int((raw_correct & ~candidate_correct).sum())
        net = rescue - kill
        test_changed = full_fit_pred != raw_test_pred
        row = {
            "model_name": MODEL_NAME,
            "variant": variant.variant,
            "candidate_id": variant.candidate_id,
            "feature_set": variant.feature_set,
            "features": ", ".join(variant.features),
            "cv_mean": _round_float(result["cv_mean"]),
            "cv_std": _round_float(result["cv_std"]),
            "oof_accuracy": _round_float(result["oof_accuracy"]),
            "oof_accuracy_delta_vs_raw_tabular": _round_float(float(result["oof_accuracy"]) - float(base["oof_accuracy"])),
            "oof_changed_rows": int(changed.sum()),
            "oof_changed_pct": _round_float(float(changed.mean() * 100)),
            "oof_0_to_1": int(((base_oof == 0) & (oof == 1)).sum()),
            "oof_1_to_0": int(((base_oof == 1) & (oof == 0)).sum()),
            "rescue": rescue,
            "kill": kill,
            "net": net,
            "oof_pred_1_count": int(result["pred_1_count"]),
            "oof_pred_1_rate": _round_float(result["pred_1_rate"]),
            "oof_pred_1_rate_delta_vs_raw_tabular": _round_float(float(result["pred_1_rate"]) - float(base["pred_1_rate"])),
            "test_changed_rows": int(test_changed.sum()),
            "test_changed_pct": _round_float(float(test_changed.mean() * 100)),
            "test_0_to_1": int(((raw_test_pred == 0) & (full_fit_pred == 1)).sum()),
            "test_1_to_0": int(((raw_test_pred == 1) & (full_fit_pred == 0)).sum()),
            "test_pred_1_count": int((full_fit_pred == 1).sum()),
            "test_pred_1_rate": _round_float(float((full_fit_pred == 1).mean())),
            "test_pred_1_rate_delta_vs_raw_tabular": _round_float(
                float((full_fit_pred == 1).mean()) - float((raw_test_pred == 1).mean())
            ),
            "train_survival_rate": _round_float(train_survival_rate),
            "decision": "",
        }
        rows.append(row)

    candidate_row = next(row for row in rows if row["variant"] == CANDIDATE_VARIANT)
    candidate_decision = _decision(candidate_row)
    for row in rows:
        if row["variant"] == CANDIDATE_VARIANT:
            row["decision"] = candidate_decision
    return rows


def _changed_ids(diff_rows: list[dict[str, object]], split: str) -> list[int]:
    return [int(row["PassengerId"]) for row in diff_rows if row["split"] == split]


def _id_list(values: list[object]) -> str:
    return " ".join(str(int(value)) for value in values)


def _diff_summary(diff_rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    subset = [row for row in diff_rows if row["split"] == split]
    if not subset:
        return []
    frame = pd.DataFrame(subset)
    return (
        frame.groupby(["diff_direction", "rescue_or_kill"], dropna=False)
        .agg(count=("PassengerId", "count"), PassengerIds=("PassengerId", lambda values: _id_list(list(values))))
        .reset_index()
        .sort_values(["count", "diff_direction"], ascending=[False, True])
        .to_dict("records")
    )


def _write_report(
    train: pd.DataFrame,
    metric_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    model_panel_rows: list[dict[str, object]],
) -> None:
    baseline_row = next(row for row in metric_rows if row["variant"] == BASELINE_VARIANT)
    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    decision = str(candidate_row["decision"])
    train_changed_ids = _changed_ids(diff_rows, "train_oof")
    test_changed_ids = _changed_ids(diff_rows, "test_full_fit")
    train_diff_rows = [row for row in diff_rows if row["split"] == "train_oof"]
    test_diff_rows = [row for row in diff_rows if row["split"] == "test_full_fit"]

    lines = [
        "# 18 SurnameSurvival Fold-Safe Check",
        "",
        "## Purpose",
        "",
        "This is one fixed, predeclared `SurnameSurvival` validation check on top of the unchanged `raw_tabular / GradientBoostingClassifier` baseline.",
        "",
        f"Current public leader remains frozen: `{CURRENT_PUBLIC_LEADER_SUBMISSION}`, public score `{CURRENT_PUBLIC_LEADER_SCORE}`. No submission was created.",
        "",
        "## Feature definition",
        "",
        "- `Surname` is the substring before the comma in `Name`.",
        "- `SurnameSurvival` is a smoothed surname survival rate computed from labels.",
        f"- Formula: `(survived_sum + alpha * global_survival_rate) / (surname_count + alpha)`.",
        f"- Fixed parameters: `min_count={MIN_COUNT}`, `alpha={ALPHA}`.",
        "- Fallback is the train-fold global survival rate for OOF and the full-train global survival rate for test transform.",
        "",
        "## Method boundary / anti-leakage notes",
        "",
        "- This is a target-derived train-side validation lane, not part of the closed Family/Surname structural lane.",
        "- For each CV fold, the surname map is fitted only on train-fold rows.",
        "- Validation-fold labels are never used to build validation encodings.",
        "- Validation rows receive train-fold map values; unknown surnames and train-fold counts below `min_count` receive the train-fold global survival rate.",
        "- After OOF validation, the test audit fits the encoder on full train only and transforms test from that full-train map.",
        "- Test labels are never used.",
        "- PassengerId is not used as a predictive feature, lookup key, rule, or tuning input.",
        "- No GradientBoostingClassifier hyperparameters, thresholds, gates, or public leaderboard choices are tuned here.",
        "- FamilySurnameSizeBand, mismatch-only, Fare/FareLog, Title, Age, Ticket, Deck, CabinCount, and broad CabinKnown are not reopened.",
        "",
        "## Baseline vs candidate feature sets",
        "",
        _markdown_table(
            [
                {
                    "variant": variant.variant,
                    "features": ", ".join(variant.features),
                    "purpose": variant.purpose,
                }
                for variant in VARIANTS
            ],
            ["variant", "features", "purpose"],
        ),
        "",
        "## Model panel",
        "",
        _markdown_table(model_panel_rows, MODEL_PANEL_COLUMNS),
        "",
        "## OOF / test metrics",
        "",
        _markdown_table(metric_rows, METRIC_COLUMNS),
        "",
        "## Required metric summary",
        "",
        f"- baseline OOF accuracy: `{baseline_row['oof_accuracy']}`",
        f"- candidate OOF accuracy: `{candidate_row['oof_accuracy']}`",
        f"- delta vs raw_tabular: `{candidate_row['oof_accuracy_delta_vs_raw_tabular']}`",
        f"- candidate CV mean/std: `{candidate_row['cv_mean']}` / `{candidate_row['cv_std']}`",
        f"- OOF changed rows count: `{candidate_row['oof_changed_rows']}`",
        f"- OOF 0 -> 1 count: `{candidate_row['oof_0_to_1']}`",
        f"- OOF 1 -> 0 count: `{candidate_row['oof_1_to_0']}`",
        f"- rescue / kill / net: `{candidate_row['rescue']}` / `{candidate_row['kill']}` / `{candidate_row['net']}`",
        f"- test changed rows count: `{candidate_row['test_changed_rows']}`",
        f"- test 0 -> 1 count: `{candidate_row['test_0_to_1']}`",
        f"- test 1 -> 0 count: `{candidate_row['test_1_to_0']}`",
        f"- baseline test predicted survivors and rate: `{baseline_row['test_pred_1_count']}` / `{baseline_row['test_pred_1_rate']}`",
        f"- candidate test predicted survivors and rate: `{candidate_row['test_pred_1_count']}` / `{candidate_row['test_pred_1_rate']}`",
        "",
        "## OOF changed-row audit",
        "",
        f"- changed PassengerIds: `{_id_list(train_changed_ids)}`",
        "",
        _markdown_table(_diff_summary(diff_rows, "train_oof"), ["diff_direction", "rescue_or_kill", "count", "PassengerIds"]),
        "",
        "OOF row-level audit:",
        "",
        _markdown_table(train_diff_rows, DIFF_ROW_COLUMNS),
        "",
        "## Test full-fit diff audit",
        "",
        f"- changed PassengerIds: `{_id_list(test_changed_ids)}`",
        "",
        _markdown_table(_diff_summary(diff_rows, "test_full_fit"), ["diff_direction", "rescue_or_kill", "count", "PassengerIds"]),
        "",
        "Test row-level audit:",
        "",
        _markdown_table(test_diff_rows, DIFF_ROW_COLUMNS),
        "",
        "## Decision",
        "",
        f"Decision: **{decision}**",
        "",
    ]

    if decision == "KEEP_TRAIN_SIDE":
        lines.extend(
            [
                "Reason: candidate OOF accuracy improves over `raw_tabular` and the rescue/kill/net audit is positive under the fixed train-side rule.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Reason: candidate does not satisfy the fixed train-side rule requiring OOF improvement and positive rescue/kill/net evidence.",
                "",
            ]
        )

    lines.extend(
        [
            "## Submission status",
            "",
            "No submission was created.",
            "",
            "## Output files",
            "",
            f"- report: `{_relative(REPORT_PATH)}`",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train = _add_surname(pd.read_csv(TRAIN_PATH))
    test = _add_surname(pd.read_csv(TEST_PATH))
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]

    results = {
        BASELINE_VARIANT: _evaluate_raw_tabular(train, splits, y),
        CANDIDATE_VARIANT: _evaluate_surname_survival(train, splits, y),
    }
    raw_test_pred = _fit_full_predict_raw(train, test)
    candidate_test_pred, _, test_encoded, _, test_surname_counts = _fit_full_predict_surname(train, test)

    metric_rows = _metric_rows(train, results, raw_test_pred, candidate_test_pred)
    diff_rows = _diff_rows(train, test_encoded, results, raw_test_pred, candidate_test_pred, test_surname_counts)
    model_panel_rows = _model_panel_rows()
    _write_report(train, metric_rows, diff_rows, model_panel_rows)

    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    print(f"wrote {_relative(REPORT_PATH)}")
    print(f"decision={candidate_row['decision']}")
    print(
        "oof_delta={oof_delta} rescue={rescue} kill={kill} net={net} test_changed_rows={test_changed}".format(
            oof_delta=candidate_row["oof_accuracy_delta_vs_raw_tabular"],
            rescue=candidate_row["rescue"],
            kill=candidate_row["kill"],
            net=candidate_row["net"],
            test_changed=candidate_row["test_changed_rows"],
        )
    )
    print("no submission was created")


if __name__ == "__main__":
    main()
