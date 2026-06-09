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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "18c_surname_survival_directional_overlay_check.md"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
SURNAME_FEATURE = "Surname"
SURNAME_SURVIVAL_RATE = "SurnameSurvivalRate"
SURNAME_COUNT_FROM_ENCODER = "SurnameCountFromEncoder"
SURNAME_OVERLAY_ACTIVE = "SurnameOverlayActive"
SURNAME_OVERLAY_DIRECTION = "SurnameOverlayDirection"
SURNAME_OVERLAY_REASON = "SurnameOverlayReason"
GLOBAL_SURVIVAL_RATE = "GlobalSurvivalRateFromEncoder"

MIN_COUNT = 3
ALPHA = 5.0
CURRENT_PUBLIC_LEADER_SUBMISSION = "submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"
CURRENT_PUBLIC_LEADER_SCORE = "0.79904"

BASELINE_FEATURES = list(RAW_TABULAR)

BASELINE_METRIC_COLUMNS = [
    "variant",
    "model",
    "features",
    "cv_mean",
    "cv_std",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
    "oof_accuracy",
    "test_pred_1_count",
    "test_pred_1_rate",
]

CANDIDATE_METRIC_COLUMNS = [
    "candidate",
    "rule",
    "baseline_oof_accuracy",
    "candidate_oof_accuracy",
    "delta_vs_raw_tabular",
    "cv_mean",
    "cv_std",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
    "oof_changed_rows",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "test_changed_rows",
    "test_0_to_1",
    "test_1_to_0",
    "baseline_test_pred_1_count",
    "baseline_test_pred_1_rate",
    "candidate_test_pred_1_count",
    "candidate_test_pred_1_rate",
    "inactive_changed_rows_oof",
    "inactive_changed_rows_test",
    "diagnostic_status",
]

DIRECTIONAL_COLUMNS = [
    "candidate",
    "oof_changed_rows",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "test_changed_rows",
    "test_0_to_1",
    "test_1_to_0",
    "test_survivor_count",
    "test_survivor_rate",
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

INVARIANT_COLUMNS = [
    "candidate",
    "inactive_changed_rows_oof",
    "inactive_changed_rows_test",
    "status",
]

OOF_DIFF_COLUMNS = [
    "candidate",
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
    "candidate",
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


@dataclass(frozen=True)
class SurnameEncoder:
    mapping: pd.Series
    counts: pd.Series
    global_survival_rate: float


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    rule: str
    mode: str
    positive_threshold: float | None = None
    negative_threshold: float | str | None = None


CANDIDATES = [
    CandidateSpec(
        name="surname_overlay_broad_reference_min3",
        rule="active count >= 3; rate >= 0.5 set to 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline",
        mode="broad",
        positive_threshold=0.5,
        negative_threshold="global",
    ),
    CandidateSpec(
        name="surname_overlay_upshift_only_min3",
        rule="active count >= 3; baseline_pred == 0; rate >= 0.5 set to 1; otherwise keep baseline",
        mode="upshift_only",
        positive_threshold=0.5,
    ),
    CandidateSpec(
        name="surname_overlay_downshift_only_min3",
        rule="active count >= 3; baseline_pred == 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline",
        mode="downshift_only",
        negative_threshold="global",
    ),
    CandidateSpec(
        name="surname_overlay_downshift_strict_min3",
        rule="active count >= 3; baseline_pred == 1; rate <= 0.30 set to 0; otherwise keep baseline",
        mode="downshift_only",
        negative_threshold=0.30,
    ),
    CandidateSpec(
        name="surname_overlay_upshift_strict_min3",
        rule="active count >= 3; baseline_pred == 0; rate >= 0.60 set to 1; otherwise keep baseline",
        mode="upshift_only",
        positive_threshold=0.60,
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


def _add_surname_stats_columns(frame: pd.DataFrame, encoder: SurnameEncoder) -> pd.DataFrame:
    out = frame.copy()
    counts = out[SURNAME_FEATURE].map(encoder.counts).fillna(0).astype(int)
    rates = out[SURNAME_FEATURE].map(encoder.mapping).fillna(encoder.global_survival_rate).astype(float)
    out[SURNAME_SURVIVAL_RATE] = rates
    out[SURNAME_COUNT_FROM_ENCODER] = counts
    out[GLOBAL_SURVIVAL_RATE] = encoder.global_survival_rate
    return out


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


def _evaluate_baseline_oof(
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    baseline_oof = np.full(len(train), -1, dtype=int)
    fold_scores: list[float] = []
    overlay_parts: list[pd.DataFrame] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        estimator, _, build_error = _build_estimator()
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(fold_train[BASELINE_FEATURES], y.iloc[train_idx])
        fold_pred = estimator.predict(fold_valid[BASELINE_FEATURES]).astype(int)

        encoder = _fit_surname_encoder(fold_train)
        fold_overlay = _add_surname_stats_columns(fold_valid, encoder)
        fold_overlay["Fold"] = fold_id
        fold_overlay["baseline_pred"] = fold_pred

        baseline_oof[valid_idx] = fold_pred
        fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))
        overlay_parts.append(fold_overlay)

    if (baseline_oof < 0).any():
        raise RuntimeError("baseline OOF prediction assignment incomplete")

    overlay_frame = pd.concat(overlay_parts).sort_index()
    return {
        "baseline_oof": baseline_oof,
        "fold_scores": fold_scores,
        "overlay_frame": overlay_frame,
    }


def _evaluate_baseline_test(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    estimator, _, build_error = _build_estimator()
    if estimator is None:
        raise RuntimeError(build_error)
    estimator.fit(train[BASELINE_FEATURES], train[TARGET].astype(int))
    baseline_pred = estimator.predict(test[BASELINE_FEATURES]).astype(int)

    encoder = _fit_surname_encoder(train)
    overlay_frame = _add_surname_stats_columns(test, encoder)
    overlay_frame["baseline_pred"] = baseline_pred
    return {
        "baseline_pred": baseline_pred,
        "overlay_frame": overlay_frame,
    }


def _threshold_label(value: float | str | None) -> str:
    if value is None:
        return "none"
    if value == "global":
        return "global"
    return str(value)


def _apply_candidate(
    spec: CandidateSpec,
    overlay_frame: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    baseline = overlay_frame["baseline_pred"].astype(int).to_numpy()
    rates = overlay_frame[SURNAME_SURVIVAL_RATE].astype(float).to_numpy()
    counts = overlay_frame[SURNAME_COUNT_FROM_ENCODER].astype(int).to_numpy()
    global_rates = overlay_frame[GLOBAL_SURVIVAL_RATE].astype(float).to_numpy()
    active = counts >= MIN_COUNT

    candidate = baseline.copy()
    direction = np.full(len(overlay_frame), "keep_baseline", dtype=object)
    reason = np.full(len(overlay_frame), "inactive_count_lt_min", dtype=object)
    reason[active] = "active_rule_not_triggered"

    set_to_1 = np.zeros(len(overlay_frame), dtype=bool)
    set_to_0 = np.zeros(len(overlay_frame), dtype=bool)

    if spec.mode in {"broad", "upshift_only"}:
        threshold = float(spec.positive_threshold)
        positive_mask = active & (rates >= threshold)
        if spec.mode == "upshift_only":
            positive_mask &= baseline == 0
        set_to_1 = positive_mask

    if spec.mode in {"broad", "downshift_only"}:
        if spec.negative_threshold == "global":
            negative_mask = active & (rates <= global_rates)
            negative_reason = "active_rate_le_global"
        else:
            threshold = float(spec.negative_threshold)
            negative_mask = active & (rates <= threshold)
            negative_reason = f"active_rate_le_{threshold}"
        if spec.mode == "downshift_only":
            negative_mask &= baseline == 1
        set_to_0 = negative_mask
        reason[set_to_0] = negative_reason

    candidate[set_to_1] = 1
    candidate[set_to_0] = 0
    direction[set_to_1] = "set_to_1"
    direction[set_to_0] = "set_to_0"
    if spec.positive_threshold is not None:
        reason[set_to_1] = f"active_rate_ge_{float(spec.positive_threshold)}"

    out = overlay_frame.copy()
    out["candidate"] = spec.name
    out["candidate_pred"] = candidate.astype(int)
    out[SURNAME_OVERLAY_ACTIVE] = active.astype(int)
    out[SURNAME_OVERLAY_DIRECTION] = direction
    out[SURNAME_OVERLAY_REASON] = reason
    return candidate.astype(int), out


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


def _diagnostic_status(delta: float, net: int) -> str:
    if delta > 0 and net > 0:
        return "OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    if delta < 0 or net < 0:
        return "OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    return "OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN"


def _verify_inactive_invariant(frame: pd.DataFrame, split_name: str, candidate: str) -> int:
    changed = frame["baseline_pred"].astype(int).to_numpy() != frame["candidate_pred"].astype(int).to_numpy()
    inactive = frame[SURNAME_COUNT_FROM_ENCODER].astype(int).to_numpy() < MIN_COUNT
    inactive_changed = int((changed & inactive).sum())
    if inactive_changed != 0:
        raise RuntimeError(f"{split_name} {candidate}: inactive changed rows invariant violation: {inactive_changed}")
    return inactive_changed


def _candidate_results(
    train: pd.DataFrame,
    oof_result: dict[str, object],
    test_result: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    y = train[TARGET].astype(int).to_numpy()
    baseline_oof = oof_result["baseline_oof"]
    baseline_test = test_result["baseline_pred"]
    baseline_oof_accuracy = float((baseline_oof == y).mean())
    baseline_test_count = int((baseline_test == 1).sum())
    baseline_test_rate = float((baseline_test == 1).mean())

    rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}

    for spec in CANDIDATES:
        candidate_oof, oof_frame = _apply_candidate(spec, oof_result["overlay_frame"])
        candidate_test, test_frame = _apply_candidate(spec, test_result["overlay_frame"])
        inactive_oof = _verify_inactive_invariant(oof_frame, "OOF", spec.name)
        inactive_test = _verify_inactive_invariant(test_frame, "test", spec.name)

        candidate_correct = candidate_oof == y
        baseline_correct = baseline_oof == y
        rescue = int((~baseline_correct & candidate_correct).sum())
        kill = int((baseline_correct & ~candidate_correct).sum())
        net = rescue - kill
        candidate_oof_accuracy = float((candidate_oof == y).mean())
        delta = candidate_oof_accuracy - baseline_oof_accuracy
        status = _diagnostic_status(delta, net)

        fold_scores = []
        for fold_id in sorted(oof_frame["Fold"].unique()):
            mask = oof_frame["Fold"].eq(fold_id).to_numpy()
            idx = oof_frame.index[mask].to_numpy()
            fold_scores.append(float((candidate_oof[mask] == y[idx]).mean()))

        oof_changed = candidate_oof != baseline_oof
        test_changed = candidate_test != baseline_test
        row = {
            "candidate": spec.name,
            "rule": spec.rule,
            "baseline_oof_accuracy": _round_float(baseline_oof_accuracy),
            "candidate_oof_accuracy": _round_float(candidate_oof_accuracy),
            "delta_vs_raw_tabular": _round_float(delta),
            "cv_mean": _round_float(float(np.mean(fold_scores))),
            "cv_std": _round_float(float(np.std(fold_scores))),
            "fold_1": _round_float(fold_scores[0]),
            "fold_2": _round_float(fold_scores[1]),
            "fold_3": _round_float(fold_scores[2]),
            "fold_4": _round_float(fold_scores[3]),
            "fold_5": _round_float(fold_scores[4]),
            "oof_changed_rows": int(oof_changed.sum()),
            "oof_0_to_1": int(((baseline_oof == 0) & (candidate_oof == 1)).sum()),
            "oof_1_to_0": int(((baseline_oof == 1) & (candidate_oof == 0)).sum()),
            "rescue": rescue,
            "kill": kill,
            "net": net,
            "test_changed_rows": int(test_changed.sum()),
            "test_0_to_1": int(((baseline_test == 0) & (candidate_test == 1)).sum()),
            "test_1_to_0": int(((baseline_test == 1) & (candidate_test == 0)).sum()),
            "baseline_test_pred_1_count": baseline_test_count,
            "baseline_test_pred_1_rate": _round_float(baseline_test_rate),
            "candidate_test_pred_1_count": int((candidate_test == 1).sum()),
            "candidate_test_pred_1_rate": _round_float(float((candidate_test == 1).mean())),
            "inactive_changed_rows_oof": inactive_oof,
            "inactive_changed_rows_test": inactive_test,
            "diagnostic_status": status,
        }
        rows.append(row)
        details[spec.name] = {
            "spec": spec,
            "candidate_oof": candidate_oof,
            "candidate_test": candidate_test,
            "oof_frame": oof_frame,
            "test_frame": test_frame,
            "metric_row": row,
        }

    return rows, details


def _baseline_metric_row(oof_result: dict[str, object], test_result: dict[str, object], y: pd.Series) -> dict[str, object]:
    baseline_oof = oof_result["baseline_oof"]
    baseline_test = test_result["baseline_pred"]
    return {
        "variant": BASELINE_VARIANT,
        "model": MODEL_NAME,
        "features": ", ".join(BASELINE_FEATURES),
        "cv_mean": _round_float(float(np.mean(oof_result["fold_scores"]))),
        "cv_std": _round_float(float(np.std(oof_result["fold_scores"]))),
        "fold_1": _round_float(oof_result["fold_scores"][0]),
        "fold_2": _round_float(oof_result["fold_scores"][1]),
        "fold_3": _round_float(oof_result["fold_scores"][2]),
        "fold_4": _round_float(oof_result["fold_scores"][3]),
        "fold_5": _round_float(oof_result["fold_scores"][4]),
        "oof_accuracy": _round_float(float((baseline_oof == y.to_numpy()).mean())),
        "test_pred_1_count": int((baseline_test == 1).sum()),
        "test_pred_1_rate": _round_float(float((baseline_test == 1).mean())),
    }


def _directional_rows(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in candidate_rows:
        rows.append(
            {
                "candidate": row["candidate"],
                "oof_changed_rows": row["oof_changed_rows"],
                "oof_0_to_1": row["oof_0_to_1"],
                "oof_1_to_0": row["oof_1_to_0"],
                "rescue": row["rescue"],
                "kill": row["kill"],
                "net": row["net"],
                "test_changed_rows": row["test_changed_rows"],
                "test_0_to_1": row["test_0_to_1"],
                "test_1_to_0": row["test_1_to_0"],
                "test_survivor_count": row["candidate_test_pred_1_count"],
                "test_survivor_rate": row["candidate_test_pred_1_rate"],
                "diagnostic_status": row["diagnostic_status"],
            }
        )
    return rows


def _invariant_rows(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in candidate_rows:
        passed = row["inactive_changed_rows_oof"] == 0 and row["inactive_changed_rows_test"] == 0
        rows.append(
            {
                "candidate": row["candidate"],
                "inactive_changed_rows_oof": row["inactive_changed_rows_oof"],
                "inactive_changed_rows_test": row["inactive_changed_rows_test"],
                "status": "PASS" if passed else "VIOLATION",
            }
        )
    return rows


def _oof_diff_rows(
    train: pd.DataFrame,
    oof_result: dict[str, object],
    details: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    baseline_oof = oof_result["baseline_oof"]
    rows: list[dict[str, object]] = []
    for candidate, detail in details.items():
        frame = detail["oof_frame"]
        candidate_oof = detail["candidate_oof"]
        changed_idx = np.flatnonzero(candidate_oof != baseline_oof)
        for idx in changed_idx:
            row = frame.loc[idx]
            baseline_pred = int(row["baseline_pred"])
            candidate_pred = int(row["candidate_pred"])
            rows.append(
                {
                    "candidate": candidate,
                    "PassengerId": _csv_scalar(row[ID_COLUMN]),
                    "Survived": int(y[idx]),
                    "baseline_pred": baseline_pred,
                    "candidate_pred": candidate_pred,
                    "diff_direction": _direction(baseline_pred, candidate_pred),
                    "rescue_or_kill": _train_diff_type(
                        bool(baseline_pred == y[idx]),
                        bool(candidate_pred == y[idx]),
                    ),
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


def _test_diff_rows(
    test_result: dict[str, object],
    details: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    baseline_test = test_result["baseline_pred"]
    rows: list[dict[str, object]] = []
    for candidate, detail in details.items():
        frame = detail["test_frame"]
        candidate_test = detail["candidate_test"]
        changed_idx = np.flatnonzero(candidate_test != baseline_test)
        for idx in changed_idx:
            row = frame.iloc[int(idx)]
            baseline_pred = int(row["baseline_pred"])
            candidate_pred = int(row["candidate_pred"])
            rows.append(
                {
                    "candidate": candidate,
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


def _interpretation(candidate_rows: list[dict[str, object]]) -> str:
    row_by_candidate = {str(row["candidate"]): row for row in candidate_rows}
    downshift_only = row_by_candidate["surname_overlay_downshift_only_min3"]
    downshift_strict = row_by_candidate["surname_overlay_downshift_strict_min3"]
    return "\n".join(
        [
            "Step 18C does not make a public-transfer claim.",
            "",
            "Among the predeclared `min_count=3` directional overlay diagnostics:",
            "",
            "- `broad_reference_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.",
            "- `upshift_only_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.",
            (
                "- `downshift_only_min3` is close to train-side neutral but still OOF-negative by the current criterion; "
                f"public transfer remains unknown. OOF delta `{downshift_only['delta_vs_raw_tabular']}`, net `{downshift_only['net']}`."
            ),
            (
                "- `downshift_strict_min3` is the only OOF-positive diagnostic candidate, with a small controlled OOF signal; "
                f"public transfer remains unknown. OOF delta `{downshift_strict['delta_vs_raw_tabular']}`, net `{downshift_strict['net']}`."
            ),
            "- `upshift_strict_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.",
            "",
            "These statuses do not prove public performance. They only describe train-side OOF diagnostics under the predeclared overlay rules.",
        ]
    )


def _write_report(
    baseline_row: dict[str, object],
    candidate_rows: list[dict[str, object]],
    model_panel_rows: list[dict[str, object]],
    oof_diff_rows: list[dict[str, object]],
    test_diff_rows: list[dict[str, object]],
) -> None:
    directional_rows = _directional_rows(candidate_rows)
    invariant_rows = _invariant_rows(candidate_rows)
    all_invariants_pass = all(row["status"] == "PASS" for row in invariant_rows)
    if not all_invariants_pass:
        raise RuntimeError("inactive changed rows invariant violation before report write")

    lines = [
        "# 18C SurnameSurvival Directional Overlay Check",
        "",
        "## Purpose",
        "",
        "This diagnostic checks whether a narrow surname target-derived correction exists when the baseline `raw_tabular / GradientBoostingClassifier` predictions are kept fixed and only predeclared post-model overlays are applied.",
        "",
        f"Current public leader remains frozen: `{CURRENT_PUBLIC_LEADER_SUBMISSION}`, public score `{CURRENT_PUBLIC_LEADER_SCORE}`.",
        "",
        "## Why Step 18C exists",
        "",
        "Step 18 tested `SurnameSurvival` as a normal GB feature. Step 18B moved to a controlled post-model overlay and passed the inactive-row invariant, but the broad active overlay was train-side negative. Step 18C keeps the post-model overlay approach, raises the active-count requirement, and separates broad, upshift-only, downshift-only, and strict directional rules.",
        "",
        "## Why min_count = 3",
        "",
        "`min_count = 2` is no longer used because two train-fold records are too weak a basis for a target-derived surname overlay. Step 18C uses one predeclared stricter gate, `min_count = 3`, without comparing or tuning against `min_count = 2`.",
        "",
        "## Method boundary",
        "",
        "- This is diagnostic only and not public tuning.",
        "- No candidate trains a new GB model with `SurnameSurvival` as a feature.",
        "- The baseline model and baseline features are unchanged.",
        "- `GradientBoostingClassifier` hyperparameters are unchanged.",
        "- Existing submissions and the frozen public leader are not altered.",
        "- PassengerId is not used as a rule, feature, lookup key, or tuning input.",
        "- Step 17 structural docs and closed structural lanes are not reopened.",
        "",
        "## Anti-leakage / fold-safe notes",
        "",
        "- For each CV fold, baseline GB is trained only on train-fold rows.",
        "- Validation-fold baseline predictions are true OOF predictions.",
        "- For each CV fold, the surname encoder is fitted only on train-fold rows.",
        "- Validation-fold labels are never used to build surname encoding.",
        "- Validation rows receive surname stats from the train-fold map.",
        "- Unknown surnames and train-fold surname counts below `min_count = 3` are inactive and keep baseline predictions.",
        "- OOF `global_survival_rate` is fold-specific.",
        "- Test overlay is transformed from a full-train surname map after OOF validation.",
        "- Test labels are never used.",
        "",
        "## Candidate overlay rules",
        "",
        _markdown_table(
            [
                {
                    "candidate": spec.name,
                    "rule": spec.rule,
                    "positive_threshold": _threshold_label(spec.positive_threshold),
                    "negative_threshold": _threshold_label(spec.negative_threshold),
                }
                for spec in CANDIDATES
            ],
            ["candidate", "rule", "positive_threshold", "negative_threshold"],
        ),
        "",
        "## Baseline metrics",
        "",
        _markdown_table([baseline_row], BASELINE_METRIC_COLUMNS),
        "",
        "## Candidate metrics table",
        "",
        _markdown_table(candidate_rows, CANDIDATE_METRIC_COLUMNS),
        "",
        "## Directional comparison table",
        "",
        _markdown_table(directional_rows, DIRECTIONAL_COLUMNS),
        "",
        "## Model panel",
        "",
        _markdown_table(model_panel_rows, MODEL_PANEL_COLUMNS),
        "",
        "## Inactive changed rows invariant",
        "",
        _markdown_table(invariant_rows, INVARIANT_COLUMNS),
        "",
        "Every candidate satisfies: `No inactive row changed prediction.`",
        "",
        "## OOF diff audit per candidate",
        "",
        _markdown_table(oof_diff_rows, OOF_DIFF_COLUMNS),
        "",
        "## Test diff audit per candidate",
        "",
        _markdown_table(test_diff_rows, TEST_DIFF_COLUMNS),
        "",
        "## Interpretation",
        "",
        _interpretation(candidate_rows),
        "",
        "## Diagnostic status",
        "",
        "Diagnostic status is assigned per candidate in the tables above:",
        "",
        "- `OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN` if OOF delta > 0 and net > 0.",
        "- `OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN` if OOF delta < 0 or net < 0.",
        "- `OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN` otherwise.",
        "",
        "OOF status is only a diagnostic train-side status. Public transfer remains unknown. Submission was not created. Final project decision is not automatic.",
        "",
        "Candidate wording guide:",
        "",
        "- OOF-negative candidates: This candidate is OOF-negative under the current predeclared diagnostic criterion. Public transfer remains unknown.",
        "- Near-neutral candidates: This candidate is near-neutral on train-side diagnostics. Public transfer remains unknown.",
        "- OOF-positive candidates: This candidate has a small OOF-positive diagnostic signal under the current predeclared rule. Public transfer remains unknown.",
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
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))

    oof_result = _evaluate_baseline_oof(train, splits, y)
    test_result = _evaluate_baseline_test(train, test)
    candidate_rows, details = _candidate_results(train, oof_result, test_result)
    baseline_row = _baseline_metric_row(oof_result, test_result, y)
    model_panel_rows = _model_panel_rows()
    oof_rows = _oof_diff_rows(train, oof_result, details)
    test_rows = _test_diff_rows(test_result, details)

    _write_report(baseline_row, candidate_rows, model_panel_rows, oof_rows, test_rows)

    print(f"wrote {_relative(REPORT_PATH)}")
    print("candidate comparison:")
    for row in _directional_rows(candidate_rows):
        print(
            "{candidate}: oof_delta={delta} rescue={rescue} kill={kill} net={net} "
            "test_changed={test_changed} status={status}".format(
                candidate=row["candidate"],
                delta=next(item for item in candidate_rows if item["candidate"] == row["candidate"])["delta_vs_raw_tabular"],
                rescue=row["rescue"],
                kill=row["kill"],
                net=row["net"],
                test_changed=row["test_changed_rows"],
                status=row["diagnostic_status"],
            )
        )
    print("inactive_changed_oof=0 inactive_changed_test=0 for all candidates")
    print("no submission was created")


if __name__ == "__main__":
    main()
