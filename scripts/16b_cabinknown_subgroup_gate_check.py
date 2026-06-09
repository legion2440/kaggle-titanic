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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "16b_cabinknown_subgroup_gate_check.md"
CSV_PATH = REPORTS_DIR / "16b_cabinknown_subgroup_gate_check.csv"
DIFF_ROWS_PATH = REPORTS_DIR / "16b_cabinknown_subgroup_gate_diff_rows.csv"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
FULL_CABINKNOWN_VARIANT = "raw_plus_cabinknown"
SUBGROUP_VARIANT = "male_pclass1_cabin_unknown_downshift"
CURRENT_LEADER_SUBMISSION = "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv"
CURRENT_LEADER_PUBLIC_SCORE = "0.79665"
FULL_CABINKNOWN_PUBLIC_SCORE = "0.77990"
SUBGROUP_PUBLIC_SCORE = "0.79904"

EXPECTED_RAW_OOF_ACCURACY = 0.827160
EXPECTED_FULL_CABIN_OOF_ACCURACY = 0.839506
EXPECTED_FULL_CABIN_CHANGED_ROWS = 31
EXPECTED_FULL_CABIN_RESCUE = 21
EXPECTED_FULL_CABIN_KILL = 10
EXPECTED_FULL_CABIN_NET = 11
EXPECTED_FULL_CABIN_TEST_CHANGED_ROWS = 19
EXPECTED_TRAIN_IDS = [31, 35, 156, 296, 448, 794]
EXPECTED_TEST_IDS = [915, 1040, 1215]
REFERENCE_TOLERANCE = 0.000001
TEST_SURVIVAL_RATE_RISK_MARGIN = 0.02


@dataclass(frozen=True)
class FeatureVariantSpec:
    variant: str
    features: list[str]


FEATURE_VARIANTS = [
    FeatureVariantSpec(BASELINE_VARIANT, list(RAW_TABULAR)),
    FeatureVariantSpec(FULL_CABINKNOWN_VARIANT, [*RAW_TABULAR, "CabinKnown"]),
]

SUMMARY_COLUMNS = [
    "model_name",
    "variant",
    "rule",
    "oof_accuracy",
    "oof_accuracy_delta_vs_raw_tabular",
    "oof_changed_rows",
    "oof_changed_pct",
    "rescue",
    "kill",
    "net",
    "pred_1_count",
    "pred_1_rate",
    "pred_1_rate_delta_vs_raw_tabular",
    "test_changed_rows_vs_raw_tabular_full_fit",
    "test_changed_pct_vs_raw_tabular_full_fit",
    "test_pred_1_count",
    "test_pred_1_rate",
    "test_pred_1_rate_delta_vs_current_leader",
    "train_survival_rate",
    "current_leader_pred_1_rate",
    "calibration_sanity_flag",
    "train_changed_passenger_ids",
    "test_changed_passenger_ids",
    "submission_file",
    "frozen_public_score",
    "public_status",
]

REFERENCE_COLUMNS = [
    "variant",
    "oof_accuracy",
    "pred_1_count",
    "pred_1_rate",
    "changed_rows_vs_raw_tabular",
    "rescue",
    "kill",
    "net",
    "test_changed_rows_vs_raw_tabular",
    "status",
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
    "variant",
    "PassengerId",
    "Survived",
    "raw_tabular_pred",
    "raw_plus_cabinknown_pred",
    "gated_pred",
    "raw_tabular_correct",
    "gated_correct",
    "diff_type",
    "direction",
    "Sex",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Cabin",
    "CabinKnown",
    "Embarked",
]

SANITY_COLUMNS = ["check", "status", "detail"]


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


def _id_list(values: list[object]) -> str:
    return " ".join(str(int(value)) for value in values)


def _add_cabinknown(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    return out


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPECS_BY_NAME[MODEL_NAME])


def _build_estimator(features: list[str]) -> tuple[Pipeline | None, dict[str, Any], str]:
    model, used_params, adjustment = _build_model()
    if model is None:
        return None, used_params, adjustment
    return (
        Pipeline(
            steps=[
                ("preprocess", make_preprocessor("unscaled_tree", list(features))),
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


def _evaluate_feature_variant(
    variant: FeatureVariantSpec,
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    missing_features = [feature for feature in variant.features if feature not in train.columns]
    if missing_features:
        raise ValueError(f"{variant.variant}: missing feature columns: {missing_features}")

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)
    for i, (train_idx, valid_idx) in enumerate(splits):
        estimator, _, build_error = _build_estimator(variant.features)
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(train[variant.features].iloc[train_idx], y.iloc[train_idx])
        fold_pred = estimator.predict(train[variant.features].iloc[valid_idx]).astype(int)
        if i < 5:
            oof[valid_idx] = fold_pred
        fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

    if (oof < 0).any():
        raise RuntimeError(f"{variant.variant}: OOF prediction assignment incomplete")

    return {
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_accuracy": float((oof == y.to_numpy()).mean()),
        "pred_1_count": int((oof == 1).sum()),
        "pred_1_rate": float((oof == 1).mean()),
        "oof": oof,
    }


def _fit_full_predict(variant: FeatureVariantSpec, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    missing_train = [feature for feature in variant.features if feature not in train.columns]
    missing_test = [feature for feature in variant.features if feature not in test.columns]
    if missing_train:
        raise ValueError(f"{variant.variant}: missing train feature columns: {missing_train}")
    if missing_test:
        raise ValueError(f"{variant.variant}: missing test feature columns: {missing_test}")
    estimator, _, build_error = _build_estimator(variant.features)
    if estimator is None:
        raise RuntimeError(build_error)
    estimator.fit(train[variant.features], train[TARGET].astype(int))
    return estimator.predict(test[variant.features]).astype(int)


def _read_current_leader_predictions() -> tuple[pd.Series | None, str]:
    path = PROJECT_ROOT / CURRENT_LEADER_SUBMISSION
    if not path.exists():
        return None, "current leader submission not found"
    try:
        frame = pd.read_csv(path)
        if ID_COLUMN in frame.columns:
            frame = frame.sort_values(ID_COLUMN).reset_index(drop=True)
        return frame[TARGET].astype(int), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _direction(raw_pred: int, cabin_pred: int) -> str:
    if raw_pred == 1 and cabin_pred == 0:
        return "downshift_1_to_0"
    if raw_pred == 0 and cabin_pred == 1:
        return "upshift_0_to_1"
    return "unchanged"


def _train_diff_type(raw_correct: bool, gated_correct: bool) -> str:
    if not raw_correct and gated_correct:
        return "rescue"
    if raw_correct and not gated_correct:
        return "kill"
    return "changed_same_correctness"


def _subgroup_mask(frame: pd.DataFrame, raw_pred: np.ndarray, cabin_pred: np.ndarray) -> np.ndarray:
    return (
        (raw_pred == 1)
        & (cabin_pred == 0)
        & frame["Sex"].eq("male").to_numpy()
        & frame["Pclass"].eq(1).to_numpy()
        & frame["CabinKnown"].eq(0).to_numpy()
    )


def _apply_subgroup_gate(frame: pd.DataFrame, raw_pred: np.ndarray, cabin_pred: np.ndarray) -> np.ndarray:
    gated = raw_pred.copy()
    mask = _subgroup_mask(frame, raw_pred, cabin_pred)
    gated[mask] = cabin_pred[mask]
    return gated


def _reference_comparison(
    y: np.ndarray,
    raw_oof: np.ndarray,
    cabin_oof: np.ndarray,
    raw_test_pred: np.ndarray,
    cabin_test_pred: np.ndarray,
) -> dict[str, int]:
    changed = cabin_oof != raw_oof
    raw_correct = raw_oof == y
    cabin_correct = cabin_oof == y
    rescue = int((~raw_correct & cabin_correct).sum())
    kill = int((raw_correct & ~cabin_correct).sum())
    return {
        "changed_rows": int(changed.sum()),
        "rescue": rescue,
        "kill": kill,
        "net": rescue - kill,
        "test_changed_rows": int((cabin_test_pred != raw_test_pred).sum()),
    }


def _assert_step16_reference(
    feature_results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
    y: np.ndarray,
) -> None:
    raw_result = feature_results[BASELINE_VARIANT]
    cabin_result = feature_results[FULL_CABINKNOWN_VARIANT]
    reference = _reference_comparison(
        y,
        raw_result["oof"],
        cabin_result["oof"],
        full_fit_predictions[BASELINE_VARIANT],
        full_fit_predictions[FULL_CABINKNOWN_VARIANT],
    )
    checks = [
        (
            "raw_tabular OOF accuracy",
            abs(float(raw_result["oof_accuracy"]) - EXPECTED_RAW_OOF_ACCURACY) <= REFERENCE_TOLERANCE,
            float(raw_result["oof_accuracy"]),
        ),
        (
            "raw_plus_cabinknown OOF accuracy",
            abs(float(cabin_result["oof_accuracy"]) - EXPECTED_FULL_CABIN_OOF_ACCURACY) <= REFERENCE_TOLERANCE,
            float(cabin_result["oof_accuracy"]),
        ),
        ("OOF changed rows", reference["changed_rows"] == EXPECTED_FULL_CABIN_CHANGED_ROWS, reference["changed_rows"]),
        ("rescue", reference["rescue"] == EXPECTED_FULL_CABIN_RESCUE, reference["rescue"]),
        ("kill", reference["kill"] == EXPECTED_FULL_CABIN_KILL, reference["kill"]),
        ("net", reference["net"] == EXPECTED_FULL_CABIN_NET, reference["net"]),
        (
            "test changed rows",
            reference["test_changed_rows"] == EXPECTED_FULL_CABIN_TEST_CHANGED_ROWS,
            reference["test_changed_rows"],
        ),
    ]
    failed = [f"{name}: observed={observed}" for name, ok, observed in checks if not ok]
    if failed:
        raise RuntimeError("Step 16 reference reproduction failed: " + "; ".join(failed))


def _assert_expected_ids(train: pd.DataFrame, test: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    train_ids = train.loc[train_mask, ID_COLUMN].astype(int).tolist()
    test_ids = test.loc[test_mask, ID_COLUMN].astype(int).tolist()
    if train_ids != EXPECTED_TRAIN_IDS:
        raise RuntimeError(f"OOF subgroup PassengerId mismatch: expected={EXPECTED_TRAIN_IDS}; observed={train_ids}")
    if test_ids != EXPECTED_TEST_IDS:
        raise RuntimeError(f"Test subgroup PassengerId mismatch: expected={EXPECTED_TEST_IDS}; observed={test_ids}")


def _reference_rows(
    feature_results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
    y: np.ndarray,
) -> list[dict[str, object]]:
    raw_oof = feature_results[BASELINE_VARIANT]["oof"]
    raw_test = full_fit_predictions[BASELINE_VARIANT]
    rows = []
    for variant in FEATURE_VARIANTS:
        result = feature_results[variant.variant]
        oof = result["oof"]
        changed = oof != raw_oof
        if variant.variant == BASELINE_VARIANT:
            rescue = kill = net = test_changed = 0
        else:
            correct = oof == y
            raw_correct = raw_oof == y
            rescue = int((~raw_correct & correct).sum())
            kill = int((raw_correct & ~correct).sum())
            net = rescue - kill
            test_changed = int((full_fit_predictions[variant.variant] != raw_test).sum())
        rows.append(
            {
                "variant": variant.variant,
                "oof_accuracy": _round_float(result["oof_accuracy"]),
                "pred_1_count": int(result["pred_1_count"]),
                "pred_1_rate": _round_float(result["pred_1_rate"]),
                "changed_rows_vs_raw_tabular": int(changed.sum()),
                "rescue": rescue,
                "kill": kill,
                "net": net,
                "test_changed_rows_vs_raw_tabular": test_changed,
                "status": "REFERENCE_REPRODUCED",
            }
        )
    return rows


def _summary_row(
    train: pd.DataFrame,
    feature_results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
    gated_oof: np.ndarray,
    gated_test: np.ndarray,
    current_leader_pred: pd.Series | None,
) -> dict[str, object]:
    y = train[TARGET].astype(int).to_numpy()
    raw_oof = feature_results[BASELINE_VARIANT]["oof"]
    raw_oof_accuracy = float(feature_results[BASELINE_VARIANT]["oof_accuracy"])
    raw_pred_1_rate = float(feature_results[BASELINE_VARIANT]["pred_1_rate"])
    raw_test = full_fit_predictions[BASELINE_VARIANT]
    raw_correct = raw_oof == y
    gated_correct = gated_oof == y
    changed = gated_oof != raw_oof
    rescue = int((~raw_correct & gated_correct).sum())
    kill = int((raw_correct & ~gated_correct).sum())
    oof_accuracy = float(gated_correct.mean())
    pred_1_rate = float((gated_oof == 1).mean())
    test_changed = gated_test != raw_test
    test_pred_1_rate = float((gated_test == 1).mean())
    train_survival_rate = float(train[TARGET].mean())
    current_leader_pred_1_rate = (
        float((current_leader_pred == 1).mean()) if current_leader_pred is not None else None
    )
    calibration_flag = (
        "RISK_TEST_PRED_RATE_ABOVE_TRAIN_SURVIVAL"
        if test_pred_1_rate > train_survival_rate + TEST_SURVIVAL_RATE_RISK_MARGIN
        else "OK"
    )
    return {
        "model_name": MODEL_NAME,
        "variant": SUBGROUP_VARIANT,
        "rule": 'raw_tabular_pred == 1 and raw_plus_cabinknown_pred == 0 and Sex == "male" and Pclass == 1 and CabinKnown == 0',
        "oof_accuracy": _round_float(oof_accuracy),
        "oof_accuracy_delta_vs_raw_tabular": _round_float(oof_accuracy - raw_oof_accuracy),
        "oof_changed_rows": int(changed.sum()),
        "oof_changed_pct": _round_float(float(changed.mean() * 100)),
        "rescue": rescue,
        "kill": kill,
        "net": rescue - kill,
        "pred_1_count": int((gated_oof == 1).sum()),
        "pred_1_rate": _round_float(pred_1_rate),
        "pred_1_rate_delta_vs_raw_tabular": _round_float(pred_1_rate - raw_pred_1_rate),
        "test_changed_rows_vs_raw_tabular_full_fit": int(test_changed.sum()),
        "test_changed_pct_vs_raw_tabular_full_fit": _round_float(float(test_changed.mean() * 100)),
        "test_pred_1_count": int((gated_test == 1).sum()),
        "test_pred_1_rate": _round_float(test_pred_1_rate),
        "test_pred_1_rate_delta_vs_current_leader": (
            "n/a"
            if current_leader_pred_1_rate is None
            else _round_float(test_pred_1_rate - current_leader_pred_1_rate)
        ),
        "train_survival_rate": _round_float(train_survival_rate),
        "current_leader_pred_1_rate": (
            "n/a" if current_leader_pred_1_rate is None else _round_float(current_leader_pred_1_rate)
        ),
        "calibration_sanity_flag": calibration_flag,
        "train_changed_passenger_ids": _id_list(EXPECTED_TRAIN_IDS),
        "test_changed_passenger_ids": _id_list(EXPECTED_TEST_IDS),
        "submission_file": _relative(SUBMISSION_PATH),
        "frozen_public_score": SUBGROUP_PUBLIC_SCORE,
        "public_status": "CURRENT_PUBLIC_LEADER",
    }


def _base_row_fields(frame: pd.DataFrame, idx: int) -> dict[str, object]:
    row = frame.iloc[idx]
    return {
        "PassengerId": _csv_scalar(row[ID_COLUMN]),
        "Sex": _csv_scalar(row["Sex"]),
        "Pclass": _csv_scalar(row["Pclass"]),
        "Age": _csv_scalar(row["Age"]),
        "SibSp": _csv_scalar(row["SibSp"]),
        "Parch": _csv_scalar(row["Parch"]),
        "Fare": _csv_scalar(row["Fare"]),
        "Cabin": _csv_scalar(row["Cabin"]),
        "CabinKnown": _csv_scalar(row["CabinKnown"]),
        "Embarked": _csv_scalar(row["Embarked"]),
    }


def _diff_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
    gated_oof: np.ndarray,
    gated_test: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = train[TARGET].astype(int).to_numpy()
    raw_oof = feature_results[BASELINE_VARIANT]["oof"]
    cabin_oof = feature_results[FULL_CABINKNOWN_VARIANT]["oof"]
    raw_test = full_fit_predictions[BASELINE_VARIANT]
    cabin_test = full_fit_predictions[FULL_CABINKNOWN_VARIANT]

    for idx in np.flatnonzero(gated_oof != raw_oof):
        raw_pred = int(raw_oof[idx])
        cabin_pred = int(cabin_oof[idx])
        gated_pred = int(gated_oof[idx])
        raw_correct = bool(raw_pred == y[idx])
        gated_correct = bool(gated_pred == y[idx])
        rows.append(
            {
                "split": "train_oof",
                "variant": SUBGROUP_VARIANT,
                "Survived": int(y[idx]),
                "raw_tabular_pred": raw_pred,
                "raw_plus_cabinknown_pred": cabin_pred,
                "gated_pred": gated_pred,
                "raw_tabular_correct": raw_correct,
                "gated_correct": gated_correct,
                "diff_type": _train_diff_type(raw_correct, gated_correct),
                "direction": _direction(raw_pred, cabin_pred),
                **_base_row_fields(train, int(idx)),
            }
        )

    for idx in np.flatnonzero(gated_test != raw_test):
        raw_pred = int(raw_test[idx])
        cabin_pred = int(cabin_test[idx])
        rows.append(
            {
                "split": "test_full_fit",
                "variant": SUBGROUP_VARIANT,
                "Survived": "",
                "raw_tabular_pred": raw_pred,
                "raw_plus_cabinknown_pred": cabin_pred,
                "gated_pred": int(gated_test[idx]),
                "raw_tabular_correct": "",
                "gated_correct": "",
                "diff_type": "test_changed",
                "direction": _direction(raw_pred, cabin_pred),
                **_base_row_fields(test, int(idx)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _write_submission(test: pd.DataFrame, gated_test: np.ndarray) -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET: gated_test.astype(int),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


def _sanity_rows(test: pd.DataFrame, raw_test: np.ndarray, gated_test: np.ndarray) -> list[dict[str, object]]:
    rows = []
    frame = pd.read_csv(SUBMISSION_PATH)
    changed_ids = test.loc[gated_test != raw_test, ID_COLUMN].astype(int).tolist()
    checks = [
        ("submission exists", SUBMISSION_PATH.exists(), _relative(SUBMISSION_PATH)),
        ("418 rows", len(frame) == len(test) == 418, f"rows={len(frame)}"),
        ("columns exactly PassengerId,Survived", list(frame.columns) == [ID_COLUMN, TARGET], ",".join(frame.columns)),
        ("PassengerId order matches data/test.csv", frame[ID_COLUMN].tolist() == test[ID_COLUMN].tolist(), "order checked"),
        ("Survived values only 0/1", set(frame[TARGET].unique()).issubset({0, 1}), f"values={sorted(frame[TARGET].unique().tolist())}"),
        ("no duplicate PassengerId", not frame[ID_COLUMN].duplicated().any(), "duplicates checked"),
        ("changed PassengerIds exactly expected", changed_ids == EXPECTED_TEST_IDS, f"changed={changed_ids}"),
    ]
    for check, ok, detail in checks:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})
    return rows


def _build_report(
    train: pd.DataFrame,
    reference_rows: list[dict[str, object]],
    model_panel_rows: list[dict[str, object]],
    summary_row: dict[str, object],
    diff_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    current_leader_error: str,
) -> str:
    train_diff_rows = [row for row in diff_rows if row["split"] == "train_oof"]
    test_diff_rows = [row for row in diff_rows if row["split"] == "test_full_fit"]
    leader_source = CURRENT_LEADER_SUBMISSION if not current_leader_error else current_leader_error
    overall_status = "PASS" if all(row["status"] == "PASS" for row in sanity_rows) else "FAIL"
    return "\n".join(
        [
            "# 16b CabinKnown Subgroup Gate Check",
            "",
            "## Scope",
            "",
            "- Frozen subgroup gate for the pre-identified CabinKnown diff subgroup.",
            "- Model is `GradientBoostingClassifier` only.",
            "- No test labels, PassengerId corrections, post-public tuning, weight/blend branch, or additional subgroup variants.",
            "- Deck, TicketPrefix, Ticket, Family, FareLog, Age, and Title are not changed or reopened.",
            "",
            "## EDA / OOF rationale",
            "",
            "- Full `raw_plus_cabinknown` survived train-side but failed public transfer.",
            f"- Step 16 full CabinKnown public score: `{FULL_CABINKNOWN_PUBLIC_SCORE}`.",
            "- Direction-only Step 16b downshift/upshift outputs are diagnostic only and are not final subgroup gates.",
            "- The selected subgroup was identified before this frozen checkpoint from OOF/group diagnostics:",
            "  `raw_tabular_pred == 1`, `raw_plus_cabinknown_pred == 0`, `Sex == \"male\"`, `Pclass == 1`, `CabinKnown == 0`.",
            "- This is a model-diff subgroup, not a raw PassengerId correction.",
            f"- Current leader source for prediction-rate comparison: `{leader_source}`.",
            "",
            "## Step 16 reference reproduction",
            "",
            "- The script stops before submission generation if these reference metrics do not match.",
            "",
            _markdown_table(reference_rows, REFERENCE_COLUMNS),
            "",
            "## Subgroup rule",
            "",
            "```python",
            "gated_pred = raw_tabular_pred",
            "if (",
            "    raw_tabular_pred == 1",
            "    and raw_plus_cabinknown_pred == 0",
            "    and Sex == \"male\"",
            "    and Pclass == 1",
            "    and CabinKnown == 0",
            "):",
            "    gated_pred = raw_plus_cabinknown_pred",
            "```",
            "",
            "## Model panel",
            "",
            _markdown_table(model_panel_rows, MODEL_PANEL_COLUMNS),
            "",
            "## OOF table",
            "",
            _markdown_table([summary_row], SUMMARY_COLUMNS),
            "",
            "## Row-level OOF diff table",
            "",
            _markdown_table(train_diff_rows, DIFF_ROW_COLUMNS),
            "",
            "## Test diff table",
            "",
            _markdown_table(test_diff_rows, DIFF_ROW_COLUMNS),
            "",
            "## Submission sanity checks",
            "",
            f"- overall status: `{overall_status}`",
            "",
            _markdown_table(sanity_rows, SANITY_COLUMNS),
            "",
            "## Frozen public checkpoint",
            "",
            f"- submission: `{SUBMISSION_PATH.name}`",
            f"- public score: `{SUBGROUP_PUBLIC_SCORE}`",
            "- status: `CURRENT_PUBLIC_LEADER`",
            "- Public score is recorded only after this subgroup was already identified by OOF/group diagnostics.",
            "",
            "## Reading boundary",
            "",
            "- No further subgroup gates are introduced here.",
            "- No public tuning from this result.",
            "- Weight/blend branch, if pursued, is separate and predeclared.",
            "- Direction-only downshift/upshift remains diagnostic/deprecated, not current candidate logic.",
        ]
    ) + "\n"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train = _add_cabinknown(train_raw)
    test = _add_cabinknown(test_raw)
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]

    feature_results = {
        variant.variant: _evaluate_feature_variant(variant, train, splits, y)
        for variant in FEATURE_VARIANTS
    }
    full_fit_predictions = {
        variant.variant: _fit_full_predict(variant, train, test)
        for variant in FEATURE_VARIANTS
    }
    _assert_step16_reference(feature_results, full_fit_predictions, y.to_numpy())

    raw_oof = feature_results[BASELINE_VARIANT]["oof"]
    cabin_oof = feature_results[FULL_CABINKNOWN_VARIANT]["oof"]
    raw_test = full_fit_predictions[BASELINE_VARIANT]
    cabin_test = full_fit_predictions[FULL_CABINKNOWN_VARIANT]
    train_mask = _subgroup_mask(train, raw_oof, cabin_oof)
    test_mask = _subgroup_mask(test, raw_test, cabin_test)
    _assert_expected_ids(train, test, train_mask, test_mask)

    gated_oof = _apply_subgroup_gate(train, raw_oof, cabin_oof)
    gated_test = _apply_subgroup_gate(test, raw_test, cabin_test)
    current_leader_pred, current_leader_error = _read_current_leader_predictions()
    reference_rows = _reference_rows(feature_results, full_fit_predictions, y.to_numpy())
    summary_row = _summary_row(train, feature_results, full_fit_predictions, gated_oof, gated_test, current_leader_pred)
    diff_rows = _diff_rows(train, test, feature_results, full_fit_predictions, gated_oof, gated_test)

    _write_submission(test, gated_test)
    sanity_rows = _sanity_rows(test, raw_test, gated_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(CSV_PATH, [summary_row], SUMMARY_COLUMNS)
    _write_csv(DIFF_ROWS_PATH, diff_rows, DIFF_ROW_COLUMNS)
    REPORT_PATH.write_text(
        _build_report(
            train,
            reference_rows,
            _model_panel_rows(),
            summary_row,
            diff_rows,
            sanity_rows,
            current_leader_error,
        ),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {DIFF_ROWS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {SUBMISSION_PATH.relative_to(PROJECT_ROOT)}")
    if any(row["status"] != "PASS" for row in sanity_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
