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


REPORT_PATH = REPORTS_DIR / "15_farelog_controlled_check.md"
CSV_PATH = REPORTS_DIR / "15_farelog_controlled_check.csv"
DIFF_ROWS_PATH = REPORTS_DIR / "15_farelog_diff_rows.csv"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
BASELINE_CANDIDATE_ID = "raw_tabular__GradientBoostingClassifier"
CURRENT_LEADER_PUBLIC_SCORE = "0.79665"
CURRENT_LEADER_SUBMISSION = "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv"

MEANINGFUL_WORSE_TOLERANCE = 0.0025
PRED_RATE_INFLATION_MARGIN = 0.03
TEST_SURVIVAL_RATE_RISK_MARGIN = 0.02

RAW_FARELOG_REPLACE_FARE = ["Sex", "Pclass", "Embarked", "Age", "SibSp", "Parch", "FareLog"]


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    candidate_id: str
    feature_set: str
    features: list[str]
    output_file: str
    purpose: str
    submission_candidate: bool


VARIANTS = [
    VariantSpec(
        variant="raw_tabular",
        candidate_id=BASELINE_CANDIDATE_ID,
        feature_set="raw_tabular",
        features=list(RAW_TABULAR),
        output_file="",
        purpose="current clean GB baseline reference; raw Fare is already included",
        submission_candidate=False,
    ),
    VariantSpec(
        variant="raw_plus_farelog",
        candidate_id="raw_plus_farelog__GradientBoostingClassifier",
        feature_set="raw_plus_farelog",
        features=[*RAW_TABULAR, "FareLog"],
        output_file="submission_15a_gb_raw_plus_farelog.csv",
        purpose="controlled check of FareLog as a transformation added to raw Fare",
        submission_candidate=True,
    ),
    VariantSpec(
        variant="raw_farelog_replace_fare",
        candidate_id="raw_farelog_replace_fare__GradientBoostingClassifier",
        feature_set="raw_farelog_replace_fare",
        features=RAW_FARELOG_REPLACE_FARE,
        output_file="submission_15c_gb_farelog_replace_fare.csv",
        purpose="controlled check of replacing raw Fare with FareLog",
        submission_candidate=True,
    ),
]

CSV_COLUMNS = [
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
    "status",
    "submission_file",
    "submission_status",
]

MODEL_PANEL_COLUMNS = [
    "variant",
    "model_class",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

SANITY_COLUMNS = ["check", "status", "detail"]

DIFF_ROW_COLUMNS = [
    "split",
    "variant",
    "PassengerId",
    "Survived",
    "raw_tabular_pred",
    "candidate_pred",
    "raw_tabular_correct",
    "candidate_correct",
    "diff_type",
    "Sex",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FareLog",
    "Embarked",
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


def _add_farelog(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["FareLog"] = np.log1p(out["Fare"])
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
    rows = []
    model, used_params, adjustment = _build_model()
    spec = MODEL_SPECS_BY_NAME[MODEL_NAME]
    package_version = baseline04._package_version(spec.version_package)

    for variant in VARIANTS:
        if model is not None and hasattr(model, "get_params"):
            actual_params: object = model.get_params(deep=False)
        elif model is not None:
            actual_params = "get_params_unavailable"
        else:
            actual_params = "model_unavailable"
        rows.append(
            {
                "variant": variant.variant,
                "model_class": MODEL_NAME,
                "package": spec.package,
                "package_version": package_version,
                "preprocessing_mode": spec.preprocessing_mode,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": "" if model is None else adjustment,
                "error": adjustment if model is None else "",
            }
        )
    return rows


def _evaluate_variant(
    variant: VariantSpec,
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    missing_features = [feature for feature in variant.features if feature not in train.columns]
    if missing_features:
        return {
            "status": "fail",
            "error": "missing feature columns: " + ", ".join(missing_features),
        }

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)

    try:
        for i, (train_idx, valid_idx) in enumerate(splits):
            estimator, _, build_error = _build_estimator(variant.features)
            if estimator is None:
                raise RuntimeError(build_error)
            estimator.fit(train[variant.features].iloc[train_idx], y.iloc[train_idx])
            fold_pred = estimator.predict(train[variant.features].iloc[valid_idx]).astype(int)
            if i < 5:
                oof[valid_idx] = fold_pred
            fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))
    except Exception as exc:
        return {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if (oof < 0).any():
        return {
            "status": "fail",
            "error": "OOF prediction assignment incomplete",
        }

    return {
        "status": "ok",
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_accuracy": float((oof == y.to_numpy()).mean()),
        "pred_1_rate": float((oof == 1).mean()),
        "pred_1_count": int((oof == 1).sum()),
        "oof": oof,
        "error": "",
    }


def _fit_full_predict(
    variant: VariantSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray | None, str]:
    missing_train = [feature for feature in variant.features if feature not in train.columns]
    missing_test = [feature for feature in variant.features if feature not in test.columns]
    if missing_train:
        return None, "missing train feature columns: " + ", ".join(missing_train)
    if missing_test:
        return None, "missing test feature columns: " + ", ".join(missing_test)

    estimator, _, build_error = _build_estimator(variant.features)
    if estimator is None:
        return None, build_error

    try:
        estimator.fit(train[variant.features], train[TARGET].astype(int))
        return estimator.predict(test[variant.features]).astype(int), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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


def _status_for_candidate(
    cv_delta: float,
    oof_delta: float,
    net: int,
    pred_1_rate_delta: float,
    calibration_flag: str,
) -> str:
    if cv_delta > 0 and oof_delta > 0 and net > 0 and pred_1_rate_delta <= PRED_RATE_INFLATION_MARGIN:
        if calibration_flag == "RISK_TEST_PRED_RATE_ABOVE_TRAIN_SURVIVAL":
            return "HOLD_FOR_MANUAL_REVIEW"
        return "KEEP_CANDIDATE"
    if cv_delta < -MEANINGFUL_WORSE_TOLERANCE or oof_delta < -MEANINGFUL_WORSE_TOLERANCE or net < 0:
        return "REJECTED_TRAIN_SIDE"
    return "HOLD_FOR_MANUAL_REVIEW"


def _comparison_rows(
    train: pd.DataFrame,
    results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray | None],
    current_leader_pred: pd.Series | None,
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    base = results[BASELINE_VARIANT]
    base_oof = base["oof"]
    base_oof_accuracy = float(base["oof_accuracy"])
    base_pred_1_rate = float(base["pred_1_rate"])
    base_full_fit_pred = full_fit_predictions[BASELINE_VARIANT]
    train_survival_rate = float(train[TARGET].mean())
    current_leader_pred_1_rate = (
        float((current_leader_pred == 1).mean()) if current_leader_pred is not None else None
    )

    rows = []
    for variant in VARIANTS:
        result = results[variant.variant]
        if result["status"] != "ok":
            rows.append(
                {
                    "model_name": MODEL_NAME,
                    "variant": variant.variant,
                    "candidate_id": variant.candidate_id,
                    "feature_set": variant.feature_set,
                    "features": ", ".join(variant.features),
                    "status": "FAIL: " + str(result["error"]),
                    "submission_file": "",
                    "submission_status": "not generated",
                }
            )
            continue

        oof = result["oof"]
        changed = oof != base_oof
        base_correct = base_oof == y
        candidate_correct = oof == y
        rescue = int((~base_correct & candidate_correct).sum())
        kill = int((base_correct & ~candidate_correct).sum())
        net = rescue - kill
        cv_delta = float(result["cv_mean"]) - float(base["cv_mean"])
        oof_delta = float(result["oof_accuracy"]) - base_oof_accuracy
        pred_1_rate = float(result["pred_1_rate"])
        pred_1_rate_delta = pred_1_rate - base_pred_1_rate

        candidate_full_fit_pred = full_fit_predictions.get(variant.variant)
        if candidate_full_fit_pred is not None and base_full_fit_pred is not None:
            test_changed = candidate_full_fit_pred != base_full_fit_pred
            test_changed_rows: object = int(test_changed.sum())
            test_changed_pct: object = _round_float(float(test_changed.mean() * 100))
            test_pred_1_count: object = int((candidate_full_fit_pred == 1).sum())
            test_pred_1_rate: object = float((candidate_full_fit_pred == 1).mean())
        else:
            test_changed_rows = "n/a"
            test_changed_pct = "n/a"
            test_pred_1_count = "n/a"
            test_pred_1_rate = None

        if test_pred_1_rate is None:
            test_pred_1_rate_delta_vs_current_leader: object = "n/a"
            calibration_flag = "TEST_PRED_UNAVAILABLE"
        else:
            test_pred_1_rate_delta_vs_current_leader = (
                "n/a"
                if current_leader_pred_1_rate is None
                else _round_float(test_pred_1_rate - current_leader_pred_1_rate)
            )
            calibration_flag = (
                "RISK_TEST_PRED_RATE_ABOVE_TRAIN_SURVIVAL"
                if test_pred_1_rate > train_survival_rate + TEST_SURVIVAL_RATE_RISK_MARGIN
                else "OK"
            )

        if variant.variant == BASELINE_VARIANT:
            status = "BASELINE_REFERENCE"
            submission_file = CURRENT_LEADER_SUBMISSION
            submission_status = "existing leader reference"
        else:
            status = _status_for_candidate(
                cv_delta,
                oof_delta,
                net,
                pred_1_rate_delta,
                calibration_flag,
            )
            submission_file = "submissions/" + variant.output_file
            submission_status = "pending decision"

        rows.append(
            {
                "model_name": MODEL_NAME,
                "variant": variant.variant,
                "candidate_id": variant.candidate_id,
                "feature_set": variant.feature_set,
                "features": ", ".join(variant.features),
                "cv_mean": _round_float(result["cv_mean"]),
                "cv_std": _round_float(result["cv_std"]),
                "oof_accuracy": _round_float(result["oof_accuracy"]),
                "oof_accuracy_delta_vs_raw_tabular": _round_float(oof_delta),
                "oof_changed_rows": int(changed.sum()),
                "oof_changed_pct": _round_float(float(changed.mean() * 100)),
                "rescue": rescue,
                "kill": kill,
                "net": net,
                "pred_1_count": int(result["pred_1_count"]),
                "pred_1_rate": _round_float(pred_1_rate),
                "pred_1_rate_delta_vs_raw_tabular": _round_float(pred_1_rate_delta),
                "test_changed_rows_vs_raw_tabular_full_fit": test_changed_rows,
                "test_changed_pct_vs_raw_tabular_full_fit": test_changed_pct,
                "test_pred_1_count": test_pred_1_count,
                "test_pred_1_rate": "n/a" if test_pred_1_rate is None else _round_float(test_pred_1_rate),
                "test_pred_1_rate_delta_vs_current_leader": test_pred_1_rate_delta_vs_current_leader,
                "train_survival_rate": _round_float(train_survival_rate),
                "current_leader_pred_1_rate": (
                    "n/a"
                    if current_leader_pred_1_rate is None
                    else _round_float(current_leader_pred_1_rate)
                ),
                "calibration_sanity_flag": calibration_flag,
                "status": status,
                "submission_file": submission_file,
                "submission_status": submission_status,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)


def _csv_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    return value


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
        "FareLog": _csv_scalar(row["FareLog"]),
        "Embarked": _csv_scalar(row["Embarked"]),
    }


def _train_diff_type(
    raw_correct: bool,
    candidate_correct: bool,
) -> str:
    if not raw_correct and candidate_correct:
        return "rescue"
    if raw_correct and not candidate_correct:
        return "kill"
    return "changed_same_correctness"


def _diff_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_oof = results[BASELINE_VARIANT]["oof"]
    raw_test_pred = full_fit_predictions[BASELINE_VARIANT]
    y = train[TARGET].astype(int).to_numpy()

    for variant in VARIANTS:
        if variant.variant == BASELINE_VARIANT:
            continue

        candidate_oof = results[variant.variant]["oof"]
        train_changed_idx = np.flatnonzero(candidate_oof != raw_oof)
        for idx in train_changed_idx:
            raw_pred = int(raw_oof[idx])
            candidate_pred = int(candidate_oof[idx])
            raw_correct = bool(raw_pred == y[idx])
            candidate_correct = bool(candidate_pred == y[idx])
            rows.append(
                {
                    "split": "train_oof",
                    "variant": variant.variant,
                    "Survived": int(y[idx]),
                    "raw_tabular_pred": raw_pred,
                    "candidate_pred": candidate_pred,
                    "raw_tabular_correct": raw_correct,
                    "candidate_correct": candidate_correct,
                    "diff_type": _train_diff_type(raw_correct, candidate_correct),
                    **_base_row_fields(train, int(idx)),
                }
            )

        candidate_test_pred = full_fit_predictions.get(variant.variant)
        if candidate_test_pred is None or raw_test_pred is None:
            continue
        test_changed_idx = np.flatnonzero(candidate_test_pred != raw_test_pred)
        for idx in test_changed_idx:
            rows.append(
                {
                    "split": "test_full_fit",
                    "variant": variant.variant,
                    "Survived": "",
                    "raw_tabular_pred": int(raw_test_pred[idx]),
                    "candidate_pred": int(candidate_test_pred[idx]),
                    "raw_tabular_correct": "",
                    "candidate_correct": "",
                    "diff_type": "test_changed",
                    **_base_row_fields(test, int(idx)),
                }
            )

    return rows


def _write_diff_rows(rows: list[dict[str, object]]) -> None:
    with DIFF_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFF_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in DIFF_ROW_COLUMNS} for row in rows)


def _write_submissions(
    rows: list[dict[str, object]],
    test: pd.DataFrame,
    full_fit_predictions: dict[str, np.ndarray | None],
) -> list[dict[str, object]]:
    submission_rows = []
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    row_by_variant = {str(row["variant"]): row for row in rows}
    for variant in VARIANTS:
        if not variant.submission_candidate:
            continue
        row = row_by_variant[variant.variant]
        output_path = SUBMISSIONS_DIR / variant.output_file
        prediction = full_fit_predictions.get(variant.variant)
        if row["status"] != "KEEP_CANDIDATE":
            row["submission_status"] = "not generated: " + str(row["status"])
            submission_rows.append(
                {
                    "variant": variant.variant,
                    "output_file": "submissions/" + variant.output_file,
                    "rows": "",
                    "pred_1_count": "",
                    "pred_1_rate": "",
                    "status": row["submission_status"],
                }
            )
            continue
        if prediction is None:
            row["submission_status"] = "not generated: full-fit prediction unavailable"
            submission_rows.append(
                {
                    "variant": variant.variant,
                    "output_file": "submissions/" + variant.output_file,
                    "rows": "",
                    "pred_1_count": "",
                    "pred_1_rate": "",
                    "status": row["submission_status"],
                }
            )
            continue

        submission = pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN].to_numpy(),
                TARGET: prediction.astype(int),
            }
        )
        submission.to_csv(output_path, index=False)
        pred_1_count = int((submission[TARGET] == 1).sum())
        pred_1_rate = float((submission[TARGET] == 1).mean())
        row["submission_status"] = "generated"
        submission_rows.append(
            {
                "variant": variant.variant,
                "output_file": _relative(output_path),
                "rows": len(submission),
                "pred_1_count": pred_1_count,
                "pred_1_rate": _round_float(pred_1_rate),
                "status": "generated",
            }
        )

    return submission_rows


def _sanity_rows(test: pd.DataFrame, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sanity = []
    forbidden_15b = SUBMISSIONS_DIR / "submission_15b_gb_raw_plus_farelog_gated.csv"
    sanity.append(
        {
            "check": "forbidden gated submission_15b absent",
            "status": "PASS" if not forbidden_15b.exists() else "FAIL",
            "detail": _relative(forbidden_15b),
        }
    )

    for row in rows:
        if row["variant"] == BASELINE_VARIANT:
            continue
        output_file = str(row["submission_file"])
        path = PROJECT_ROOT / output_file
        expected_generated = row["status"] == "KEEP_CANDIDATE"
        exists = path.exists()
        sanity.append(
            {
                "check": f"{row['variant']}: submission existence follows train-side status",
                "status": "PASS" if exists == expected_generated else "FAIL",
                "detail": f"status={row['status']}; exists={exists}; file={output_file}",
            }
        )
        if not exists:
            continue
        frame = pd.read_csv(path)
        expected_columns = [ID_COLUMN, TARGET]
        sanity.extend(
            [
                {
                    "check": f"{row['variant']}: 418 rows",
                    "status": "PASS" if len(frame) == len(test) == 418 else "FAIL",
                    "detail": f"rows={len(frame)}",
                },
                {
                    "check": f"{row['variant']}: columns exactly PassengerId,Survived",
                    "status": "PASS" if list(frame.columns) == expected_columns else "FAIL",
                    "detail": ",".join(frame.columns),
                },
                {
                    "check": f"{row['variant']}: PassengerId order matches data/test.csv",
                    "status": "PASS" if frame[ID_COLUMN].tolist() == test[ID_COLUMN].tolist() else "FAIL",
                    "detail": "order checked",
                },
                {
                    "check": f"{row['variant']}: Survived values only 0/1",
                    "status": "PASS" if set(frame[TARGET].unique()).issubset({0, 1}) else "FAIL",
                    "detail": f"values={sorted(frame[TARGET].unique().tolist())}",
                },
                {
                    "check": f"{row['variant']}: no duplicate PassengerId",
                    "status": "PASS" if not frame[ID_COLUMN].duplicated().any() else "FAIL",
                    "detail": "duplicates checked",
                },
            ]
        )
    return sanity


def _variant_rows() -> list[dict[str, object]]:
    return [
        {
            "variant": variant.variant,
            "candidate_id": variant.candidate_id,
            "features": ", ".join(variant.features),
            "submission_candidate": variant.submission_candidate,
            "output_file": "submissions/" + variant.output_file if variant.output_file else "",
            "purpose": variant.purpose,
        }
        for variant in VARIANTS
    ]


def _build_report(
    train: pd.DataFrame,
    model_panel_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    submission_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    current_leader_error: str,
) -> str:
    keep_rows = [row for row in comparison_rows if row["status"] == "KEEP_CANDIDATE"]
    hold_rows = [row for row in comparison_rows if row["status"] == "HOLD_FOR_MANUAL_REVIEW"]
    rejected_rows = [row for row in comparison_rows if row["status"] == "REJECTED_TRAIN_SIDE"]
    overall_status = "PASS" if all(row["status"] == "PASS" for row in sanity_rows) else "FAIL"
    leader_source = CURRENT_LEADER_SUBMISSION if not current_leader_error else current_leader_error

    lines = [
        "# 15 FareLog Controlled Check",
        "",
        "## Scope",
        "",
        "- controlled train-side CV/OOF check first",
        "- full `train.csv` fitting is used only after train-side status is assigned",
        "- `test.csv` is used only for inference and prediction-rate diagnostics",
        "- no public score or Kaggle leaderboard use",
        "- no `gender_submission.csv` as truth",
        "- no test target or row-level test correctness",
        "- no gated FareLog variant in this step",
        "- no Family / FamilySize / FamilySizeBucket next-step work",
        "",
        "## Current context",
        "",
        f"- current clean public leader: `raw_tabular / {MODEL_NAME}`",
        f"- current clean public score: `{CURRENT_LEADER_PUBLIC_SCORE}`",
        "- raw_tabular already contains raw `Fare`: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`",
        f"- train_survival_rate: `{_round_float(float(train[TARGET].mean()))}`",
        f"- current_leader_pred_1_rate source: `{leader_source}`",
        "",
        "## Fixed variants",
        "",
        _markdown_table(
            _variant_rows(),
            ["variant", "candidate_id", "features", "submission_candidate", "output_file", "purpose"],
        ),
        "",
        "## Excluded from this step",
        "",
        "- `submission_15b_gb_raw_plus_farelog_gated.csv` is not created.",
        "- Gated FareLog requires manual inspection and tuning after OOF/test diff review.",
        "- No gating is improvised inside this first controlled check.",
        "",
        "## Survival rule",
        "",
        "- A variant can become `KEEP_CANDIDATE` only with positive train-side evidence: CV/OOF better than raw_tabular, positive rescue/kill net, and no obvious OOF predicted-survival inflation.",
        "- If evidence is mixed, the status is `HOLD_FOR_MANUAL_REVIEW` or `REJECTED_TRAIN_SIDE`; no forced submission is created.",
        f"- Meaningful-worse tolerance: `{MEANINGFUL_WORSE_TOLERANCE}`.",
        f"- Test prediction-rate sanity risk threshold: train_survival_rate + `{TEST_SURVIVAL_RATE_RISK_MARGIN}`.",
        "- A test prediction-rate sanity risk is marked clearly and is not used as the only rejection rule.",
        "",
        "## Model panel",
        "",
        _markdown_table(model_panel_rows, MODEL_PANEL_COLUMNS),
        "",
        "## Train-side and test-diff diagnostics",
        "",
        _markdown_table(comparison_rows, CSV_COLUMNS),
        "",
        "## Candidate decision",
        "",
        f"- KEEP_CANDIDATE: `{len(keep_rows)}`",
        f"- HOLD_FOR_MANUAL_REVIEW: `{len(hold_rows)}`",
        f"- REJECTED_TRAIN_SIDE: `{len(rejected_rows)}`",
        "",
        _markdown_table(
            [row for row in comparison_rows if row["variant"] != BASELINE_VARIANT],
            CSV_COLUMNS,
        ),
        "",
        "## Conditional submission generation",
        "",
        _markdown_table(
            submission_rows,
            ["variant", "output_file", "rows", "pred_1_count", "pred_1_rate", "status"],
        ),
        "",
        "## Row-level diff diagnostics",
        "",
        "- `reports/15_farelog_diff_rows.csv` generated",
        "- compare whether raw_plus_farelog and raw_farelog_replace_fare changed the same PassengerId(s)",
        "- no decision changed by this diagnostic",
        "",
        "## Sanity checks",
        "",
        f"- overall status: `{overall_status}`",
        "",
        _markdown_table(sanity_rows, SANITY_COLUMNS),
        "",
        "## Reading boundary",
        "",
        "- This report does not accept or reject FareLog as a general feature.",
        "- It only records whether the two fixed FareLog candidates survive this controlled GB check.",
        "- No public-score tuning or micro-variants are allowed after failed transfer.",
        "- The gated FareLog branch remains separate and requires manual review before any public-facing file.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train = _add_farelog(train_raw)
    test = _add_farelog(test_raw)
    y = train[TARGET].astype(int)

    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]

    results = {
        variant.variant: _evaluate_variant(variant, train, splits, y)
        for variant in VARIANTS
    }
    if any(result["status"] != "ok" for result in results.values()):
        failed = {
            variant: result["error"]
            for variant, result in results.items()
            if result["status"] != "ok"
        }
        raise RuntimeError(f"Train-side evaluation failed: {failed}")

    full_fit_predictions = {}
    full_fit_errors = {}
    for variant in VARIANTS:
        prediction, error = _fit_full_predict(variant, train, test)
        full_fit_predictions[variant.variant] = prediction
        full_fit_errors[variant.variant] = error

    current_leader_pred, current_leader_error = _read_current_leader_predictions()
    comparison_rows = _comparison_rows(train, results, full_fit_predictions, current_leader_pred)
    diff_rows = _diff_rows(train, test, results, full_fit_predictions)
    submission_rows = _write_submissions(comparison_rows, test, full_fit_predictions)
    sanity_rows = _sanity_rows(test, comparison_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_rows)
    _write_diff_rows(diff_rows)
    REPORT_PATH.write_text(
        _build_report(
            train,
            _model_panel_rows(),
            comparison_rows,
            submission_rows,
            sanity_rows,
            current_leader_error,
        ),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"wrote {DIFF_ROWS_PATH.relative_to(PROJECT_ROOT)}")
    for row in submission_rows:
        print(f"{row['variant']}: {row['status']}")
    if any(row["status"] != "PASS" for row in sanity_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
