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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
import scripts.preprocessing as preprocessing


REPORT_PATH = REPORTS_DIR / "17_family_surname_size_band_check.md"
CSV_PATH = REPORTS_DIR / "17_family_surname_size_band_check.csv"
DIFF_ROWS_PATH = REPORTS_DIR / "17_family_surname_size_band_diff_rows.csv"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
CANDIDATE_VARIANT = "raw_family_surname_size_band"
FAMILY_SURNAME_SIZE_BAND = "FamilySurnameSizeBand"

MEANINGFUL_WORSE_TOLERANCE = 0.0025
TEST_SURVIVAL_RATE_RISK_MARGIN = 0.02
BOUNDED_TEST_DIFF_PCT = 5.0

RAW_FAMILY_SURNAME_SIZE_BAND = [
    "Sex",
    "Pclass",
    "Embarked",
    "Age",
    "Fare",
    FAMILY_SURNAME_SIZE_BAND,
]


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    candidate_id: str
    feature_set: str
    features: list[str]
    purpose: str


VARIANTS = [
    VariantSpec(
        variant=BASELINE_VARIANT,
        candidate_id="raw_tabular__GradientBoostingClassifier",
        feature_set="raw_tabular",
        features=list(RAW_TABULAR),
        purpose="baseline GB reference with raw SibSp and Parch",
    ),
    VariantSpec(
        variant=CANDIDATE_VARIANT,
        candidate_id="raw_family_surname_size_band__GradientBoostingClassifier",
        feature_set="raw_family_surname_size_band",
        features=list(RAW_FAMILY_SURNAME_SIZE_BAND),
        purpose="controlled replacement of SibSp/Parch with one plain FamilySurnameSizeBand categorical feature",
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
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
    "oof_accuracy",
    "oof_accuracy_delta_vs_raw_tabular",
    "oof_changed_rows",
    "oof_changed_pct",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "pred_1_count",
    "pred_1_rate",
    "pred_1_rate_delta_vs_raw_tabular",
    "test_changed_rows_vs_raw_tabular_full_fit",
    "test_changed_pct_vs_raw_tabular_full_fit",
    "test_0_to_1",
    "test_1_to_0",
    "test_pred_1_count",
    "test_pred_1_rate",
    "test_pred_1_rate_delta_vs_raw_tabular",
    "train_survival_rate",
    "prediction_rate_sanity_flag",
    "final_status",
]

DIFF_ROW_COLUMNS = [
    "split",
    "variant",
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
    "FamilySize",
    "Surname",
    "SurnameCount",
    "FamilySurnameSize",
    "FamilySurnameSizeBand",
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


def _size_band(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    numeric_value = int(value)
    if numeric_value == 1:
        return "alone"
    if numeric_value <= 4:
        return "small"
    if numeric_value <= 6:
        return "medium"
    return "large"


def _add_family_surname_size_band(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.concat(
        [
            train_raw.assign(Dataset="train"),
            test_raw.assign(Dataset="test"),
        ],
        ignore_index=True,
        sort=False,
    )
    manifest["FamilySize"] = manifest["SibSp"] + manifest["Parch"] + 1
    manifest["Surname"] = manifest["Name"].str.extract(r"^([^,]+),", expand=False).str.strip()
    surname_counts = manifest["Surname"].value_counts(dropna=False)
    manifest["SurnameCount"] = manifest["Surname"].map(surname_counts).astype(int)
    manifest["FamilySurnameSize"] = manifest[["FamilySize", "SurnameCount"]].max(axis=1)
    manifest["FamilySurnameSizeBand"] = manifest["FamilySurnameSize"].map(_size_band)
    return (
        manifest.loc[manifest["Dataset"].eq("train")].copy(),
        manifest.loc[manifest["Dataset"].eq("test")].copy(),
    )


def _ensure_size_band_preprocessor_support() -> None:
    if FAMILY_SURNAME_SIZE_BAND not in preprocessing.CATEGORICAL_FEATURES:
        preprocessing.CATEGORICAL_FEATURES.append(FAMILY_SURNAME_SIZE_BAND)


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPECS_BY_NAME[MODEL_NAME])


def _build_estimator(features: list[str]) -> tuple[Pipeline | None, dict[str, Any], str]:
    _ensure_size_band_preprocessor_support()
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


def _evaluate_variant(
    variant: VariantSpec,
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> dict[str, object]:
    missing_features = [feature for feature in variant.features if feature not in train.columns]
    if missing_features:
        raise ValueError(f"{variant.variant}: missing feature columns: {missing_features}")

    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)
    for train_idx, valid_idx in splits:
        estimator, _, build_error = _build_estimator(variant.features)
        if estimator is None:
            raise RuntimeError(build_error)
        estimator.fit(train[variant.features].iloc[train_idx], y.iloc[train_idx])
        fold_pred = estimator.predict(train[variant.features].iloc[valid_idx]).astype(int)
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


def _fit_full_predict(variant: VariantSpec, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
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
        "FamilySize": _csv_scalar(row["FamilySize"]),
        "Surname": _csv_scalar(row["Surname"]),
        "SurnameCount": _csv_scalar(row["SurnameCount"]),
        "FamilySurnameSize": _csv_scalar(row["FamilySurnameSize"]),
        "FamilySurnameSizeBand": _csv_scalar(row["FamilySurnameSizeBand"]),
    }


def _diff_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_oof = results[BASELINE_VARIANT]["oof"]
    candidate_oof = results[CANDIDATE_VARIANT]["oof"]
    raw_test_pred = full_fit_predictions[BASELINE_VARIANT]
    candidate_test_pred = full_fit_predictions[CANDIDATE_VARIANT]
    y = train[TARGET].astype(int).to_numpy()

    for idx in np.flatnonzero(candidate_oof != raw_oof):
        raw_pred = int(raw_oof[idx])
        candidate_pred = int(candidate_oof[idx])
        raw_correct = bool(raw_pred == y[idx])
        candidate_correct = bool(candidate_pred == y[idx])
        rows.append(
            {
                "split": "train_oof",
                "variant": CANDIDATE_VARIANT,
                "Survived": int(y[idx]),
                "raw_pred": raw_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(raw_pred, candidate_pred),
                "rescue_or_kill": _train_diff_type(raw_correct, candidate_correct),
                **_base_row_fields(train, int(idx)),
            }
        )

    for idx in np.flatnonzero(candidate_test_pred != raw_test_pred):
        raw_pred = int(raw_test_pred[idx])
        candidate_pred = int(candidate_test_pred[idx])
        rows.append(
            {
                "split": "test_full_fit",
                "variant": CANDIDATE_VARIANT,
                "Survived": "",
                "raw_pred": raw_pred,
                "candidate_pred": candidate_pred,
                "diff_direction": _direction(raw_pred, candidate_pred),
                "rescue_or_kill": "test_changed",
                **_base_row_fields(test, int(idx)),
            }
        )

    return rows


def _write_diff_rows(rows: list[dict[str, object]]) -> None:
    with DIFF_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFF_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in DIFF_ROW_COLUMNS} for row in rows)


def _final_status(candidate_row: dict[str, object]) -> str:
    oof_delta = float(candidate_row["oof_accuracy_delta_vs_raw_tabular"])
    cv_delta = float(candidate_row["cv_mean"]) - float(candidate_row["baseline_cv_mean"])
    net = int(candidate_row["net"])
    test_changed_pct = float(candidate_row["test_changed_pct_vs_raw_tabular_full_fit"])
    pred_rate_flag = str(candidate_row["prediction_rate_sanity_flag"])

    if oof_delta < 0 or cv_delta < -MEANINGFUL_WORSE_TOLERANCE or net <= 0:
        return "REJECTED_TRAIN_SIDE"
    if test_changed_pct > BOUNDED_TEST_DIFF_PCT or pred_rate_flag != "OK":
        return "HOLD_FOR_MANUAL_REVIEW"
    return "KEEP_FOR_MANUAL_REVIEW"


def _metric_rows(
    train: pd.DataFrame,
    results: dict[str, dict[str, object]],
    full_fit_predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    y = train[TARGET].astype(int).to_numpy()
    base = results[BASELINE_VARIANT]
    base_oof = base["oof"]
    base_full_fit_pred = full_fit_predictions[BASELINE_VARIANT]
    train_survival_rate = float(train[TARGET].mean())

    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        result = results[variant.variant]
        oof = result["oof"]
        full_fit_pred = full_fit_predictions[variant.variant]
        changed = oof != base_oof
        raw_correct = base_oof == y
        candidate_correct = oof == y
        rescue = int((~raw_correct & candidate_correct).sum())
        kill = int((raw_correct & ~candidate_correct).sum())
        net = rescue - kill
        test_changed = full_fit_pred != base_full_fit_pred
        test_pred_1_rate = float((full_fit_pred == 1).mean())
        prediction_rate_sanity_flag = (
            "RISK_TEST_PRED_RATE_ABOVE_TRAIN_SURVIVAL"
            if test_pred_1_rate > train_survival_rate + TEST_SURVIVAL_RATE_RISK_MARGIN
            else "OK"
        )
        row = {
            "model_name": MODEL_NAME,
            "variant": variant.variant,
            "candidate_id": variant.candidate_id,
            "feature_set": variant.feature_set,
            "features": ", ".join(variant.features),
            "cv_mean": _round_float(result["cv_mean"]),
            "cv_std": _round_float(result["cv_std"]),
            "fold_1": _round_float(result["fold_scores"][0]),
            "fold_2": _round_float(result["fold_scores"][1]),
            "fold_3": _round_float(result["fold_scores"][2]),
            "fold_4": _round_float(result["fold_scores"][3]),
            "fold_5": _round_float(result["fold_scores"][4]),
            "oof_accuracy": _round_float(result["oof_accuracy"]),
            "oof_accuracy_delta_vs_raw_tabular": _round_float(float(result["oof_accuracy"]) - float(base["oof_accuracy"])),
            "oof_changed_rows": int(changed.sum()),
            "oof_changed_pct": _round_float(float(changed.mean() * 100)),
            "oof_0_to_1": int(((base_oof == 0) & (oof == 1)).sum()),
            "oof_1_to_0": int(((base_oof == 1) & (oof == 0)).sum()),
            "rescue": rescue,
            "kill": kill,
            "net": net,
            "pred_1_count": int(result["pred_1_count"]),
            "pred_1_rate": _round_float(result["pred_1_rate"]),
            "pred_1_rate_delta_vs_raw_tabular": _round_float(float(result["pred_1_rate"]) - float(base["pred_1_rate"])),
            "test_changed_rows_vs_raw_tabular_full_fit": int(test_changed.sum()),
            "test_changed_pct_vs_raw_tabular_full_fit": _round_float(float(test_changed.mean() * 100)),
            "test_0_to_1": int(((base_full_fit_pred == 0) & (full_fit_pred == 1)).sum()),
            "test_1_to_0": int(((base_full_fit_pred == 1) & (full_fit_pred == 0)).sum()),
            "test_pred_1_count": int((full_fit_pred == 1).sum()),
            "test_pred_1_rate": _round_float(test_pred_1_rate),
            "test_pred_1_rate_delta_vs_raw_tabular": _round_float(
                test_pred_1_rate - float((base_full_fit_pred == 1).mean())
            ),
            "train_survival_rate": _round_float(train_survival_rate),
            "prediction_rate_sanity_flag": prediction_rate_sanity_flag,
            "final_status": "",
        }
        row["baseline_cv_mean"] = _round_float(base["cv_mean"])
        rows.append(row)

    candidate_row = next(row for row in rows if row["variant"] == CANDIDATE_VARIANT)
    candidate_status = _final_status(candidate_row)
    for row in rows:
        if row["variant"] == CANDIDATE_VARIANT:
            row["final_status"] = candidate_status
        row.pop("baseline_cv_mean", None)
    return rows


def _write_metrics_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)


def _changed_ids(diff_rows: list[dict[str, object]], split: str) -> list[int]:
    return [int(row["PassengerId"]) for row in diff_rows if row["split"] == split]


def _bucket_change_summary(diff_rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    subset = [row for row in diff_rows if row["split"] == split]
    if not subset:
        return []
    frame = pd.DataFrame(subset)
    group_cols = ["FamilySurnameSizeBand", "diff_direction", "rescue_or_kill"]
    return (
        frame.groupby(group_cols, dropna=False)
        .agg(count=("PassengerId", "count"), PassengerIds=("PassengerId", lambda values: _id_list(list(values))))
        .reset_index()
        .sort_values(["count", "FamilySurnameSizeBand"], ascending=[False, True])
        .to_dict("records")
    )


def _write_report(
    train: pd.DataFrame,
    metric_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    model_panel_rows: list[dict[str, object]],
) -> None:
    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    final_status = str(candidate_row["final_status"])
    train_changed_ids = _changed_ids(diff_rows, "train_oof")
    test_changed_ids = _changed_ids(diff_rows, "test_full_fit")
    train_diff_rows = [row for row in diff_rows if row["split"] == "train_oof"]
    test_diff_rows = [row for row in diff_rows if row["split"] == "test_full_fit"]

    lines = [
        "# 17B FamilySurnameSizeBand Replacement Controlled Check",
        "",
        "## Purpose",
        "",
        "This is corrected Step 17B: a controlled `GradientBoostingClassifier` check of plain `FamilySurnameSizeBand` as a replacement for raw `SibSp` and `Parch`.",
        "",
        "This is not additive on top of `SibSp/Parch`, not overlap-aware, and not target encoding. Mismatch-aware features are explicitly out of scope. No submission was created.",
        "",
        "## Feature boundary",
        "",
        "- `FamilySize = SibSp + Parch + 1`.",
        "- `Surname` is the substring before the comma in `Name`.",
        "- `SurnameCount` is counted over the combined train/test passenger manifest without `Survived`.",
        "- `FamilySurnameSize = max(FamilySize, SurnameCount)`.",
        "- `FamilySurnameSizeBand`: `alone=1`, `small=2-4`, `medium=5-6`, `large=7+`.",
        "- Sex and Pclass remain separate raw features and are not baked into the size band.",
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
        "## OOF result",
        "",
        _markdown_table(metric_rows, CSV_COLUMNS),
        "",
        "## OOF changed-row audit",
        "",
        f"- changed rows: `{candidate_row['oof_changed_rows']}`",
        f"- 0 -> 1: `{candidate_row['oof_0_to_1']}`",
        f"- 1 -> 0: `{candidate_row['oof_1_to_0']}`",
        f"- rescue / kill / net: `{candidate_row['rescue']}` / `{candidate_row['kill']}` / `{candidate_row['net']}`",
        f"- changed PassengerIds: `{_id_list(train_changed_ids)}`",
        "",
        "OOF changed rows by plain size band:",
        "",
        _markdown_table(
            _bucket_change_summary(diff_rows, "train_oof"),
            ["FamilySurnameSizeBand", "diff_direction", "rescue_or_kill", "count", "PassengerIds"],
        ),
        "",
        "OOF row-level audit:",
        "",
        _markdown_table(train_diff_rows, DIFF_ROW_COLUMNS),
        "",
        "## Test full-fit diff audit",
        "",
        f"- changed rows: `{candidate_row['test_changed_rows_vs_raw_tabular_full_fit']}`",
        f"- 0 -> 1: `{candidate_row['test_0_to_1']}`",
        f"- 1 -> 0: `{candidate_row['test_1_to_0']}`",
        f"- changed PassengerIds: `{_id_list(test_changed_ids)}`",
        f"- test pred_1_count: `{candidate_row['test_pred_1_count']}`",
        f"- test pred_1_rate: `{candidate_row['test_pred_1_rate']}`",
        f"- pred_1_rate delta vs raw_tabular: `{candidate_row['test_pred_1_rate_delta_vs_raw_tabular']}`",
        "",
        "Test changed rows by plain size band:",
        "",
        _markdown_table(
            _bucket_change_summary(diff_rows, "test_full_fit"),
            ["FamilySurnameSizeBand", "diff_direction", "rescue_or_kill", "count", "PassengerIds"],
        ),
        "",
        "Test row-level audit:",
        "",
        _markdown_table(test_diff_rows, DIFF_ROW_COLUMNS),
        "",
        "## Prediction-rate sanity",
        "",
        f"- train survival rate: `{_round_float(float(train[TARGET].mean()))}`",
        f"- raw_tabular test pred_1_rate: `{metric_rows[0]['test_pred_1_rate']}`",
        f"- candidate test pred_1_rate: `{candidate_row['test_pred_1_rate']}`",
        f"- candidate delta vs raw_tabular: `{candidate_row['test_pred_1_rate_delta_vs_raw_tabular']}`",
        f"- sanity flag: `{candidate_row['prediction_rate_sanity_flag']}`",
        "",
        "## Decision",
        "",
        f"Final status: **{final_status}**",
        "",
    ]

    if final_status == "KEEP_FOR_MANUAL_REVIEW":
        lines.extend(
            [
                "Reason: train-side OOF evidence is non-negative, rescue/kill net is positive, test diff is bounded, and prediction rate is sane.",
                "",
            ]
        )
    elif final_status == "REJECTED_TRAIN_SIDE":
        lines.extend(
            [
                "Reason: train-side evidence is negative or weak under the predeclared Step 15/16-style decision rule.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Reason: train-side or test-side evidence is mixed enough to require manual review before any promotion.",
                "",
            ]
        )

    lines.extend(
        [
            "## Validation guard",
            "",
            "Active corrected Step 17B files should contain only plain size-band logic. Overlap-aware equality flags and prefixed overlap buckets are out of scope.",
            "",
            "## Output files",
            "",
            f"- metrics CSV: `{_relative(CSV_PATH)}`",
            f"- row-level diff CSV: `{_relative(DIFF_ROWS_PATH)}`",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train, test = _add_family_surname_size_band(train_raw, test_raw)
    y = train[TARGET].astype(int)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))

    results = {
        variant.variant: _evaluate_variant(variant, train, splits, y)
        for variant in VARIANTS
    }
    full_fit_predictions = {
        variant.variant: _fit_full_predict(variant, train, test)
        for variant in VARIANTS
    }
    metric_rows = _metric_rows(train, results, full_fit_predictions)
    diff_rows = _diff_rows(train, test, results, full_fit_predictions)
    model_panel_rows = _model_panel_rows()

    _write_metrics_csv(metric_rows)
    _write_diff_rows(diff_rows)
    _write_report(train, metric_rows, diff_rows, model_panel_rows)

    candidate_row = next(row for row in metric_rows if row["variant"] == CANDIDATE_VARIANT)
    print(f"wrote {_relative(CSV_PATH)}")
    print(f"wrote {_relative(DIFF_ROWS_PATH)}")
    print(f"wrote {_relative(REPORT_PATH)}")
    print(f"final_status={candidate_row['final_status']}")
    print(
        "oof_delta={oof_delta} rescue={rescue} kill={kill} net={net} test_changed_rows={test_changed}".format(
            oof_delta=candidate_row["oof_accuracy_delta_vs_raw_tabular"],
            rescue=candidate_row["rescue"],
            kill=candidate_row["kill"],
            net=candidate_row["net"],
            test_changed=candidate_row["test_changed_rows_vs_raw_tabular_full_fit"],
        )
    )


if __name__ == "__main__":
    main()
