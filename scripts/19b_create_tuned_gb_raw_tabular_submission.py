from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "19b_tuned_gb_raw_tabular_submission_review.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_19b_tuned_gb_raw_tabular.csv"

MODEL_NAME = "GradientBoostingClassifier"
FEATURE_SET_NAME = "raw_tabular"
PREPROCESSING_MODE = "unscaled_tree"
BASELINE_FEATURES = list(RAW_TABULAR)

CURRENT_PUBLIC_LEADER_SUBMISSION = "submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"
CURRENT_PUBLIC_LEADER_SCORE = "0.79904"
DEFAULT_RAW_TABULAR_PUBLIC_SCORE = "0.79665"
PUBLIC_SCORE = "0.79186"
PUBLIC_STATUS = "REJECT_PUBLIC_TRANSFER"

SOURCE_CANDIDATE_ID = "stage2_d3_leaf5_split10_mfnone_lr0p07_n150"
SOURCE_OOF_ACCURACY = 0.843996
SOURCE_DELTA_VS_DEFAULT = 0.016835
SOURCE_RESCUE = 21
SOURCE_KILL = 6
SOURCE_NET = 15
SOURCE_TEST_CHANGED = 18
SOURCE_TEST_SURVIVORS = 139
SOURCE_TEST_RATE = 0.332536
SOURCE_STATUS = "OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"

DEFAULT_PARAMS: dict[str, object] = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": None,
    "subsample": 1.0,
    "random_state": RANDOM_STATE,
}

TUNED_PARAMS: dict[str, object] = {
    "loss": "log_loss",
    "learning_rate": 0.07,
    "n_estimators": 150,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": None,
    "subsample": 1.0,
    "ccp_alpha": 0.0,
    "random_state": RANDOM_STATE,
}

SOURCE_COLUMNS = [
    "candidate_id",
    "oof_accuracy",
    "delta_vs_default_raw_tabular_gb",
    "rescue_kill_net",
    "test_changed_rows_vs_default_full_fit",
    "test_survivors",
    "test_survivor_rate",
    "diagnostic_status",
]

PARAM_COLUMNS = ["param", "value"]
FEATURE_COLUMNS = ["feature_set", "features", "preprocessing"]
VALIDATION_COLUMNS = ["check", "status", "detail"]
PREDICTION_AUDIT_COLUMNS = ["metric", "value"]
DIFF_COLUMNS = ["comparison", "changed_rows", "0_to_1", "1_to_0"]
REFERENCE_COLUMNS = ["metric", "value"]
MODEL_PANEL_COLUMNS = [
    "model_class",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_tuned_params",
    "actual_resolved_params",
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


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _build_estimator(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(PREPROCESSING_MODE, BASELINE_FEATURES)),
            ("model", GradientBoostingClassifier(**params)),
        ]
    )


def _fit_full_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    estimator = _build_estimator(params)
    estimator.fit(train[BASELINE_FEATURES], train[TARGET].astype(int))
    return estimator.predict(test[BASELINE_FEATURES]).astype(int)


def _prediction_rate(pred: np.ndarray) -> float:
    return float((pred == 1).mean())


def _diff_counts(reference_pred: np.ndarray, candidate_pred: np.ndarray) -> dict[str, int]:
    return {
        "changed_rows": int((reference_pred != candidate_pred).sum()),
        "0_to_1": int(((reference_pred == 0) & (candidate_pred == 1)).sum()),
        "1_to_0": int(((reference_pred == 1) & (candidate_pred == 0)).sum()),
    }


def _write_submission(test: pd.DataFrame, pred: np.ndarray) -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET: pred.astype(int),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


def _validate_submission(test: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    exists = SUBMISSION_PATH.exists()
    rows.append(
        {
            "check": "file exists",
            "status": "PASS" if exists else "FAIL",
            "detail": _relative(SUBMISSION_PATH),
        }
    )
    if not exists:
        return rows

    submission = pd.read_csv(SUBMISSION_PATH)
    columns_ok = list(submission.columns) == [ID_COLUMN, TARGET]
    row_count_ok = len(submission) == len(test)
    passenger_id_ok = submission[ID_COLUMN].equals(test[ID_COLUMN])
    survived_non_null_ok = not submission[TARGET].isna().any()
    survived_values = set(submission[TARGET].dropna().astype(int).unique())
    survived_values_ok = survived_values.issubset({0, 1})
    survived_integer_ok = pd.api.types.is_integer_dtype(submission[TARGET])
    duplicate_ids_ok = not submission[ID_COLUMN].duplicated().any()

    checks = [
        ("columns exactly PassengerId,Survived", columns_ok, ", ".join(submission.columns)),
        ("row count equals test row count", row_count_ok, f"{len(submission)} / {len(test)}"),
        ("PassengerId matches test order", passenger_id_ok, "same order and values"),
        ("Survived has no NaN", survived_non_null_ok, f"nan_count={int(submission[TARGET].isna().sum())}"),
        ("Survived values are 0/1", survived_values_ok, ",".join(map(str, sorted(survived_values)))),
        ("Survived dtype is integer", survived_integer_ok, str(submission[TARGET].dtype)),
        ("PassengerId has no duplicates", duplicate_ids_ok, f"duplicate_count={int(submission[ID_COLUMN].duplicated().sum())}"),
    ]
    rows.extend(
        {
            "check": check,
            "status": "PASS" if ok else "FAIL",
            "detail": detail,
        }
        for check, ok, detail in checks
    )
    return rows


def _validate_or_raise(validation_rows: list[dict[str, object]]) -> None:
    failed = [row for row in validation_rows if row["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Submission validation failed: {failed}")


def _source_candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": SOURCE_CANDIDATE_ID,
            "oof_accuracy": _round_float(SOURCE_OOF_ACCURACY),
            "delta_vs_default_raw_tabular_gb": _round_float(SOURCE_DELTA_VS_DEFAULT),
            "rescue_kill_net": f"{SOURCE_RESCUE} / {SOURCE_KILL} / +{SOURCE_NET}",
            "test_changed_rows_vs_default_full_fit": SOURCE_TEST_CHANGED,
            "test_survivors": SOURCE_TEST_SURVIVORS,
            "test_survivor_rate": _round_float(SOURCE_TEST_RATE),
            "diagnostic_status": SOURCE_STATUS,
        }
    ]


def _param_rows(params: dict[str, object]) -> list[dict[str, object]]:
    return [{"param": key, "value": "None" if value is None else value} for key, value in params.items()]


def _feature_rows() -> list[dict[str, object]]:
    return [
        {
            "feature_set": FEATURE_SET_NAME,
            "features": ", ".join(BASELINE_FEATURES),
            "preprocessing": f"make_preprocessor(\"{PREPROCESSING_MODE}\", RAW_TABULAR)",
        }
    ]


def _prediction_audit_rows(test: pd.DataFrame, tuned_pred: np.ndarray, default_pred: np.ndarray) -> list[dict[str, object]]:
    tuned_survivors = int((tuned_pred == 1).sum())
    default_survivors = int((default_pred == 1).sum())
    return [
        {"metric": "test row count", "value": len(test)},
        {"metric": "predicted survivors count", "value": tuned_survivors},
        {"metric": "predicted survivors rate", "value": _round_float(_prediction_rate(tuned_pred))},
        {"metric": "predicted died count", "value": int((tuned_pred == 0).sum())},
        {"metric": "default raw_tabular GB predicted survivors count", "value": default_survivors},
        {"metric": "default raw_tabular GB predicted survivors rate", "value": _round_float(_prediction_rate(default_pred))},
        {"metric": "CabinKnown gate applied", "value": "NO"},
        {"metric": "SurnameSurvival applied", "value": "NO"},
        {"metric": "PassengerId rule used", "value": "NO"},
    ]


def _diff_rows(default_pred: np.ndarray, tuned_pred: np.ndarray) -> list[dict[str, object]]:
    diff = _diff_counts(default_pred, tuned_pred)
    return [
        {
            "comparison": "submission_19b_tuned_gb_raw_tabular vs default raw_tabular GB full-fit prediction",
            "changed_rows": diff["changed_rows"],
            "0_to_1": diff["0_to_1"],
            "1_to_0": diff["1_to_0"],
        }
    ]


def _frozen_leader_reference_rows(test: pd.DataFrame, tuned_pred: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frozen_path = PROJECT_ROOT / CURRENT_PUBLIC_LEADER_SUBMISSION
    reference_rows: list[dict[str, object]] = [
        {"metric": "frozen public leader path", "value": CURRENT_PUBLIC_LEADER_SUBMISSION},
        {"metric": "frozen public leader public score", "value": CURRENT_PUBLIC_LEADER_SCORE},
        {"metric": "used for 19B selection logic", "value": "NO"},
        {"metric": "modified by 19B", "value": "NO"},
    ]
    diff_rows: list[dict[str, object]] = []

    if not frozen_path.exists():
        reference_rows.append({"metric": "frozen public leader file exists", "value": "NO"})
        return reference_rows, diff_rows

    frozen = pd.read_csv(frozen_path)
    frozen_valid = (
        list(frozen.columns) == [ID_COLUMN, TARGET]
        and len(frozen) == len(test)
        and frozen[ID_COLUMN].equals(test[ID_COLUMN])
        and not frozen[TARGET].isna().any()
    )
    reference_rows.extend(
        [
            {"metric": "frozen public leader file exists", "value": "YES"},
            {"metric": "frozen public leader shape/order valid", "value": "YES" if frozen_valid else "NO"},
        ]
    )
    if not frozen_valid:
        return reference_rows, diff_rows

    frozen_pred = frozen[TARGET].astype(int).to_numpy()
    frozen_survivors = int((frozen_pred == 1).sum())
    reference_rows.extend(
        [
            {"metric": "frozen public leader predicted survivors count", "value": frozen_survivors},
            {"metric": "frozen public leader predicted survivors rate", "value": _round_float(_prediction_rate(frozen_pred))},
        ]
    )
    diff = _diff_counts(frozen_pred, tuned_pred)
    diff_rows.append(
        {
            "comparison": "submission_19b_tuned_gb_raw_tabular vs frozen public leader reference",
            "changed_rows": diff["changed_rows"],
            "0_to_1": diff["0_to_1"],
            "1_to_0": diff["1_to_0"],
        }
    )
    return reference_rows, diff_rows


def _model_panel_rows() -> list[dict[str, object]]:
    model = GradientBoostingClassifier(**TUNED_PARAMS)
    return [
        {
            "model_class": MODEL_NAME,
            "package": "scikit-learn",
            "package_version": baseline04._package_version("scikit-learn"),
            "preprocessing_mode": PREPROCESSING_MODE,
            "explicit_tuned_params": _json_dumps(TUNED_PARAMS),
            "actual_resolved_params": _json_dumps(model.get_params(deep=False)),
        }
    ]


def _report_lines(
    test: pd.DataFrame,
    tuned_pred: np.ndarray,
    default_pred: np.ndarray,
    validation_rows: list[dict[str, object]],
    frozen_reference_rows: list[dict[str, object]],
    frozen_diff_rows: list[dict[str, object]],
) -> list[str]:
    tuned_survivors = int((tuned_pred == 1).sum())
    tuned_rate = _round_float(_prediction_rate(tuned_pred))
    default_diff = _diff_counts(default_pred, tuned_pred)
    source_match_status = (
        "PASS"
        if tuned_survivors == SOURCE_TEST_SURVIVORS
        and _round_float(_prediction_rate(tuned_pred)) == _round_float(SOURCE_TEST_RATE)
        and default_diff["changed_rows"] == SOURCE_TEST_CHANGED
        else "FAIL"
    )

    lines = [
        "# 19B Tuned GB Raw Tabular Submission Review",
        "",
        "## Purpose",
        "",
        "Step 19B creates one public-check submission for the best Step 19A `raw_tabular / GradientBoostingClassifier` tuning candidate.",
        "",
        "This report records the public result for this specific standalone tuned raw_tabular GB public-check candidate.",
        "",
        "## Method boundary",
        "",
        "- Fit one tuned `GradientBoostingClassifier` on full train only.",
        "- Feature set is pure `raw_tabular`: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`.",
        "- No new tuning, threshold changes, post-processing, overlays, SurnameSurvival, CabinKnown gate, or public-score selection are used.",
        "- PassengerId is used only as the required submission row identifier and validation key; it is not used as a rule, feature, lookup key, or tuning input.",
        "- CabinKnown gate re-check is explicitly out of scope for Step 19B.",
        "",
        "## Source candidate from Step 19A",
        "",
        _markdown_table(_source_candidate_rows(), SOURCE_COLUMNS),
        "",
        f"Source candidate audit match after 19B full-fit prediction: `{source_match_status}`.",
        "",
        "## Model params",
        "",
        _markdown_table(_param_rows(TUNED_PARAMS), PARAM_COLUMNS),
        "",
        "## Feature set",
        "",
        _markdown_table(_feature_rows(), FEATURE_COLUMNS),
        "",
        "## Anti-leakage notes",
        "",
        "- Train labels are used only to fit the full-train supervised model.",
        "- Test labels are not used.",
        "- Public score is not used.",
        "- PassengerId is not used as a model feature or rule.",
        "- `make_preprocessor(\"unscaled_tree\", RAW_TABULAR)` is the only preprocessing pipeline used.",
        "",
        "## Submission file path",
        "",
        f"`{_relative(SUBMISSION_PATH)}`",
        "",
        "## Submission validation",
        "",
        _markdown_table(validation_rows, VALIDATION_COLUMNS),
        "",
        "## Prediction audit",
        "",
        _markdown_table(_prediction_audit_rows(test, tuned_pred, default_pred), PREDICTION_AUDIT_COLUMNS),
        "",
        "## Comparison vs default raw_tabular GB full-fit prediction",
        "",
        _markdown_table(_diff_rows(default_pred, tuned_pred), DIFF_COLUMNS),
        "",
        "This comparison is audit-only. It does not change the selected candidate and does not use public score.",
        "",
        "## Public result",
        "",
        f"Submission: {_relative(SUBMISSION_PATH)}",
        "",
        f"Public score: {PUBLIC_SCORE}",
        "",
        f"Status: {PUBLIC_STATUS}",
        "",
        "## Public comparison",
        "",
        f"default raw_tabular GB public score: {DEFAULT_RAW_TABULAR_PUBLIC_SCORE}",
        "",
        f"current frozen public leader public score: {CURRENT_PUBLIC_LEADER_SCORE}",
        "",
        f"19B tuned raw_tabular GB public score: {PUBLIC_SCORE}",
        "",
        "## Comparison vs frozen public leader as reference only",
        "",
        _markdown_table(frozen_reference_rows, REFERENCE_COLUMNS),
        "",
    ]
    if frozen_diff_rows:
        lines.extend(
            [
                _markdown_table(frozen_diff_rows, DIFF_COLUMNS),
                "",
            ]
        )
    else:
        lines.extend(["No prediction diff table was produced because the frozen reference file was unavailable or invalid.", ""])

    lines.extend(
        [
            "The frozen public leader is a benchmark reference only. It is not used in Step 19B selection logic and is not modified.",
            "",
            "## Final interpretation",
            "",
            "The strong train-side OOF signal from Step 19A did not transfer to public for this standalone tuned raw_tabular GB candidate.",
            "",
            "Step 19B is rejected by public transfer.",
            "",
            "This does not prove that all GB tuning is useless. It only rejects this specific standalone tuned raw_tabular GB public-check candidate.",
            "",
            "Current frozen public leader remains:",
            "",
            CURRENT_PUBLIC_LEADER_SUBMISSION,
            "",
            f"public score {CURRENT_PUBLIC_LEADER_SCORE}",
            "",
            "## Model panel",
            "",
            _markdown_table(_model_panel_rows(), MODEL_PANEL_COLUMNS),
            "",
            "## Submission status",
            "",
            f"Submission created: `{_relative(SUBMISSION_PATH)}`.",
            "",
            f"Public status: `{PUBLIC_STATUS}` with public score `{PUBLIC_SCORE}`.",
            "",
            f"Predicted survivors: `{tuned_survivors}` / `{len(test)}`; rate `{tuned_rate}`.",
            "",
            "No CabinKnown gate, SurnameSurvival, overlay, post-processing rule, threshold adjustment, or PassengerId rule was applied.",
            "",
            "## Next-step boundary",
            "",
            "After public result, the next possible separate step is: `re-check CabinKnown subgroup gate on tuned raw_tabular GB`.",
            "",
            "Do not do that in Step 19B.",
            "",
            "## Output files",
            "",
            f"- script: `scripts/19b_create_tuned_gb_raw_tabular_submission.py`",
            f"- report: `{_relative(REPORT_PATH)}`",
            f"- submission: `{_relative(SUBMISSION_PATH)}`",
            "",
        ]
    )
    return lines


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    default_pred = _fit_full_predict(train, test, DEFAULT_PARAMS)
    tuned_pred = _fit_full_predict(train, test, TUNED_PARAMS)

    _write_submission(test, tuned_pred)
    validation_rows = _validate_submission(test)
    _validate_or_raise(validation_rows)

    frozen_reference_rows, frozen_diff_rows = _frozen_leader_reference_rows(test, tuned_pred)

    REPORT_PATH.write_text(
        "\n".join(
            _report_lines(
                test,
                tuned_pred,
                default_pred,
                validation_rows,
                frozen_reference_rows,
                frozen_diff_rows,
            )
        ),
        encoding="utf-8",
    )

    tuned_survivors = int((tuned_pred == 1).sum())
    diff = _diff_counts(default_pred, tuned_pred)
    print(f"wrote {_relative(SUBMISSION_PATH)}")
    print(f"wrote {_relative(REPORT_PATH)}")
    print(f"candidate_id={SOURCE_CANDIDATE_ID}")
    print(f"predicted_survivors={tuned_survivors}")
    print(f"predicted_survivor_rate={_round_float(_prediction_rate(tuned_pred))}")
    print(
        "diff_vs_default_raw_tabular_full_fit=changed:{changed} 0_to_1:{up} 1_to_0:{down}".format(
            changed=diff["changed_rows"],
            up=diff["0_to_1"],
            down=diff["1_to_0"],
        )
    )
    print("no CabinKnown gate applied")
    print("no SurnameSurvival applied")
    print(f"public_score={PUBLIC_SCORE}")
    print(f"public_status={PUBLIC_STATUS}")
    print("submission created")


if __name__ == "__main__":
    main()
