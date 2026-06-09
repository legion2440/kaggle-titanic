from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "12_agebucket_frozen_checkpoint.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

FEATURE_SET_NAME = "raw_no_age_no_sex_plus_agebucket_v1"
FEATURES = ["Pclass", "Embarked", "SibSp", "Parch", "Fare", "AgeBucket"]
BASELINE_FILE = "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv"
BASELINE_CANDIDATE_ID = "raw_tabular__GradientBoostingClassifier"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    preprocessing_mode: str
    output_file: str
    reason: str


CANDIDATES = [
    Candidate(
        "agebucket_v1__SVC",
        "SVC",
        "scaled_linear",
        "submission_12a_svc_agebucket_v1.csv",
        "train-side KEEP_CANDIDATE in step 11",
    ),
    Candidate(
        "agebucket_v1__GradientBoostingClassifier",
        "GradientBoostingClassifier",
        "unscaled_tree",
        "submission_12b_gb_agebucket_v1.csv",
        "primary-lane transfer probe by explicit review override",
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
MODEL_SPECS = baseline04.MODEL_SPECS


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _model_spec_by_name() -> dict[str, object]:
    return {spec.model: spec for spec in MODEL_SPECS}


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _agebucket_v1_row(row: pd.Series) -> str:
    if pd.isna(row["Age"]) and row["Sex"] == "female":
        return "AgeMissingFemale"
    if pd.isna(row["Age"]) and row["Sex"] == "male":
        return "AgeMissingMale"
    if row["Sex"] == "female" and row["Age"] < 14:
        return "ChildFemale"
    if row["Sex"] == "male" and row["Age"] < 14:
        return "ChildMale"
    if row["Sex"] == "female":
        return "AdultFemale"
    return "AdultMale"


def _add_agebucket_v1(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["AgeBucket"] = out.apply(_agebucket_v1_row, axis=1)
    return out


def _model_panel_rows() -> list[dict[str, object]]:
    specs = _model_spec_by_name()
    rows = []

    for candidate in CANDIDATES:
        spec = specs[candidate.model]
        model, used_params, error_or_adjustment = baseline04._build_model(spec)
        if model is not None and hasattr(model, "get_params"):
            actual_params = model.get_params(deep=False)
        elif model is not None:
            actual_params = "get_params_unavailable"
        else:
            actual_params = "model_unavailable"

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "model": candidate.model,
                "package": spec.package,
                "package_version": baseline04._package_version(spec.version_package),
                "preprocessing_mode": candidate.preprocessing_mode,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": "" if model is not None else "model unavailable",
                "error": "" if model is not None else error_or_adjustment,
            }
        )
    return rows


def _fit_predict_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, object], pd.Series | None]:
    specs = _model_spec_by_name()
    spec = specs[candidate.model]
    output_path = SUBMISSIONS_DIR / candidate.output_file
    base_row = {
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "output_file": _relative(output_path),
        "rows": "",
        "pred_0_count": "",
        "pred_1_count": "",
        "pred_1_rate": "",
    }

    try:
        missing_train = [feature for feature in FEATURES if feature not in train.columns]
        missing_test = [feature for feature in FEATURES if feature not in test.columns]
        if TARGET not in train.columns:
            raise ValueError(f"missing target column: {TARGET}")
        if ID_COLUMN not in test.columns:
            raise ValueError(f"missing test id column: {ID_COLUMN}")
        if missing_train:
            raise ValueError("missing train feature columns: " + ", ".join(missing_train))
        if missing_test:
            raise ValueError("missing test feature columns: " + ", ".join(missing_test))

        model, _, build_error = baseline04._build_model(spec)
        if model is None:
            raise RuntimeError(build_error)

        pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(candidate.preprocessing_mode, list(FEATURES))),
                ("model", model),
            ]
        )
        pipeline.fit(train[FEATURES], train[TARGET])
        predictions = pd.Series(pipeline.predict(test[FEATURES]), name=TARGET).astype(int)
        submission = pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN].to_numpy(),
                TARGET: predictions.to_numpy(),
            }
        )
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)

        pred_counts = predictions.value_counts().to_dict()
        return (
            {
                **base_row,
                "rows": len(submission),
                "pred_0_count": int(pred_counts.get(0, 0)),
                "pred_1_count": int(pred_counts.get(1, 0)),
                "pred_1_rate": _round_float(float((predictions == 1).mean())),
                "status": "PASS",
            },
            predictions,
        )
    except Exception as exc:
        return ({**base_row, "status": f"FAIL: {type(exc).__name__}: {exc}"}, None)


def _baseline_diff_rows(predictions: dict[str, pd.Series]) -> list[dict[str, object]]:
    rows = []
    baseline_path = PROJECT_ROOT / BASELINE_FILE

    for candidate in CANDIDATES:
        pred = predictions.get(candidate.candidate_id)
        base = {
            "baseline_candidate_id": BASELINE_CANDIDATE_ID,
            "candidate_id": candidate.candidate_id,
            "baseline_file": BASELINE_FILE,
            "changed_predictions": "n/a",
            "changed_pct": "n/a",
            "baseline_pred_1_count": "n/a",
            "candidate_pred_1_count": "n/a",
            "baseline_pred_1_rate": "n/a",
            "candidate_pred_1_rate": "n/a",
            "delta_pred_1_rate": "n/a",
            "status": "n/a",
        }

        if pred is None:
            rows.append({**base, "status": "candidate prediction unavailable"})
            continue
        if not baseline_path.exists():
            rows.append({**base, "status": "baseline file not found"})
            continue

        baseline_frame = pd.read_csv(baseline_path)
        baseline_pred = baseline_frame[TARGET].astype(int)
        if len(baseline_pred) != len(pred):
            rows.append({**base, "status": "row count mismatch"})
            continue

        changed = baseline_pred.to_numpy() != pred.to_numpy()
        baseline_pred_1_rate = float((baseline_pred == 1).mean())
        candidate_pred_1_rate = float((pred == 1).mean())
        rows.append(
            {
                **base,
                "changed_predictions": int(changed.sum()),
                "changed_pct": _round_float(float(changed.mean() * 100)),
                "baseline_pred_1_count": int((baseline_pred == 1).sum()),
                "candidate_pred_1_count": int((pred == 1).sum()),
                "baseline_pred_1_rate": _round_float(baseline_pred_1_rate),
                "candidate_pred_1_rate": _round_float(candidate_pred_1_rate),
                "delta_pred_1_rate": _round_float(candidate_pred_1_rate - baseline_pred_1_rate),
                "status": "PASS",
            }
        )
    return rows


def _submission_sanity_rows(generated_rows: list[dict[str, object]], test: pd.DataFrame) -> tuple[list[dict[str, object]], bool]:
    rows = []
    all_passed = True
    expected_order = test[ID_COLUMN].tolist() if ID_COLUMN in test.columns else []

    def add_check(check: str, passed: bool, detail: str) -> None:
        nonlocal all_passed
        all_passed = all_passed and passed
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    passed_files = [row for row in generated_rows if row["status"] == "PASS"]
    add_check("exactly 2 submission files generated", len(passed_files) == 2, f"{len(passed_files)} of 2 files")

    for row in generated_rows:
        output_path = PROJECT_ROOT / str(row["output_file"])
        label = str(row["candidate_id"])
        if row["status"] != "PASS" or not output_path.exists():
            add_check(label, False, f"missing or failed file: {row['status']}")
            continue

        frame = pd.read_csv(output_path)
        columns_ok = list(frame.columns) == [ID_COLUMN, TARGET]
        rows_ok = len(frame) == len(test) == 418
        order_ok = columns_ok and frame[ID_COLUMN].tolist() == expected_order
        values = set(frame[TARGET].dropna().astype(int).unique().tolist()) if TARGET in frame.columns else set()
        values_ok = values.issubset({0, 1}) and not frame[TARGET].isna().any()
        duplicates_ok = not frame[ID_COLUMN].duplicated().any() if ID_COLUMN in frame.columns else False

        add_check(f"{label}: 418 rows", rows_ok, f"rows={len(frame)}")
        add_check(f"{label}: columns exactly PassengerId,Survived", columns_ok, ",".join(frame.columns))
        add_check(f"{label}: PassengerId order matches data/test.csv", order_ok, "order checked")
        add_check(f"{label}: Survived values only 0/1", values_ok, f"values={sorted(values)}")
        add_check(f"{label}: no duplicate PassengerId", duplicates_ok, "duplicates checked")

    return rows, all_passed


def _public_score_rows() -> list[dict[str, object]]:
    return [
        {"output_file": "submission_12a_svc_agebucket_v1.csv", "public_score": "TBD"},
        {"output_file": "submission_12b_gb_agebucket_v1.csv", "public_score": "TBD"},
    ]


def _build_report(
    generated_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# 12 AgeBucket Frozen Checkpoint",
        "",
        "## Scope",
        "",
        "- frozen checkpoint for AgeBucket v1 transfer probe",
        "- candidates are fixed before any public score",
        "- full `train.csv` fitting is allowed",
        "- `test.csv` is used only for inference",
        "- no submission logic uses public score",
        "",
        "## Method boundary",
        "",
        "- This is a frozen checkpoint, not feature acceptance.",
        "- No post-score tuning.",
        "- No new features.",
        "- AgeBucket mapping is unchanged from step 11.",
        "- Broad `Title` remains closed.",
        "- No Master fallback, Old buckets, Mrs/Miss, Surname, target-derived family/group survival, or PassengerId corrections.",
        "- `gender_submission.csv` is not used as truth.",
        "- Test target is not used.",
        "",
        "## Fixed candidate list",
        "",
        _markdown_table(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    "model": candidate.model,
                    "feature_set": FEATURE_SET_NAME,
                    "features": ", ".join(FEATURES),
                    "preprocessing_mode": candidate.preprocessing_mode,
                    "output_file": f"submissions/{candidate.output_file}",
                    "reason": candidate.reason,
                }
                for candidate in CANDIDATES
            ],
            ["candidate_id", "model", "feature_set", "features", "preprocessing_mode", "output_file", "reason"],
        ),
        "",
        "## Why checkpoint is allowed",
        "",
        "- SVC is train-side `KEEP_CANDIDATE` in `reports/11_agebucket_feature_check.md`.",
        "- GradientBoostingClassifier is a primary-lane transfer probe by explicit review override.",
        "- The candidate list is fixed before public score and must not be changed post-score.",
        "",
        "## Why CatBoost is excluded",
        "",
        "- CatBoost was `DEFERRED` in step 11.",
        "- This checkpoint is intentionally limited to SVC and the primary GB transfer probe.",
        "- CatBoost is not included to avoid broadening the transfer probe after review.",
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
        "## Model panel used",
        "",
        _markdown_table(
            panel_rows,
            [
                "candidate_id",
                "model",
                "package",
                "package_version",
                "preprocessing_mode",
                "explicit_technical_params",
                "actual_resolved_params",
                "parameter_adjustments",
                "error",
            ],
        ),
        "",
        "## Submission diagnostics",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(
            generated_rows,
            ["candidate_id", "model", "output_file", "rows", "pred_0_count", "pred_1_count", "pred_1_rate", "status"],
        ),
        "",
        "## Diff vs raw GB baseline submission",
        "",
        _markdown_table(
            diff_rows,
            [
                "baseline_candidate_id",
                "candidate_id",
                "baseline_file",
                "changed_predictions",
                "changed_pct",
                "baseline_pred_1_count",
                "candidate_pred_1_count",
                "baseline_pred_1_rate",
                "candidate_pred_1_rate",
                "delta_pred_1_rate",
                "status",
            ],
        ),
        "",
        "## Sanity checks",
        "",
        _markdown_table(sanity_rows, ["check", "status", "detail"]),
        "",
        "## Public score placeholder",
        "",
        "Public score:",
        "",
        "- submission_12a_svc_agebucket_v1.csv: TBD",
        "- submission_12b_gb_agebucket_v1.csv: TBD",
        "",
        _markdown_table(_public_score_rows(), ["output_file", "public_score"]),
        "",
        "## Decision rule after public score",
        "",
        "- If neither beats current public baseline `raw_tabular / GradientBoostingClassifier = 0.79665`, close AgeBucket v1 as public-transfer failed.",
        "- If one beats baseline, mark as checkpoint leader/candidate, but do not do row-level tuning.",
        "- Do not use public result to create micro-variants.",
    ]

    failures = [row for row in generated_rows if row["status"] != "PASS"]
    if failures:
        lines.extend(["", "## Failure details", "", _markdown_table(failures, ["candidate_id", "model", "output_file", "status"])])

    return "\n".join(lines) + "\n"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train = _add_agebucket_v1(train_raw)
    test = _add_agebucket_v1(test_raw)

    generated_rows = []
    predictions: dict[str, pd.Series] = {}
    for candidate in CANDIDATES:
        row, pred = _fit_predict_candidate(candidate, train, test)
        generated_rows.append(row)
        if pred is not None:
            predictions[candidate.candidate_id] = pred

    panel_rows = _model_panel_rows()
    diff_rows = _baseline_diff_rows(predictions)
    sanity_rows, sanity_passed = _submission_sanity_rows(generated_rows, test_raw)
    generation_passed = all(row["status"] == "PASS" for row in generated_rows)
    panel_passed = all(row["error"] == "" for row in panel_rows)
    diff_passed = all(row["status"] in {"PASS", "baseline file not found"} for row in diff_rows)
    all_passed = generation_passed and panel_passed and diff_passed and sanity_passed

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(generated_rows, panel_rows, diff_rows, sanity_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"submission files: {sum(row['status'] == 'PASS' for row in generated_rows)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    for row in generated_rows:
        print(
            f"{row['candidate_id']}: rows={row['rows']} pred_0={row['pred_0_count']} "
            f"pred_1={row['pred_1_count']} pred_1_rate={row['pred_1_rate']} status={row['status']}"
        )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
