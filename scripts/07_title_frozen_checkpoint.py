from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from itertools import combinations
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
from scripts.features import RAW_TABULAR, add_clean_features
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "07_title_frozen_checkpoint.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

FEATURE_SET_NAME = "raw_plus_title"
FEATURE_NAMES = [*RAW_TABULAR, "Title"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    feature_set: str
    model: str
    output_file: str


CANDIDATES = [
    Candidate(
        "raw_plus_title__SVC",
        FEATURE_SET_NAME,
        "SVC",
        "submission_07_title_raw_plus_title_svc.csv",
    ),
    Candidate(
        "raw_plus_title__GradientBoostingClassifier",
        FEATURE_SET_NAME,
        "GradientBoostingClassifier",
        "submission_07_title_raw_plus_title_gradient_boosting.csv",
    ),
    Candidate(
        "raw_plus_title__CatBoostClassifier",
        FEATURE_SET_NAME,
        "CatBoostClassifier",
        "submission_07_title_raw_plus_title_catboost.csv",
    ),
]

BASELINE_COMPARISONS = {
    "raw_plus_title__SVC": (
        "raw_tabular__SVC",
        "submissions/submission_05_baseline_raw_tabular_svc.csv",
    ),
    "raw_plus_title__GradientBoostingClassifier": (
        "raw_tabular__GradientBoostingClassifier",
        "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv",
    ),
    "raw_plus_title__CatBoostClassifier": (
        "raw_tabular__CatBoostClassifier",
        "submissions/submission_05_baseline_raw_tabular_catboost.csv",
    ),
}

PUBLIC_SCORES = {
    "submissions/submission_07_title_raw_plus_title_svc.csv": "0.77990",
    "submissions/submission_07_title_raw_plus_title_gradient_boosting.csv": "0.76794",
    "submissions/submission_07_title_raw_plus_title_catboost.csv": "0.76076",
}

PUBLIC_CHECKPOINT_STATUS = {
    "raw_plus_title__SVC": {
        "public_score": "0.77990",
        "baseline_public_score": "0.77990",
        "public_delta": "0.00000",
        "status": "CHECKPOINTED_NO_GAIN",
        "note": "matched raw_tabular SVC baseline; no public gain",
    },
    "raw_plus_title__GradientBoostingClassifier": {
        "public_score": "0.76794",
        "baseline_public_score": "0.79665",
        "public_delta": "-0.02871",
        "status": "REJECTED_PUBLIC_TRANSFER",
        "note": "train-side Title gain did not transfer; unrestricted Title damaged public score",
    },
    "raw_plus_title__CatBoostClassifier": {
        "public_score": "0.76076",
        "baseline_public_score": "0.77990",
        "public_delta": "-0.01914",
        "status": "REJECTED_PUBLIC_TRANSFER",
        "note": "train-side Title gain did not transfer; unrestricted Title damaged public score",
    },
}

EXCLUDED_MODEL_REASONS = {
    "RandomForestClassifier": "rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane",
    "LGBMClassifier": "rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane",
    "HistGradientBoostingClassifier": "rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane",
    "XGBClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
    "ExtraTreesClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
    "DecisionTreeClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
    "AdaBoostClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
    "LinearSVC": "not an active Title checkpoint lane from `06_title_feature_check`",
    "KNeighborsClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
    "GaussianNB": "not an active Title checkpoint lane from `06_title_feature_check`",
    "DummyClassifier": "not an active Title checkpoint lane from `06_title_feature_check`",
}


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


def _prepare_title_frame(raw: pd.DataFrame) -> pd.DataFrame:
    clean = add_clean_features(raw)
    frame = raw.copy()
    frame["Title"] = clean["Title"]
    return frame


def _model_panel_rows(candidates: list[Candidate]) -> list[dict[str, object]]:
    specs = _model_spec_by_name()
    rows = []

    for candidate in candidates:
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
                "model": candidate.model,
                "feature_set": candidate.feature_set,
                "package": spec.package,
                "package_version": baseline04._package_version(spec.version_package),
                "preprocessing_mode": spec.preprocessing_mode,
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
        "feature_set": candidate.feature_set,
        "model": candidate.model,
        "output_file": _relative(output_path),
        "rows": "",
        "pred_0_count": "",
        "pred_1_count": "",
        "pred_1_rate": "",
    }

    try:
        missing_train = [feature for feature in FEATURE_NAMES if feature not in train.columns]
        missing_test = [feature for feature in FEATURE_NAMES if feature not in test.columns]
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
                ("preprocess", make_preprocessor(spec.preprocessing_mode, list(FEATURE_NAMES))),
                ("model", model),
            ]
        )
        pipeline.fit(train[FEATURE_NAMES], train[TARGET])
        predictions = pd.Series(pipeline.predict(test[FEATURE_NAMES]), name=TARGET).astype(int)
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


def _baseline_difference_rows(predictions: dict[str, pd.Series]) -> list[dict[str, object]]:
    rows = []

    for candidate in CANDIDATES:
        base_candidate_id, baseline_file = BASELINE_COMPARISONS[candidate.candidate_id]
        title_pred = predictions.get(candidate.candidate_id)
        baseline_path = PROJECT_ROOT / baseline_file
        base = {
            "base_candidate_id": base_candidate_id,
            "title_candidate_id": candidate.candidate_id,
            "changed_predictions": "n/a",
            "changed_pct": "n/a",
            "base_pred_1_count": "n/a",
            "title_pred_1_count": "n/a",
            "base_pred_1_rate": "n/a",
            "title_pred_1_rate": "n/a",
            "delta_pred_1_rate": "n/a",
        }

        if title_pred is None:
            rows.append({**base, "status": "title prediction unavailable"})
            continue
        if not baseline_path.exists():
            rows.append({**base, "status": f"baseline file missing: {baseline_file}"})
            continue

        baseline_frame = pd.read_csv(baseline_path)
        baseline_pred = baseline_frame[TARGET].astype(int)
        if len(baseline_pred) != len(title_pred):
            rows.append({**base, "status": "row count mismatch"})
            continue

        changed = int((baseline_pred.to_numpy() != title_pred.to_numpy()).sum())
        base_pred_1_rate = float((baseline_pred == 1).mean())
        title_pred_1_rate = float((title_pred == 1).mean())
        rows.append(
            {
                **base,
                "changed_predictions": changed,
                "changed_pct": _round_float(changed / len(title_pred) * 100),
                "base_pred_1_count": int((baseline_pred == 1).sum()),
                "title_pred_1_count": int((title_pred == 1).sum()),
                "base_pred_1_rate": _round_float(base_pred_1_rate),
                "title_pred_1_rate": _round_float(title_pred_1_rate),
                "delta_pred_1_rate": _round_float(title_pred_1_rate - base_pred_1_rate),
                "status": "PASS",
            }
        )

    return rows


def _pairwise_difference_rows(predictions: dict[str, pd.Series]) -> list[dict[str, object]]:
    rows = []
    for candidate_a, candidate_b in combinations(CANDIDATES, 2):
        pred_a = predictions.get(candidate_a.candidate_id)
        pred_b = predictions.get(candidate_b.candidate_id)
        if pred_a is None or pred_b is None:
            rows.append(
                {
                    "candidate_a": candidate_a.candidate_id,
                    "candidate_b": candidate_b.candidate_id,
                    "changed_predictions": "n/a",
                    "changed_pct": "n/a",
                }
            )
            continue
        changed = int((pred_a.to_numpy() != pred_b.to_numpy()).sum())
        rows.append(
            {
                "candidate_a": candidate_a.candidate_id,
                "candidate_b": candidate_b.candidate_id,
                "changed_predictions": changed,
                "changed_pct": _round_float(changed / len(pred_a) * 100),
            }
        )
    return rows


def _submission_sanity_rows(
    generated_rows: list[dict[str, object]],
    test: pd.DataFrame,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    all_passed = True
    expected_order = test[ID_COLUMN].tolist() if ID_COLUMN in test.columns else []

    def add_check(check: str, passed: bool, detail: str) -> None:
        nonlocal all_passed
        all_passed = all_passed and passed
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    generated_files = [row for row in generated_rows if row["status"] == "PASS"]
    add_check(
        "exactly 3 new files generated",
        len(generated_files) == 3,
        f"{len(generated_files)} of 3 files generated",
    )

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

        add_check(f"{label}: every file has 418 rows", rows_ok, f"rows={len(frame)}")
        add_check(f"{label}: columns exactly PassengerId,Survived", columns_ok, ",".join(frame.columns))
        add_check(f"{label}: PassengerId order matches data/test.csv", order_ok, "order checked")
        add_check(f"{label}: Survived values only 0/1", values_ok, f"values={sorted(values)}")
        add_check(f"{label}: no duplicate PassengerId", duplicates_ok, "duplicates checked")

    return rows, all_passed


def _excluded_rows() -> list[dict[str, object]]:
    return [
        {"model": model, "reason": reason}
        for model, reason in EXCLUDED_MODEL_REASONS.items()
    ]


def _public_score_rows(generated_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "output_file": row["output_file"],
            "public_score": PUBLIC_SCORES.get(str(row["output_file"]), "MISSING"),
            "note": "Recorded after file generation; external checkpoint evidence only.",
        }
        for row in generated_rows
        if row["status"] == "PASS"
    ]


def _public_checkpoint_status_rows(generated_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in generated_rows:
        if row["status"] != "PASS":
            continue
        status = PUBLIC_CHECKPOINT_STATUS[str(row["candidate_id"])]
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "public_score": status["public_score"],
                "baseline_public_score": status["baseline_public_score"],
                "public_delta": status["public_delta"],
                "status": status["status"],
                "note": status["note"],
            }
        )
    return rows


def _build_report(
    generated_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    baseline_difference_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# 07 Title Frozen Checkpoint",
        "",
        "## Scope Boundary",
        "",
        "- full `train.csv` model fitting is allowed for frozen Title checkpoint file generation",
        "- `test.csv` is used only for inference",
        "- only `raw_plus_title` is used",
        "- `raw_plus_title` is defined as `RAW_TABULAR + [\"Title\"]`",
        "- `add_clean_features()` is used only to create `Title`; only `Title` is copied into the working frames",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- active model lanes are taken from `06_title_feature_check`",
        "- technical model parameters and package version logic match `04_baseline` / `06_title_feature_check`",
        "- no `gender_submission.csv` as truth",
        "- no test labels or row-level correctness checks",
        "- public scores are recorded only as post-generation checkpoint metadata",
        "- public scores are not used for training, inference, candidate selection, or row-level logic",
        "- no derived features other than `Title` are included",
        "- no hyperparameter tuning, threshold tuning, gating, probability threshold changes, PassengerId overrides, or manual correction rules",
        "- no target-derived features",
        "- candidates are fixed before public checkpointing and are not selected by public score",
        "",
        "## Candidate Selection",
        "",
        "Only active `raw_plus_title` lanes are promoted to frozen submission files.",
        "",
        _markdown_table(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    "feature_set": candidate.feature_set,
                    "model": candidate.model,
                    "output_file": f"submissions/{candidate.output_file}",
                }
                for candidate in CANDIDATES
            ],
            ["candidate_id", "feature_set", "model", "output_file"],
        ),
        "",
        "## Training / Inference Protocol",
        "",
        "1. Load `train.csv` and `test.csv`.",
        "2. Create clean features with `add_clean_features()` for train and test.",
        "3. Copy only `Title` into the train/test working frames.",
        "4. Select only `RAW_TABULAR + [\"Title\"]`.",
        "5. Build an sklearn `Pipeline` with `make_preprocessor(preprocessing_mode, feature_names)` and the model.",
        "6. Fit on full `train.csv`.",
        "7. Predict `Survived` for `test.csv` using model `.predict()` output.",
        "8. Write submission CSV with exactly `PassengerId` and `Survived`.",
        "",
        "## Model Panel Used",
        "",
        _markdown_table(
            panel_rows,
            [
                "model",
                "feature_set",
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
        "## Generated Submissions",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(
            generated_rows,
            [
                "candidate_id",
                "feature_set",
                "model",
                "output_file",
                "rows",
                "pred_0_count",
                "pred_1_count",
                "pred_1_rate",
                "status",
            ],
        ),
        "",
        "## Title vs raw_tabular baseline prediction difference",
        "",
        _markdown_table(
            baseline_difference_rows,
            [
                "base_candidate_id",
                "title_candidate_id",
                "changed_predictions",
                "changed_pct",
                "base_pred_1_count",
                "title_pred_1_count",
                "base_pred_1_rate",
                "title_pred_1_rate",
                "delta_pred_1_rate",
                "status",
            ],
        ),
        "",
        "## Pairwise prediction difference among Title submissions",
        "",
        _markdown_table(
            pairwise_rows,
            ["candidate_a", "candidate_b", "changed_predictions", "changed_pct"],
        ),
        "",
        "## Sanity Checks",
        "",
        _markdown_table(sanity_rows, ["check", "status", "detail"]),
        "",
        "## Diagnostic-only `f00_core_plus_title`",
        "",
        "- `f00_core_plus_title` was checked in `06_title_feature_check`.",
        "- It is not promoted to public checkpoint.",
        "- Reason: raw/title lanes are stronger and baseline `f00_core` is already a weak reference.",
        "",
        "## Excluded models",
        "",
        _markdown_table(_excluded_rows(), ["model", "reason"]),
        "",
        "## Public score checkpoint table",
        "",
        _markdown_table(_public_score_rows(generated_rows), ["output_file", "public_score", "note"]),
        "",
        "## Public checkpoint result summary",
        "",
        "- best public score in this checkpoint: `0.77990`",
        "- best candidate in this checkpoint: `raw_plus_title__SVC`",
        "- `raw_plus_title__SVC` matched its raw baseline public score and produced no public gain",
        "- `raw_plus_title__GradientBoostingClassifier` dropped from raw baseline `0.79665` to `0.76794`",
        "- `raw_plus_title__CatBoostClassifier` dropped from raw baseline `0.77990` to `0.76076`",
        "- unrestricted `Title` did not transfer as a full-strength direct feature",
        "- public score is checkpoint evidence only, not tuning feedback",
        "",
        "## Public checkpoint status",
        "",
        _markdown_table(
            _public_checkpoint_status_rows(generated_rows),
            [
                "candidate_id",
                "public_score",
                "baseline_public_score",
                "public_delta",
                "status",
                "note",
            ],
        ),
        "",
        "## Transfer-risk observation",
        "",
        "- `raw_plus_title` changed 10 predictions for SVC and produced no public gain.",
        "- `raw_plus_title` changed 22 predictions for GradientBoostingClassifier and CatBoostClassifier and both public scores dropped materially.",
        "- On a 418-row test set, larger flip counts can be a transfer-risk signal for overlapping derived features.",
        "- This is an empirical checkpoint observation, not a hard rule.",
        "- `Title` should not be used as an unrestricted full-strength feature in the next clean lane.",
        "- `Title` signal remains eligible for a separate gated/conservative check.",
        "",
        "## Short interpretation",
        "",
        "- These are frozen Title checkpoint files.",
        "- Public score was recorded after file generation.",
        "- No tuning, threshold change, gating, model parameter change, feature change, or PassengerId correction was made after public results.",
        "- Unrestricted `raw_plus_title` is rejected as a direct full-strength feature.",
        "- `raw_plus_title / SVC` is checkpointed with no gain.",
        "- `raw_plus_title / GradientBoostingClassifier` and `raw_plus_title / CatBoostClassifier` are rejected for public transfer.",
        "- `Title` signal is moved to `RETEST_FOR_GATING`.",
    ]

    failures = [row for row in generated_rows if row["status"] != "PASS"]
    if failures:
        lines.extend(
            [
                "",
                "## Failure details",
                "",
                _markdown_table(failures, ["candidate_id", "feature_set", "model", "output_file", "status"]),
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train = _prepare_title_frame(train_raw)
    test = _prepare_title_frame(test_raw)
    predictions: dict[str, pd.Series] = {}
    generated_rows = []

    for candidate in CANDIDATES:
        row, prediction = _fit_predict_candidate(candidate, train, test)
        generated_rows.append(row)
        if prediction is not None:
            predictions[candidate.candidate_id] = prediction

    panel_rows = _model_panel_rows(CANDIDATES)
    baseline_difference_rows = _baseline_difference_rows(predictions)
    pairwise_rows = _pairwise_difference_rows(predictions)
    sanity_rows, sanity_passed = _submission_sanity_rows(generated_rows, test_raw)
    generation_passed = all(row["status"] == "PASS" for row in generated_rows)
    baseline_diff_passed = all(row["status"] == "PASS" for row in baseline_difference_rows)
    all_passed = generation_passed and sanity_passed and baseline_diff_passed

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(
            generated_rows,
            panel_rows,
            baseline_difference_rows,
            pairwise_rows,
            sanity_rows,
            all_passed,
        ),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"submission files: {sum(row['status'] == 'PASS' for row in generated_rows)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
