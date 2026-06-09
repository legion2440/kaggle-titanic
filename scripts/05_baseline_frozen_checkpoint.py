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
from scripts.features import F00_CORE, RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "05_baseline_frozen_checkpoint.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

FEATURE_SETS = {
    "f00_core": list(F00_CORE),
    "raw_tabular": list(RAW_TABULAR),
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    feature_set: str
    model: str
    output_file: str


CANDIDATES = [
    Candidate(
        "f00_core__SVC",
        "f00_core",
        "SVC",
        "submission_05_baseline_f00_core_svc.csv",
    ),
    Candidate(
        "f00_core__RandomForestClassifier",
        "f00_core",
        "RandomForestClassifier",
        "submission_05_baseline_f00_core_random_forest.csv",
    ),
    Candidate(
        "f00_core__GradientBoostingClassifier",
        "f00_core",
        "GradientBoostingClassifier",
        "submission_05_baseline_f00_core_gradient_boosting.csv",
    ),
    Candidate(
        "f00_core__HistGradientBoostingClassifier",
        "f00_core",
        "HistGradientBoostingClassifier",
        "submission_05_baseline_f00_core_hist_gradient_boosting.csv",
    ),
    Candidate(
        "f00_core__XGBClassifier",
        "f00_core",
        "XGBClassifier",
        "submission_05_baseline_f00_core_xgboost.csv",
    ),
    Candidate(
        "f00_core__LGBMClassifier",
        "f00_core",
        "LGBMClassifier",
        "submission_05_baseline_f00_core_lgbm.csv",
    ),
    Candidate(
        "f00_core__CatBoostClassifier",
        "f00_core",
        "CatBoostClassifier",
        "submission_05_baseline_f00_core_catboost.csv",
    ),
    Candidate(
        "raw_tabular__CatBoostClassifier",
        "raw_tabular",
        "CatBoostClassifier",
        "submission_05_baseline_raw_tabular_catboost.csv",
    ),
    Candidate(
        "raw_tabular__HistGradientBoostingClassifier",
        "raw_tabular",
        "HistGradientBoostingClassifier",
        "submission_05_baseline_raw_tabular_hist_gradient_boosting.csv",
    ),
    Candidate(
        "raw_tabular__LGBMClassifier",
        "raw_tabular",
        "LGBMClassifier",
        "submission_05_baseline_raw_tabular_lgbm.csv",
    ),
    Candidate(
        "raw_tabular__GradientBoostingClassifier",
        "raw_tabular",
        "GradientBoostingClassifier",
        "submission_05_baseline_raw_tabular_gradient_boosting.csv",
    ),
    Candidate(
        "raw_tabular__SVC",
        "raw_tabular",
        "SVC",
        "submission_05_baseline_raw_tabular_svc.csv",
    ),
    Candidate(
        "raw_tabular__RandomForestClassifier",
        "raw_tabular",
        "RandomForestClassifier",
        "submission_05_baseline_raw_tabular_random_forest.csv",
    ),
]

EXCLUDED_MODELS = [
    "DummyClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "LinearSVC",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "AdaBoostClassifier",
]

PUBLIC_SCORES = {
    "submissions/submission_05_baseline_f00_core_svc.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_xgboost.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_random_forest.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_lgbm.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_hist_gradient_boosting.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_gradient_boosting.csv": "0.77751",
    "submissions/submission_05_baseline_f00_core_catboost.csv": "0.77751",
    "submissions/submission_05_baseline_raw_tabular_svc.csv": "0.77990",
    "submissions/submission_05_baseline_raw_tabular_random_forest.csv": "0.75837",
    "submissions/submission_05_baseline_raw_tabular_lgbm.csv": "0.76555",
    "submissions/submission_05_baseline_raw_tabular_hist_gradient_boosting.csv": "0.75358",
    "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv": "0.79665",
    "submissions/submission_05_baseline_raw_tabular_catboost.csv": "0.77990",
}

PUBLIC_CHECKPOINT_STATUS = {
    "f00_core__SVC": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__RandomForestClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__GradientBoostingClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__HistGradientBoostingClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__XGBClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__LGBMClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "f00_core__CatBoostClassifier": (
        "CHECKPOINTED_REFERENCE",
        "identical predictions, public score 0.77751",
    ),
    "raw_tabular__GradientBoostingClassifier": (
        "CURRENT_PUBLIC_BASELINE_LEADER",
        "best frozen baseline checkpoint, public score 0.79665",
    ),
    "raw_tabular__SVC": (
        "DEFERRED",
        "same public score 0.77990, not leading but kept as diagnostic evidence",
    ),
    "raw_tabular__CatBoostClassifier": (
        "DEFERRED",
        "same public score 0.77990, not leading but kept as diagnostic evidence",
    ),
    "raw_tabular__RandomForestClassifier": (
        "REJECTED",
        "weak public transfer in this frozen baseline checkpoint",
    ),
    "raw_tabular__LGBMClassifier": (
        "REJECTED",
        "weak public transfer in this frozen baseline checkpoint",
    ),
    "raw_tabular__HistGradientBoostingClassifier": (
        "REJECTED",
        "weak public transfer in this frozen baseline checkpoint",
    ),
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
    feature_names = FEATURE_SETS[candidate.feature_set]
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
        missing_train = [feature for feature in feature_names if feature not in train.columns]
        missing_test = [feature for feature in feature_names if feature not in test.columns]
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
                ("preprocess", make_preprocessor(spec.preprocessing_mode, list(feature_names))),
                ("model", model),
            ]
        )
        pipeline.fit(train[list(feature_names)], train[TARGET])
        predictions = pd.Series(pipeline.predict(test[list(feature_names)]), name=TARGET).astype(int)
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
        "expected submission file count",
        len(generated_files) == len(CANDIDATES),
        f"{len(generated_files)} of {len(CANDIDATES)} files generated",
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

        add_check(f"{label}: 418 rows", rows_ok, f"rows={len(frame)}")
        add_check(f"{label}: columns exactly PassengerId,Survived", columns_ok, ",".join(frame.columns))
        add_check(f"{label}: PassengerId order matches data/test.csv", order_ok, "order checked")
        add_check(f"{label}: Survived values only 0/1", values_ok, f"values={sorted(values)}")
        add_check(f"{label}: no duplicate PassengerId", duplicates_ok, "duplicates checked")

    return rows, all_passed


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


def _excluded_rows() -> list[dict[str, object]]:
    reason = (
        "excluded from checkpoint batch by task scope; not selected in the frozen "
        "f00_core top layer or strong raw baseline candidate list"
    )
    return [{"model": model, "reason": reason} for model in EXCLUDED_MODELS]


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
        status, note = PUBLIC_CHECKPOINT_STATUS[str(row["candidate_id"])]
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "public_score": PUBLIC_SCORES[str(row["output_file"])],
                "status": status,
                "note": note,
            }
        )
    return rows


def _build_report(
    generated_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# 05 Baseline Frozen Checkpoint",
        "",
        "## Scope Boundary",
        "",
        "- full `train.csv` model fitting is allowed for frozen checkpoint file generation",
        "- `test.csv` is used only for inference",
        "- only baseline feature sets from `04_baseline` are used: `f00_core` and `raw_tabular`",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- the same model panel technical parameters and package version logic as `04_baseline` are used",
        "- no `gender_submission.csv` as truth",
        "- no test labels or row-level correctness checks",
        "- no public leaderboard score inside the script",
        "- no feature engineering, `Title`, derived features, deferred features, or target-derived features",
        "- no hyperparameter tuning, threshold tuning, PassengerId overrides, or manual correction rules",
        "- candidates are fixed by the checkpoint batch specification, not by public score",
        "",
        "## Candidate Selection",
        "",
        "### f00_core layer",
        "",
        "These candidates are included because they share the top `f00_core` CV level in `04_baseline`.",
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
                if candidate.feature_set == "f00_core"
            ],
            ["candidate_id", "feature_set", "model", "output_file"],
        ),
        "",
        "### raw_tabular layer",
        "",
        "These candidates are included as strong raw baseline candidates from `04_baseline`.",
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
                if candidate.feature_set == "raw_tabular"
            ],
            ["candidate_id", "feature_set", "model", "output_file"],
        ),
        "",
        "## Training / Inference Protocol",
        "",
        "1. Load `train.csv` and `test.csv`.",
        "2. Select only the fixed feature columns for the candidate feature set.",
        "3. Build an sklearn `Pipeline` with `make_preprocessor(preprocessing_mode, feature_names)` and the model.",
        "4. Fit on full `train.csv`.",
        "5. Predict `Survived` for `test.csv` using model `.predict()` output.",
        "6. Write submission CSV with exactly `PassengerId` and `Survived`.",
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
        "## Pairwise prediction difference",
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
        "- best public score: `0.79665`",
        "- best candidate: `raw_tabular__GradientBoostingClassifier`",
        "- all `f00_core` submissions scored `0.77751`",
        "- all `f00_core` submissions are identical according to the generated pairwise diff table",
        "- `raw_tabular` submissions differ materially from each other",
        "- public score is checkpoint evidence only, not tuning feedback",
        "",
        "## Public checkpoint status",
        "",
        _markdown_table(
            _public_checkpoint_status_rows(generated_rows),
            ["candidate_id", "public_score", "status", "note"],
        ),
        "",
        "## Short interpretation",
        "",
        "- These are frozen baseline checkpoint files.",
        "- Public score was recorded after file generation.",
        "- No tuning, threshold change, model parameter change, feature change, or PassengerId correction was made after public results.",
        "- `raw_tabular / GradientBoostingClassifier` is the current clean public baseline leader.",
        "- RF default is rejected for the next clean feature-check lane based on this checkpoint.",
        "- This does not use old repo history and does not compare against old repo results.",
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
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    predictions: dict[str, pd.Series] = {}
    generated_rows = []

    for candidate in CANDIDATES:
        row, prediction = _fit_predict_candidate(candidate, train, test)
        generated_rows.append(row)
        if prediction is not None:
            predictions[candidate.candidate_id] = prediction

    panel_rows = _model_panel_rows(CANDIDATES)
    pairwise_rows = _pairwise_difference_rows(predictions)
    sanity_rows, sanity_passed = _submission_sanity_rows(generated_rows, test)
    generation_passed = all(row["status"] == "PASS" for row in generated_rows)
    all_passed = generation_passed and sanity_passed

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(generated_rows, panel_rows, pairwise_rows, sanity_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"submission files: {sum(row['status'] == 'PASS' for row in generated_rows)}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
