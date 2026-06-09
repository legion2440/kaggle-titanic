from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
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


REPORT_PATH = REPORTS_DIR / "09_title_gated_frozen_checkpoint.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

MODEL_NAME = "GradientBoostingClassifier"
BASE_FEATURE_SET = "raw_tabular"
TITLE_FEATURE_SET = "raw_plus_title"
BASE_FEATURES = list(RAW_TABULAR)
TITLE_FEATURES = [*RAW_TABULAR, "Title"]
BLEND_WEIGHT = 0.10

CANDIDATE_ID = "title_gated_w010__GradientBoostingClassifier"
OUTPUT_FILE = "submission_09_title_gated_w010_gradient_boosting.csv"
BASELINE_CANDIDATE_ID = "raw_tabular__GradientBoostingClassifier"
BASELINE_FILE = "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv"

PUBLIC_SCORES = {
    f"submissions/{OUTPUT_FILE}": "0.78229",
}

PUBLIC_CHECKPOINT_STATUS = {
    CANDIDATE_ID: {
        "public_score": "0.78229",
        "baseline_public_score": "0.79665",
        "public_delta": "-0.01436",
        "status": "REJECTED_PUBLIC_TRANSFER",
        "note": "conservative Title blend did not beat raw GB baseline despite small flip count",
    }
}

MANUAL_DIAGNOSTIC_ROWS = [
    {
        "variant": "no-kill directional variant",
        "public_score": "0.78708",
        "note": "ad-hoc diagnostic; did not beat raw GB baseline",
    },
    {
        "variant": "kill-only directional variant",
        "public_score": "0.79186",
        "note": "ad-hoc diagnostic; did not beat raw GB baseline",
    },
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
MODEL_SPEC = next(spec for spec in baseline04.MODEL_SPECS if spec.model == MODEL_NAME)


def _json_dumps(value: Any) -> str:
    return baseline04._json_dumps(value)


def _round_float(value: float) -> float:
    return baseline04._round_float(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    return baseline04._markdown_table(rows, columns)


def _build_model() -> tuple[object | None, dict[str, Any], str]:
    return baseline04._build_model(MODEL_SPEC)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_title_frame(raw: pd.DataFrame) -> pd.DataFrame:
    clean = add_clean_features(raw)
    frame = raw.copy()
    frame["Title"] = clean["Title"]
    return frame


def _class1_proba(estimator: Pipeline, x_frame: pd.DataFrame) -> pd.Series:
    probabilities = estimator.predict_proba(x_frame)
    model = estimator.named_steps["model"]
    classes = getattr(model, "classes_", getattr(estimator, "classes_", None))
    if classes is None:
        raise RuntimeError("model classes_ unavailable for probability alignment")
    class_values = list(classes)
    if 1 not in class_values:
        raise RuntimeError(f"class 1 missing from fitted classes: {class_values}")
    return pd.Series(probabilities[:, class_values.index(1)], index=x_frame.index)


def _model_panel_rows() -> list[dict[str, object]]:
    rows = []
    for role, feature_set in [
        ("base model", BASE_FEATURE_SET),
        ("title model", TITLE_FEATURE_SET),
    ]:
        model, used_params, error_or_adjustment = _build_model()
        if model is not None and hasattr(model, "get_params"):
            actual_params = model.get_params(deep=False)
        elif model is not None:
            actual_params = "get_params_unavailable"
        else:
            actual_params = "model_unavailable"

        rows.append(
            {
                "role": role,
                "feature_set": feature_set,
                "model": MODEL_SPEC.model,
                "package": MODEL_SPEC.package,
                "package_version": baseline04._package_version(MODEL_SPEC.version_package),
                "preprocessing_mode": MODEL_SPEC.preprocessing_mode,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": "" if model is not None else "model unavailable",
                "error": "" if model is not None else error_or_adjustment,
            }
        )
    return rows


def _fit_predict_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, object], pd.Series | None]:
    output_path = SUBMISSIONS_DIR / OUTPUT_FILE
    base_row = {
        "candidate_id": CANDIDATE_ID,
        "output_file": _relative(output_path),
        "rows": "",
        "pred_0_count": "",
        "pred_1_count": "",
        "pred_1_rate": "",
    }

    try:
        missing_base_train = [feature for feature in BASE_FEATURES if feature not in train.columns]
        missing_title_train = [feature for feature in TITLE_FEATURES if feature not in train.columns]
        missing_base_test = [feature for feature in BASE_FEATURES if feature not in test.columns]
        missing_title_test = [feature for feature in TITLE_FEATURES if feature not in test.columns]
        if TARGET not in train.columns:
            raise ValueError(f"missing target column: {TARGET}")
        if ID_COLUMN not in test.columns:
            raise ValueError(f"missing test id column: {ID_COLUMN}")
        if missing_base_train:
            raise ValueError("missing base train feature columns: " + ", ".join(missing_base_train))
        if missing_title_train:
            raise ValueError("missing title train feature columns: " + ", ".join(missing_title_train))
        if missing_base_test:
            raise ValueError("missing base test feature columns: " + ", ".join(missing_base_test))
        if missing_title_test:
            raise ValueError("missing title test feature columns: " + ", ".join(missing_title_test))

        base_model, _, base_error = _build_model()
        title_model, _, title_error = _build_model()
        if base_model is None:
            raise RuntimeError(base_error)
        if title_model is None:
            raise RuntimeError(title_error)

        base_pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(MODEL_SPEC.preprocessing_mode, BASE_FEATURES)),
                ("model", base_model),
            ]
        )
        title_pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(MODEL_SPEC.preprocessing_mode, TITLE_FEATURES)),
                ("model", title_model),
            ]
        )

        base_pipeline.fit(train[BASE_FEATURES], train[TARGET])
        title_pipeline.fit(train[TITLE_FEATURES], train[TARGET])
        p_base = _class1_proba(base_pipeline, test[BASE_FEATURES])
        p_title = _class1_proba(title_pipeline, test[TITLE_FEATURES])
        p_blend = (1 - BLEND_WEIGHT) * p_base + BLEND_WEIGHT * p_title
        predictions = (p_blend >= 0.5).astype(int)

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


def _baseline_comparison_row(predictions: pd.Series | None) -> dict[str, object]:
    base = {
        "base_candidate_id": BASELINE_CANDIDATE_ID,
        "gated_candidate_id": CANDIDATE_ID,
        "changed_predictions": "n/a",
        "changed_pct": "n/a",
        "base_pred_1_count": "n/a",
        "gated_pred_1_count": "n/a",
        "base_pred_1_rate": "n/a",
        "gated_pred_1_rate": "n/a",
        "delta_pred_1_rate": "n/a",
        "flip_0_to_1": "n/a",
        "flip_1_to_0": "n/a",
        "status": "n/a",
    }
    if predictions is None:
        return {**base, "status": "gated prediction unavailable"}

    baseline_path = PROJECT_ROOT / BASELINE_FILE
    if not baseline_path.exists():
        return {**base, "status": f"baseline file missing: {BASELINE_FILE}"}

    baseline_frame = pd.read_csv(baseline_path)
    baseline_pred = baseline_frame[TARGET].astype(int)
    if len(baseline_pred) != len(predictions):
        return {**base, "status": "row count mismatch"}

    changed = baseline_pred.to_numpy() != predictions.to_numpy()
    base_pred_1_rate = float((baseline_pred == 1).mean())
    gated_pred_1_rate = float((predictions == 1).mean())
    return {
        **base,
        "changed_predictions": int(changed.sum()),
        "changed_pct": _round_float(float(changed.mean() * 100)),
        "base_pred_1_count": int((baseline_pred == 1).sum()),
        "gated_pred_1_count": int((predictions == 1).sum()),
        "base_pred_1_rate": _round_float(base_pred_1_rate),
        "gated_pred_1_rate": _round_float(gated_pred_1_rate),
        "delta_pred_1_rate": _round_float(gated_pred_1_rate - base_pred_1_rate),
        "flip_0_to_1": int(((baseline_pred == 0) & (predictions == 1)).sum()),
        "flip_1_to_0": int(((baseline_pred == 1) & (predictions == 0)).sum()),
        "status": "PASS",
    }


def _submission_sanity_rows(
    generated_row: dict[str, object],
    test: pd.DataFrame,
    baseline_hash_before: str,
    baseline_hash_after: str,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    all_passed = True
    output_path = PROJECT_ROOT / str(generated_row["output_file"])
    expected_order = test[ID_COLUMN].tolist() if ID_COLUMN in test.columns else []

    def add_check(check: str, passed: bool, detail: str) -> None:
        nonlocal all_passed
        all_passed = all_passed and passed
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    generated_ok = generated_row["status"] == "PASS" and output_path.exists()
    add_check("exactly 1 new file generated", generated_ok, str(generated_row["output_file"]))
    if generated_ok:
        frame = pd.read_csv(output_path)
        columns_ok = list(frame.columns) == [ID_COLUMN, TARGET]
        rows_ok = len(frame) == len(test) == 418
        order_ok = columns_ok and frame[ID_COLUMN].tolist() == expected_order
        values = set(frame[TARGET].dropna().astype(int).unique().tolist()) if TARGET in frame.columns else set()
        values_ok = values.issubset({0, 1}) and not frame[TARGET].isna().any()
        duplicates_ok = not frame[ID_COLUMN].duplicated().any() if ID_COLUMN in frame.columns else False

        add_check("file has 418 rows", rows_ok, f"rows={len(frame)}")
        add_check("columns exactly PassengerId,Survived", columns_ok, ",".join(frame.columns))
        add_check("PassengerId order matches data/test.csv", order_ok, "order checked")
        add_check("Survived values are only 0/1", values_ok, f"values={sorted(values)}")
        add_check("no duplicate PassengerId", duplicates_ok, "duplicates checked")
    else:
        add_check("file has 418 rows", False, "generated file missing")
        add_check("columns exactly PassengerId,Survived", False, "generated file missing")
        add_check("PassengerId order matches data/test.csv", False, "generated file missing")
        add_check("Survived values are only 0/1", False, "generated file missing")
        add_check("no duplicate PassengerId", False, "generated file missing")

    add_check(
        "existing baseline submission file was not modified",
        baseline_hash_before == baseline_hash_after,
        BASELINE_FILE,
    )
    return rows, all_passed


def _public_score_rows(generated_row: dict[str, object]) -> list[dict[str, object]]:
    if generated_row["status"] != "PASS":
        return []
    return [
        {
            "output_file": generated_row["output_file"],
            "public_score": PUBLIC_SCORES.get(str(generated_row["output_file"]), "MISSING"),
            "note": "Recorded after file generation; external checkpoint evidence only.",
        }
    ]


def _public_checkpoint_status_rows(generated_row: dict[str, object]) -> list[dict[str, object]]:
    if generated_row["status"] != "PASS":
        return []
    status = PUBLIC_CHECKPOINT_STATUS[str(generated_row["candidate_id"])]
    return [
        {
            "candidate_id": generated_row["candidate_id"],
            "public_score": status["public_score"],
            "baseline_public_score": status["baseline_public_score"],
            "public_delta": status["public_delta"],
            "status": status["status"],
            "note": status["note"],
        }
    ]


def _build_report(
    generated_row: dict[str, object],
    model_rows: list[dict[str, object]],
    comparison_row: dict[str, object],
    sanity_rows: list[dict[str, object]],
    all_passed: bool,
) -> str:
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# 09 Title Gated Frozen Checkpoint",
        "",
        "## Scope Boundary",
        "",
        "- full `train.csv` model fitting is allowed for frozen checkpoint file generation",
        "- `test.csv` is used only for inference",
        "- only one preselected conservative blend candidate is evaluated",
        "- base feature set is `raw_tabular`",
        "- title feature set is `raw_plus_title`",
        "- blend weight is fixed at `0.10`",
        "- `add_clean_features()` is used only to create `Title`; only `Title` is copied from its output",
        "- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`",
        "- same technical/default model parameters as `04_baseline` and `08_title_gating_check`",
        "- `predict_proba()` is used only for base/title probabilities",
        "- no `gender_submission.csv` as truth",
        "- no test labels or row-level correctness checks",
        "- public scores are recorded only as post-generation checkpoint metadata",
        "- public scores are not used for training, inference, candidate selection, or row-level logic",
        "- no features other than `Title` are added",
        "- no hyperparameter tuning, model parameter tuning, threshold tuning, multiple-weight search, PassengerId overrides, manual correction rules, or target-derived features",
        "",
        "## Input Evidence",
        "",
        "- `raw_tabular / GradientBoostingClassifier` is current clean public baseline leader from step 05.",
        "- unrestricted `raw_plus_title / GradientBoostingClassifier` was rejected for public transfer in step 07.",
        "- `08_title_gating_check` selected `w=0.10` as conservative train-side candidate.",
        "",
        "## Candidate Definition",
        "",
        _markdown_table(
            [
                {
                    "candidate_id": CANDIDATE_ID,
                    "base_model": f"{BASE_FEATURE_SET} / {MODEL_NAME}",
                    "title_model": f"{TITLE_FEATURE_SET} / {MODEL_NAME}",
                    "blend_weight": f"{BLEND_WEIGHT:.2f}",
                    "output_file": f"submissions/{OUTPUT_FILE}",
                }
            ],
            ["candidate_id", "base_model", "title_model", "blend_weight", "output_file"],
        ),
        "",
        "- `w=0.10` is used because it was the smaller stable conservative weight among the best conservative rows in step 08.",
        "- `w=0.15` is not used because the same OOF/flip result is available with a smaller weight.",
        "- `w=0.25` and `w=0.30` are not used because they are already in the caution zone.",
        "- `w=1.00` is not used because full-strength Title was already rejected.",
        "",
        "## Training / Inference Protocol",
        "",
        "1. Load `train.csv` and `test.csv`.",
        "2. Create `Title` for train and test using `add_clean_features()`.",
        "3. Copy only `Title` into train/test working frames.",
        "4. Build the base pipeline with `RAW_TABULAR` features.",
        "5. Build the title pipeline with `RAW_TABULAR + [\"Title\"]` features.",
        "6. Fit both pipelines on full `train.csv`.",
        "7. For `test.csv`, get class-1 probabilities aligned to `Survived == 1`.",
        "8. Blend probabilities as `p_blend = 0.90 * p_base + 0.10 * p_title`.",
        "9. Predict `Survived = 1` when `p_blend >= 0.5`, otherwise `0`.",
        "10. Write submission CSV with exactly `PassengerId` and `Survived`.",
        "",
        "## Model Panel Used",
        "",
        _markdown_table(
            model_rows,
            [
                "role",
                "feature_set",
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
        "## Generated Submission",
        "",
        f"- overall status: `{status}`",
        "",
        _markdown_table(
            [generated_row],
            [
                "candidate_id",
                "output_file",
                "rows",
                "pred_0_count",
                "pred_1_count",
                "pred_1_rate",
                "status",
            ],
        ),
        "",
        "## Baseline comparison",
        "",
        _markdown_table(
            [comparison_row],
            [
                "base_candidate_id",
                "gated_candidate_id",
                "changed_predictions",
                "changed_pct",
                "base_pred_1_count",
                "gated_pred_1_count",
                "base_pred_1_rate",
                "gated_pred_1_rate",
                "delta_pred_1_rate",
                "flip_0_to_1",
                "flip_1_to_0",
                "status",
            ],
        ),
        "",
        "## Sanity Checks",
        "",
        _markdown_table(sanity_rows, ["check", "status", "detail"]),
        "",
        "## Public score checkpoint table",
        "",
        _markdown_table(_public_score_rows(generated_row), ["output_file", "public_score", "note"]),
        "",
        "## Public checkpoint result summary",
        "",
        "- baseline public score: `0.79665`",
        "- gated Title public score: `0.78229`",
        "- public delta vs baseline: `-0.01436`",
        "- gated candidate changed 6 predictions vs raw GB baseline:",
        "  - 4 flips `0 -> 1`",
        "  - 2 flips `1 -> 0`",
        "- conservative `w=0.10` improved OOF in step 08, but did not transfer to public",
        "- public score is checkpoint evidence only, not tuning feedback",
        "",
        "## Public checkpoint status",
        "",
        _markdown_table(
            _public_checkpoint_status_rows(generated_row),
            ["candidate_id", "public_score", "baseline_public_score", "public_delta", "status", "note"],
        ),
        "",
        "## Additional manual diagnostic observations",
        "",
        _markdown_table(MANUAL_DIAGNOSTIC_ROWS, ["variant", "public_score", "note"]),
        "",
        "- These variants are not promoted.",
        "- They are recorded only to explain that directional post-processing also did not beat the raw GB baseline.",
        "- Do not create new submission files in this step.",
        "- Do not use these manual diagnostics to tune row-level rules.",
        "",
        "## Title lane conclusion",
        "",
        "- Broad `Title` signal is not globally declared useless.",
        "- In the current clean active GB lane:",
        "  - unrestricted `raw_plus_title` failed public transfer in step 07;",
        "  - conservative gated `w=0.10` also failed public transfer in step 09.",
        "- Therefore broad/direct `Title` usage is closed for the current GB lane.",
        "- `Title` may still be retested later only as a narrow, constrained signal inside the Age/AgeBand/Child block, for example `Master` as a child-male proxy.",
        "- Do not continue broad Title tuning in the current lane.",
        "",
        "## Short interpretation",
        "",
        "- This is a frozen conservative Title checkpoint.",
        "- Public score was recorded after file generation.",
        "- No tuning, threshold change, row-level correction, model parameter change, feature change, or PassengerId correction was made after public results.",
        "- `title_gated_w010 / GradientBoostingClassifier` is rejected for public transfer.",
        "- Current clean public baseline leader remains `raw_tabular / GradientBoostingClassifier` with `0.79665`.",
        "- Broad Title lane is closed for now.",
        "- Narrow Title-derived child signal is deferred to a later Age/AgeBand/Child check.",
    ]

    if generated_row["status"] != "PASS":
        lines.extend(
            [
                "",
                "## Failure details",
                "",
                _markdown_table([generated_row], ["candidate_id", "output_file", "status"]),
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    baseline_path = PROJECT_ROOT / BASELINE_FILE
    baseline_hash_before = _file_sha256(baseline_path) if baseline_path.exists() else "missing_before"
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train = _prepare_title_frame(train_raw)
    test = _prepare_title_frame(test_raw)

    generated_row, predictions = _fit_predict_submission(train, test)
    baseline_hash_after = _file_sha256(baseline_path) if baseline_path.exists() else "missing_after"
    model_rows = _model_panel_rows()
    comparison_row = _baseline_comparison_row(predictions)
    sanity_rows, sanity_passed = _submission_sanity_rows(
        generated_row,
        test_raw,
        baseline_hash_before,
        baseline_hash_after,
    )
    generation_passed = generated_row["status"] == "PASS"
    comparison_passed = comparison_row["status"] == "PASS"
    model_passed = all(row["error"] == "" for row in model_rows)
    all_passed = generation_passed and comparison_passed and sanity_passed and model_passed

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(generated_row, model_rows, comparison_row, sanity_rows, all_passed),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"submission files: {1 if generated_row['status'] == 'PASS' else 0}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
