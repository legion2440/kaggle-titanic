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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import ID_COLUMN, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "14_agemissing_frozen_checkpoint.md"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

RAW_TABULAR = ["Sex", "Pclass", "Embarked", "Age", "SibSp", "Parch", "Fare"]
RAW_PLUS_AGEMISSING = [
    "Sex",
    "Pclass",
    "Embarked",
    "Age",
    "AgeMissing",
    "SibSp",
    "Parch",
    "Fare",
]
HISTGB_CATEGORICAL_FEATURES = ["Sex", "Pclass", "Embarked"]
BASELINE_FILE = "submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv"
BASELINE_CANDIDATE_ID = "raw_tabular__GradientBoostingClassifier"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model_name: str
    features: list[str]
    preprocessing_logic: str
    output_file: str
    reason: str


CANDIDATES = [
    Candidate(
        candidate_id="gb_raw_plus_agemissing_median",
        model_name="GradientBoostingClassifier",
        features=RAW_PLUS_AGEMISSING,
        preprocessing_logic=(
            "AgeMissing created before imputation; Age remains median-imputed through "
            "existing unscaled_tree preprocessing; categorical most_frequent + one-hot"
        ),
        output_file="submission_14a_gb_raw_plus_agemissing_median.csv",
        reason="step 13 KEEP_CANDIDATE: cv_mean 0.828284, net OOF +1",
    ),
    Candidate(
        candidate_id="gb_raw_age_sentinel_plus_agemissing",
        model_name="GradientBoostingClassifier",
        features=RAW_PLUS_AGEMISSING,
        preprocessing_logic=(
            "AgeMissing created before fill; missing Age replaced with sentinel -1; "
            "Age has no missing values after sentinel replacement; other numeric missing "
            "handling remains safe; categorical most_frequent + one-hot"
        ),
        output_file="submission_14b_gb_age_sentinel_plus_agemissing.csv",
        reason="step 13 KEEP_CANDIDATE: cv_mean 0.833890, net OOF +6",
    ),
    Candidate(
        candidate_id="histgb_raw_tabular_nan_native",
        model_name="HistGradientBoostingClassifier",
        features=RAW_TABULAR,
        preprocessing_logic=(
            "Age kept as NaN; no Age median imputation; Sex, Pclass, and Embarked "
            'converted to pandas categorical dtype for categorical_features="from_dtype"'
        ),
        output_file="submission_14c_histgb_raw_tabular_nan_native.csv",
        reason="step 13 diagnostic native-NaN lane: cv_mean 0.838359, net OOF +10 vs GB baseline",
    ),
]


MODEL_PANEL_COLUMNS = [
    "candidate_id",
    "model_class",
    "package",
    "package_version",
    "preprocessing_logic",
    "explicit_technical_params",
    "actual_resolved_params",
    "parameter_adjustments",
    "error",
]

SUBMISSION_COLUMNS = [
    "candidate_id",
    "output_file",
    "rows",
    "pred_0_count",
    "pred_1_count",
    "pred_1_rate",
    "status",
]

DIFF_COLUMNS = [
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


class AgeSentinelTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y: object = None) -> "AgeSentinelTransformer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        out["Age"] = out["Age"].fillna(-1)
        return out


class HistGBNativeNanPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], categorical_features: list[str]) -> None:
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.categories_: dict[str, list[object]] = {}

    def fit(self, x: pd.DataFrame, y: object = None) -> "HistGBNativeNanPreprocessor":
        self.categories_ = {}
        for feature in self.categorical_features:
            if feature in self.features:
                self.categories_[feature] = sorted(
                    value for value in x[feature].dropna().unique().tolist()
                )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x[self.features].copy()
        for feature in self.categorical_features:
            if feature in out.columns:
                out[feature] = pd.Categorical(out[feature], categories=self.categories_[feature])
        return out


class HistGBOrdinalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str], categorical_features: list[str]) -> None:
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.maps_: dict[str, dict[object, int]] = {}

    def fit(self, x: pd.DataFrame, y: object = None) -> "HistGBOrdinalPreprocessor":
        self.maps_ = {}
        for feature in self.categorical_features:
            if feature in self.features:
                categories = sorted(value for value in x[feature].dropna().unique().tolist())
                self.maps_[feature] = {category: code for code, category in enumerate(categories)}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x[self.features].copy()
        for feature in self.categorical_features:
            if feature in out.columns:
                out[feature] = out[feature].map(self.maps_[feature]).fillna(-1).astype(float)
        return out


def _add_agemissing(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["AgeMissing"] = out["Age"].isna().astype(int)
    return out


def _histgb_categorical_mask(features: list[str]) -> list[bool]:
    categorical_set = set(HISTGB_CATEGORICAL_FEATURES)
    return [feature in categorical_set for feature in features]


def _build_model(model_name: str) -> tuple[object | None, dict[str, Any], str]:
    spec = MODEL_SPECS_BY_NAME[model_name]
    return baseline04._build_model(spec)


def _build_estimator(
    candidate: Candidate,
    histgb_strategy: str = "from_dtype",
) -> tuple[Pipeline | None, dict[str, Any], str, str]:
    model, used_params, adjustment = _build_model(candidate.model_name)
    if model is None:
        return None, used_params, adjustment, histgb_strategy

    if candidate.candidate_id == "gb_raw_plus_agemissing_median":
        return (
            Pipeline(
                steps=[
                    ("preprocess", make_preprocessor("unscaled_tree", list(candidate.features))),
                    ("model", model),
                ]
            ),
            used_params,
            adjustment,
            histgb_strategy,
        )

    if candidate.candidate_id == "gb_raw_age_sentinel_plus_agemissing":
        return (
            Pipeline(
                steps=[
                    ("age_sentinel", AgeSentinelTransformer()),
                    ("preprocess", make_preprocessor("unscaled_tree", list(candidate.features))),
                    ("model", model),
                ]
            ),
            used_params,
            adjustment,
            histgb_strategy,
        )

    if candidate.candidate_id != "histgb_raw_tabular_nan_native":
        return None, used_params, f"Unsupported candidate: {candidate.candidate_id}", histgb_strategy

    if histgb_strategy == "from_dtype":
        model.set_params(categorical_features="from_dtype")
        used_params = {**used_params, "categorical_features": "from_dtype"}
        return (
            Pipeline(
                steps=[
                    (
                        "preprocess",
                        HistGBNativeNanPreprocessor(
                            list(candidate.features),
                            HISTGB_CATEGORICAL_FEATURES,
                        ),
                    ),
                    ("model", model),
                ]
            ),
            used_params,
            adjustment,
            histgb_strategy,
        )

    model.set_params(categorical_features=_histgb_categorical_mask(candidate.features))
    used_params = {
        **used_params,
        "categorical_features": _histgb_categorical_mask(candidate.features),
    }
    fallback_adjustment = (
        (adjustment + "; " if adjustment else "")
        + 'fallback categorical handling: ordinal codes; missing/unknown encoded as -1'
    )
    return (
        Pipeline(
            steps=[
                (
                    "preprocess",
                    HistGBOrdinalPreprocessor(
                        list(candidate.features),
                        HISTGB_CATEGORICAL_FEATURES,
                    ),
                ),
                ("model", model),
            ]
        ),
        used_params,
        fallback_adjustment,
        histgb_strategy,
    )


def _fit_predict_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    histgb_strategy: str,
) -> tuple[dict[str, object], pd.Series | None, str, str]:
    output_path = SUBMISSIONS_DIR / candidate.output_file
    base_row = {
        "candidate_id": candidate.candidate_id,
        "output_file": _relative(output_path),
        "rows": "",
        "pred_0_count": "",
        "pred_1_count": "",
        "pred_1_rate": "",
    }

    try:
        missing_train = [feature for feature in candidate.features if feature not in train.columns]
        missing_test = [feature for feature in candidate.features if feature not in test.columns]
        if TARGET not in train.columns:
            raise ValueError(f"missing target column: {TARGET}")
        if ID_COLUMN not in test.columns:
            raise ValueError(f"missing test id column: {ID_COLUMN}")
        if missing_train:
            raise ValueError("missing train feature columns: " + ", ".join(missing_train))
        if missing_test:
            raise ValueError("missing test feature columns: " + ", ".join(missing_test))

        effective_strategy = histgb_strategy
        try:
            estimator, _, build_error, effective_strategy = _build_estimator(
                candidate,
                histgb_strategy,
            )
            if estimator is None:
                raise RuntimeError(build_error)
            estimator.fit(train[candidate.features], train[TARGET].astype(int))
        except Exception as exc:
            if candidate.model_name != "HistGradientBoostingClassifier" or histgb_strategy != "from_dtype":
                raise
            effective_strategy = "ordinal_fallback"
            estimator, _, build_error, effective_strategy = _build_estimator(
                candidate,
                effective_strategy,
            )
            if estimator is None:
                raise RuntimeError(build_error) from exc
            estimator.fit(train[candidate.features], train[TARGET].astype(int))

        predictions = pd.Series(
            estimator.predict(test[candidate.features]),
            name=TARGET,
        ).astype(int)
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
            effective_strategy,
            "",
        )
    except Exception as exc:
        return (
            {**base_row, "status": f"FAIL: {type(exc).__name__}: {exc}"},
            None,
            histgb_strategy,
            f"{type(exc).__name__}: {exc}",
        )


def _model_panel_rows(histgb_strategy: str, histgb_error: str) -> list[dict[str, object]]:
    rows = []
    for candidate in CANDIDATES:
        estimator, used_params, adjustment, _ = _build_estimator(candidate, histgb_strategy)
        spec = MODEL_SPECS_BY_NAME[candidate.model_name]
        if estimator is None:
            actual_params: object = "model_unavailable"
            parameter_adjustments = ""
            error = adjustment or histgb_error
        else:
            model = estimator.named_steps["model"]
            actual_params = model.get_params(deep=False) if hasattr(model, "get_params") else "n/a"
            parameter_adjustments = adjustment
            error = histgb_error if candidate.model_name == "HistGradientBoostingClassifier" else ""

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "model_class": candidate.model_name,
                "package": spec.package,
                "package_version": baseline04._package_version(spec.version_package),
                "preprocessing_logic": candidate.preprocessing_logic,
                "explicit_technical_params": _json_dumps(used_params),
                "actual_resolved_params": _json_dumps(actual_params),
                "parameter_adjustments": parameter_adjustments,
                "error": error,
            }
        )
    return rows


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
        if ID_COLUMN in baseline_frame.columns:
            baseline_frame = baseline_frame.sort_values(ID_COLUMN).reset_index(drop=True)
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


def _sanity_rows(generated_rows: list[dict[str, object]], test: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    output_paths = [SUBMISSIONS_DIR / candidate.output_file for candidate in CANDIDATES]
    actual_submission_14 = sorted(SUBMISSIONS_DIR.glob("submission_14*.csv"))
    expected_names = {path.name for path in output_paths}
    actual_names = {path.name for path in actual_submission_14}
    extra_names = sorted(actual_names - expected_names)
    missing_names = sorted(expected_names - actual_names)
    rows.append(
        {
            "check": "exactly 3 submission_14 files generated",
            "status": "PASS" if len(actual_submission_14) == 3 and not extra_names and not missing_names else "FAIL",
            "detail": (
                f"count={len(actual_submission_14)}; "
                f"extra={extra_names or 'none'}; missing={missing_names or 'none'}"
            ),
        }
    )

    for candidate in CANDIDATES:
        path = SUBMISSIONS_DIR / candidate.output_file
        if not path.exists():
            rows.append(
                {
                    "check": f"{candidate.candidate_id}: file exists",
                    "status": "FAIL",
                    "detail": _relative(path),
                }
            )
            continue

        frame = pd.read_csv(path)
        expected_columns = [ID_COLUMN, TARGET]
        rows.extend(
            [
                {
                    "check": f"{candidate.candidate_id}: 418 rows",
                    "status": "PASS" if len(frame) == len(test) == 418 else "FAIL",
                    "detail": f"rows={len(frame)}",
                },
                {
                    "check": f"{candidate.candidate_id}: columns exactly PassengerId,Survived",
                    "status": "PASS" if list(frame.columns) == expected_columns else "FAIL",
                    "detail": ",".join(frame.columns),
                },
                {
                    "check": f"{candidate.candidate_id}: PassengerId order matches data/test.csv",
                    "status": "PASS"
                    if frame[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
                    else "FAIL",
                    "detail": "order checked",
                },
                {
                    "check": f"{candidate.candidate_id}: Survived values only 0/1",
                    "status": "PASS" if set(frame[TARGET].unique()).issubset({0, 1}) else "FAIL",
                    "detail": f"values={sorted(frame[TARGET].unique().tolist())}",
                },
                {
                    "check": f"{candidate.candidate_id}: no duplicate PassengerId",
                    "status": "PASS" if not frame[ID_COLUMN].duplicated().any() else "FAIL",
                    "detail": "duplicates checked",
                },
            ]
        )

    failed_submission_rows = [row for row in generated_rows if row["status"] != "PASS"]
    rows.append(
        {
            "check": "all candidate predictions succeeded",
            "status": "PASS" if not failed_submission_rows else "FAIL",
            "detail": f"failed={len(failed_submission_rows)}",
        }
    )
    return rows


def _fixed_candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "model": candidate.model_name,
            "features": ", ".join(candidate.features),
            "output_file": "submissions/" + candidate.output_file,
            "reason": candidate.reason,
        }
        for candidate in CANDIDATES
    ]


def _preprocessing_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "preprocessing_logic": candidate.preprocessing_logic,
        }
        for candidate in CANDIDATES
    ]


def _public_score_rows() -> list[dict[str, object]]:
    return [
        {"output_file": "submission_14a_gb_raw_plus_agemissing_median.csv", "public_score": "TBD"},
        {"output_file": "submission_14b_gb_age_sentinel_plus_agemissing.csv", "public_score": "TBD"},
        {"output_file": "submission_14c_histgb_raw_tabular_nan_native.csv", "public_score": "TBD"},
    ]


def _build_report(
    train: pd.DataFrame,
    submission_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
    histgb_strategy: str,
    histgb_error: str,
) -> str:
    overall_status = "PASS" if all(row["status"] == "PASS" for row in sanity_rows) else "FAIL"
    histgb_note = (
        'HistGB used `categorical_features="from_dtype"` with pandas categorical dtypes.'
        if histgb_strategy == "from_dtype"
        else "HistGB used the documented ordinal fallback because from_dtype failed."
    )
    if histgb_error:
        histgb_note += f" Fallback/error detail: `{histgb_error}`."

    lines = [
        "# 14 AgeMissing Frozen Checkpoint",
        "",
        "## Scope",
        "",
        "- frozen checkpoint for AgeMissing handling after step 13",
        "- candidates are fixed before any public score",
        "- full `train.csv` fitting is allowed",
        "- `test.csv` is used only for inference",
        "- no public score is used by the script",
        "",
        "## Method boundary",
        "",
        "- This is a frozen checkpoint, not feature acceptance.",
        "- No post-score tuning.",
        "- No micro-variants.",
        "- AgeBucket is not changed or reopened.",
        "- Broad `Title` remains closed.",
        "- No Master, Old, Mrs/Miss, Surname, target-derived group survival, or PassengerId corrections.",
        "- CatBoost / LightGBM / XGBoost are not reopened.",
        "- `gender_submission.csv` is not used as truth.",
        "- Test target is not used.",
        "",
        "## Fixed candidate list",
        "",
        _markdown_table(
            _fixed_candidate_rows(),
            ["candidate_id", "model", "features", "output_file", "reason"],
        ),
        "",
        "## Why checkpoint is allowed",
        "",
        "- `gb_raw_plus_agemissing_median` was `KEEP_CANDIDATE` in step 13 with cv_mean `0.828284`, 9 changed rows, rescue 5, kill 4, net +1.",
        "- `gb_raw_age_sentinel_plus_agemissing` was `KEEP_CANDIDATE` in step 13 with cv_mean `0.833890`, 34 changed rows, rescue 20, kill 14, net +6.",
        "- `histgb_raw_tabular_nan_native` is included once as a diagnostic native-NaN model lane with cv_mean `0.838359` and net +10 vs the GB baseline.",
        "- The candidate list is fixed before public score and must not be changed post-score.",
        "",
        "## Why excluded candidates are excluded",
        "",
        "- No `histgb_raw_plus_agemissing_nan_native` because it produced identical predictions to `histgb_raw_tabular_nan_native` in step 13.",
        "- No AgeBucket reopen.",
        "- No CatBoost/LGBM/XGB reopen.",
        "- No broad Title, Master fallback, Old buckets, Mrs/Miss, or Surname.",
        "",
        "## Exact preprocessing logic per candidate",
        "",
        _markdown_table(_preprocessing_rows(), ["candidate_id", "preprocessing_logic"]),
        "",
        f"- {histgb_note}",
        "",
        "## Model panel",
        "",
        _markdown_table(_model_panel_rows(histgb_strategy, histgb_error), MODEL_PANEL_COLUMNS),
        "",
        "## Submission diagnostics",
        "",
        f"- overall status: `{overall_status}`",
        f"- train rows: `{len(train)}`",
        "",
        _markdown_table(submission_rows, SUBMISSION_COLUMNS),
        "",
        "## Diff vs raw GB baseline submission",
        "",
        _markdown_table(diff_rows, DIFF_COLUMNS),
        "",
        "## Sanity checks",
        "",
        _markdown_table(sanity_rows, SANITY_COLUMNS),
        "",
        "## Public score placeholder",
        "",
        "Public score:",
        "- submission_14a_gb_raw_plus_agemissing_median.csv: TBD",
        "- submission_14b_gb_age_sentinel_plus_agemissing.csv: TBD",
        "- submission_14c_histgb_raw_tabular_nan_native.csv: TBD",
        "",
        _markdown_table(_public_score_rows(), ["output_file", "public_score"]),
        "",
        "## Decision rule after public score",
        "",
        "- If none beats current public baseline `raw_tabular / GradientBoostingClassifier = 0.79665`, close AgeMissing handling as public-transfer failed.",
        "- If one beats baseline, mark as checkpoint leader/candidate, but do not do row-level tuning.",
        "- If HistGB wins, mark it as new model-lane candidate, not as proof that GB feature engineering succeeded.",
        "- Do not use public result to create micro-variants.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    train = _add_agemissing(pd.read_csv(TRAIN_PATH))
    test = _add_agemissing(pd.read_csv(TEST_PATH))
    predictions: dict[str, pd.Series] = {}
    submission_rows: list[dict[str, object]] = []
    histgb_strategy = "from_dtype"
    histgb_error = ""

    for candidate in CANDIDATES:
        row, pred, effective_strategy, error = _fit_predict_candidate(
            candidate,
            train,
            test,
            histgb_strategy,
        )
        submission_rows.append(row)
        if pred is not None:
            predictions[candidate.candidate_id] = pred
        if candidate.model_name == "HistGradientBoostingClassifier":
            histgb_strategy = effective_strategy
            histgb_error = error

    diff_rows = _baseline_diff_rows(predictions)
    sanity_rows = _sanity_rows(submission_rows, test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(
            train,
            submission_rows,
            diff_rows,
            sanity_rows,
            histgb_strategy,
            histgb_error,
        ),
        encoding="utf-8",
    )

    all_passed = all(row["status"] == "PASS" for row in sanity_rows)
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"submission files: {len(list(SUBMISSIONS_DIR.glob('submission_14*.csv')))}")
    print(f"overall: {'PASS' if all_passed else 'FAIL'}")
    for row in submission_rows:
        print(
            f"{row['candidate_id']}: rows={row['rows']} pred_0={row['pred_0_count']} "
            f"pred_1={row['pred_1_count']} pred_1_rate={row['pred_1_rate']} "
            f"status={row['status']}"
        )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
