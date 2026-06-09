from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH
from scripts.features import RAW_TABULAR
from scripts.preprocessing import make_preprocessor


REPORT_PATH = REPORTS_DIR / "19a_gb_raw_tabular_tuning_stage1_stage2.md"

MODEL_NAME = "GradientBoostingClassifier"
BASELINE_VARIANT = "raw_tabular"
BASELINE_CANDIDATE_ID = "default_raw_tabular_gb"
CURRENT_PUBLIC_LEADER_SUBMISSION = "submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv"
CURRENT_PUBLIC_LEADER_SCORE = "0.79904"

BASELINE_FEATURES = list(RAW_TABULAR)

DEFAULT_PARAMS: dict[str, object] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "subsample": 1.0,
    "max_features": None,
    "random_state": RANDOM_STATE,
    "loss": "log_loss",
}

STAGE1_FIXED: dict[str, object] = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "ccp_alpha": 0.0,
}

STAGE1_GRID = {
    "max_depth": [1, 2, 3],
    "min_samples_leaf": [1, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": [None, "sqrt"],
}

STAGE2_PAIRS = [
    {"learning_rate": 0.03, "n_estimators": 300},
    {"learning_rate": 0.05, "n_estimators": 200},
    {"learning_rate": 0.07, "n_estimators": 150},
    {"learning_rate": 0.10, "n_estimators": 100},
    {"learning_rate": 0.15, "n_estimators": 70},
    {"learning_rate": 0.20, "n_estimators": 50},
]

RESULT_COLUMNS = [
    "stage",
    "candidate_id",
    "params",
    "cv_mean",
    "cv_std",
    "oof_accuracy",
    "oof_accuracy_delta_vs_default_raw_tabular",
    "oof_changed_rows_vs_default_raw_tabular",
    "oof_0_to_1",
    "oof_1_to_0",
    "rescue",
    "kill",
    "net",
    "oof_pred_1_count",
    "oof_pred_1_rate",
    "test_changed_rows_vs_default_raw_tabular_full_fit",
    "test_0_to_1",
    "test_1_to_0",
    "test_pred_1_count",
    "test_pred_1_rate",
    "diagnostic_status",
]

STAGE1_SELECTION_COLUMNS = [
    "rank",
    "candidate_id",
    "params",
    "oof_accuracy",
    "cv_std",
    "oof_changed_rows_vs_default_raw_tabular",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
]

SHORTLIST_COLUMNS = [
    "candidate_id",
    "stage",
    "params",
    "oof_accuracy",
    "delta_vs_default",
    "rescue_kill_net",
    "oof_changed_rows",
    "test_changed_rows",
    "test_pred_1_count",
    "test_pred_1_rate",
    "diagnostic_status",
]

MODEL_PANEL_COLUMNS = [
    "model_class",
    "package",
    "package_version",
    "preprocessing_mode",
    "explicit_technical_params",
    "default_resolved_params",
]


@dataclass(frozen=True)
class CandidateSpec:
    stage: str
    candidate_id: str
    params: dict[str, object]
    structure_key: tuple[int, int, int, object]


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


def _max_features_label(value: object) -> str:
    return "none" if value is None else str(value)


def _max_features_sort(value: object) -> int:
    return 0 if value is None else 1


def _params_json(params: dict[str, object]) -> str:
    return _json_dumps(params)


def _structure_key(params: dict[str, object]) -> tuple[int, int, int, object]:
    return (
        int(params["max_depth"]),
        int(params["min_samples_leaf"]),
        int(params["min_samples_split"]),
        params["max_features"],
    )


def _structure_label(params: dict[str, object]) -> str:
    return (
        f"d{int(params['max_depth'])}"
        f"_leaf{int(params['min_samples_leaf'])}"
        f"_split{int(params['min_samples_split'])}"
        f"_mf{_max_features_label(params['max_features'])}"
    )


def _lr_label(value: object) -> str:
    return str(value).replace(".", "p")


def _make_stage1_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for max_depth, min_samples_leaf, min_samples_split, max_features in product(
        STAGE1_GRID["max_depth"],
        STAGE1_GRID["min_samples_leaf"],
        STAGE1_GRID["min_samples_split"],
        STAGE1_GRID["max_features"],
    ):
        params = {
            **STAGE1_FIXED,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "random_state": RANDOM_STATE,
            "loss": "log_loss",
        }
        candidate_id = "stage1_" + _structure_label(params)
        specs.append(CandidateSpec("stage1", candidate_id, params, _structure_key(params)))
    return specs


def _make_stage2_specs(selected_stage1_rows: list[dict[str, object]]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for row in selected_stage1_rows:
        structure = {
            "max_depth": int(row["max_depth"]),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "min_samples_split": int(row["min_samples_split"]),
            "max_features": None if row["max_features"] == "" else row["max_features"],
        }
        for pair in STAGE2_PAIRS:
            params = {
                **structure,
                **pair,
                "subsample": 1.0,
                "ccp_alpha": 0.0,
                "random_state": RANDOM_STATE,
                "loss": "log_loss",
            }
            candidate_id = (
                "stage2_"
                + _structure_label(params)
                + f"_lr{_lr_label(params['learning_rate'])}"
                + f"_n{int(params['n_estimators'])}"
            )
            specs.append(CandidateSpec("stage2", candidate_id, params, _structure_key(params)))
    return specs


def _build_estimator(params: dict[str, object]) -> Pipeline:
    model = GradientBoostingClassifier(**params)
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor("unscaled_tree", BASELINE_FEATURES)),
            ("model", model),
        ]
    )


def _evaluate_oof(
    train: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, object],
) -> dict[str, object]:
    fold_scores: list[float] = []
    oof = np.full(len(train), -1, dtype=int)

    for i, (train_idx, valid_idx) in enumerate(splits):
        estimator = _build_estimator(params)
        estimator.fit(train[BASELINE_FEATURES].iloc[train_idx], y.iloc[train_idx])
        fold_pred = estimator.predict(train[BASELINE_FEATURES].iloc[valid_idx]).astype(int)
        if i < 5:
            oof[valid_idx] = fold_pred
        fold_scores.append(float((fold_pred == y.iloc[valid_idx].to_numpy()).mean()))

    if (oof < 0).any():
        raise RuntimeError("OOF prediction assignment incomplete")

    return {
        "fold_scores": fold_scores,
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_accuracy": float((oof == y.to_numpy()).mean()),
        "oof": oof,
    }


def _fit_full_predict(train: pd.DataFrame, test: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    estimator = _build_estimator(params)
    estimator.fit(train[BASELINE_FEATURES], train[TARGET].astype(int))
    return estimator.predict(test[BASELINE_FEATURES]).astype(int)


def _diagnostic_status(delta: float, net: int) -> str:
    if delta > 0 and net > 0:
        return "OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    if delta < 0 or net < 0:
        return "OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN"
    return "OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN"


def _row_from_result(
    spec: CandidateSpec,
    result: dict[str, object],
    test_pred: np.ndarray,
    y: pd.Series,
    baseline_oof: np.ndarray,
    baseline_oof_accuracy: float,
    baseline_test_pred: np.ndarray,
) -> dict[str, object]:
    oof = result["oof"]
    y_values = y.to_numpy()
    changed = oof != baseline_oof
    test_changed = test_pred != baseline_test_pred
    baseline_correct = baseline_oof == y_values
    candidate_correct = oof == y_values
    rescue = int((~baseline_correct & candidate_correct).sum())
    kill = int((baseline_correct & ~candidate_correct).sum())
    net = rescue - kill
    delta = float(result["oof_accuracy"]) - baseline_oof_accuracy
    row = {
        "stage": spec.stage,
        "candidate_id": spec.candidate_id,
        "params": _params_json(spec.params),
        "cv_mean": _round_float(result["cv_mean"]),
        "cv_std": _round_float(result["cv_std"]),
        "oof_accuracy": _round_float(result["oof_accuracy"]),
        "oof_accuracy_delta_vs_default_raw_tabular": _round_float(delta),
        "oof_changed_rows_vs_default_raw_tabular": int(changed.sum()),
        "oof_0_to_1": int(((baseline_oof == 0) & (oof == 1)).sum()),
        "oof_1_to_0": int(((baseline_oof == 1) & (oof == 0)).sum()),
        "rescue": rescue,
        "kill": kill,
        "net": net,
        "oof_pred_1_count": int((oof == 1).sum()),
        "oof_pred_1_rate": _round_float(float((oof == 1).mean())),
        "test_changed_rows_vs_default_raw_tabular_full_fit": int(test_changed.sum()),
        "test_0_to_1": int(((baseline_test_pred == 0) & (test_pred == 1)).sum()),
        "test_1_to_0": int(((baseline_test_pred == 1) & (test_pred == 0)).sum()),
        "test_pred_1_count": int((test_pred == 1).sum()),
        "test_pred_1_rate": _round_float(float((test_pred == 1).mean())),
        "diagnostic_status": _diagnostic_status(delta, net),
        "max_depth": int(spec.params["max_depth"]),
        "min_samples_leaf": int(spec.params["min_samples_leaf"]),
        "min_samples_split": int(spec.params["min_samples_split"]),
        "max_features": "" if spec.params["max_features"] is None else spec.params["max_features"],
        "learning_rate": float(spec.params["learning_rate"]),
        "n_estimators": int(spec.params["n_estimators"]),
    }
    return row


def _stage1_sort_key(row: dict[str, object]) -> tuple[float, float, int, int, int, int, int]:
    return (
        -float(row["oof_accuracy"]),
        float(row["cv_std"]),
        int(row["oof_changed_rows_vs_default_raw_tabular"]),
        int(row["max_depth"]),
        -int(row["min_samples_leaf"]),
        -int(row["min_samples_split"]),
        _max_features_sort(None if row["max_features"] == "" else row["max_features"]),
    )


def _best_oof_sort_key(row: dict[str, object]) -> tuple[float, float, int, str]:
    return (
        -float(row["oof_accuracy"]),
        float(row["cv_std"]),
        int(row["oof_changed_rows_vs_default_raw_tabular"]),
        str(row["candidate_id"]),
    )


def _positive_net_sort_key(row: dict[str, object]) -> tuple[int, float, float, str]:
    return (
        -int(row["net"]),
        -float(row["oof_accuracy"]),
        float(row["cv_std"]),
        str(row["candidate_id"]),
    )


def _simplicity_sort_key(row: dict[str, object]) -> tuple[int, int, int, int, int, float, str]:
    return (
        int(row["max_depth"]),
        -int(row["min_samples_leaf"]),
        -int(row["min_samples_split"]),
        _max_features_sort(None if row["max_features"] == "" else row["max_features"]),
        int(row["n_estimators"]),
        float(row["learning_rate"]),
        str(row["candidate_id"]),
    )


def _select_stage1_top10(stage1_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_rows = sorted(stage1_rows, key=_stage1_sort_key)
    selected: list[dict[str, object]] = []
    seen = set()
    for row in sorted_rows:
        key = (
            int(row["max_depth"]),
            int(row["min_samples_leaf"]),
            int(row["min_samples_split"]),
            row["max_features"],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == 10:
            break
    return selected


def _shortlist_rows(
    default_row: dict[str, object],
    stage1_rows: list[dict[str, object]],
    stage2_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    all_rows = [*stage1_rows, *stage2_rows]
    positive_rows = [row for row in all_rows if float(row["oof_accuracy_delta_vs_default_raw_tabular"]) > 0 and int(row["net"]) > 0]
    selected = [default_row]
    selected.append(sorted(stage1_rows, key=_best_oof_sort_key)[0])
    selected.append(sorted(stage2_rows, key=_best_oof_sort_key)[0])
    if positive_rows:
        selected.append(sorted(positive_rows, key=_positive_net_sort_key)[0])
        selected.append(sorted(positive_rows, key=_simplicity_sort_key)[0])

    deduped: list[dict[str, object]] = []
    seen_ids = set()
    for row in selected:
        if row["candidate_id"] in seen_ids:
            continue
        seen_ids.add(row["candidate_id"])
        deduped.append(row)

    return [_shortlist_row(row) for row in deduped]


def _shortlist_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_id": row["candidate_id"],
        "stage": row["stage"],
        "params": row["params"],
        "oof_accuracy": row["oof_accuracy"],
        "delta_vs_default": row["oof_accuracy_delta_vs_default_raw_tabular"],
        "rescue_kill_net": f"{row['rescue']} / {row['kill']} / {row['net']}",
        "oof_changed_rows": row["oof_changed_rows_vs_default_raw_tabular"],
        "test_changed_rows": row["test_changed_rows_vs_default_raw_tabular_full_fit"],
        "test_pred_1_count": row["test_pred_1_count"],
        "test_pred_1_rate": row["test_pred_1_rate"],
        "diagnostic_status": row["diagnostic_status"],
    }


def _stage1_selection_rows(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for rank, row in enumerate(selected_rows, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "params": row["params"],
                "oof_accuracy": row["oof_accuracy"],
                "cv_std": row["cv_std"],
                "oof_changed_rows_vs_default_raw_tabular": row["oof_changed_rows_vs_default_raw_tabular"],
                "max_depth": row["max_depth"],
                "min_samples_leaf": row["min_samples_leaf"],
                "min_samples_split": row["min_samples_split"],
                "max_features": "None" if row["max_features"] == "" else row["max_features"],
            }
        )
    return rows


def _model_panel_rows() -> list[dict[str, object]]:
    model = GradientBoostingClassifier(**DEFAULT_PARAMS)
    return [
        {
            "model_class": MODEL_NAME,
            "package": "scikit-learn",
            "package_version": baseline04._package_version("scikit-learn"),
            "preprocessing_mode": "unscaled_tree",
            "explicit_technical_params": _params_json(DEFAULT_PARAMS),
            "default_resolved_params": _json_dumps(model.get_params(deep=False)),
        }
    ]


def _report_lines(
    default_row: dict[str, object],
    stage1_rows: list[dict[str, object]],
    selected_stage1_rows: list[dict[str, object]],
    stage2_rows: list[dict[str, object]],
    shortlist_rows: list[dict[str, object]],
) -> list[str]:
    all_candidate_rows = [*stage1_rows, *stage2_rows]
    positive_rows = [
        row
        for row in all_candidate_rows
        if float(row["oof_accuracy_delta_vs_default_raw_tabular"]) > 0 and int(row["net"]) > 0
    ]
    best_oof = sorted(all_candidate_rows, key=_best_oof_sort_key)[0]
    interpretation = (
        "No train-side OOF-positive GB tuning candidate was found in Stage 1 + Stage 2. "
        "Public transfer remains unknown. No submission was created."
        if not positive_rows
        else "One or more train-side OOF-positive GB tuning candidates were found. "
        "Public transfer remains unknown. Submission decision is separate and no submission was created."
    )

    return [
        "# 19A GB Raw Tabular Tuning Stage 1 + Stage 2",
        "",
        "## Purpose",
        "",
        "This step runs train-side / OOF tuning diagnostics for `raw_tabular / GradientBoostingClassifier` only.",
        "",
        f"Current public leader remains a frozen benchmark: `{CURRENT_PUBLIC_LEADER_SUBMISSION}`, public score `{CURRENT_PUBLIC_LEADER_SCORE}`. Public score is not used for parameter selection.",
        "",
        "## Method boundary",
        "",
        "- Feature set is unchanged: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`.",
        "- No new features, `SurnameSurvival`, CabinKnown gate, post-processing rules, threshold tuning, PassengerId rules, test labels, or public score are used.",
        "- CV strategy matches previous GB checks: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`.",
        "- Diagnostics are train-side OOF only; public transfer remains unknown and submission decision is separate.",
        "- No submission was created.",
        "",
        "## Baseline params and baseline metrics",
        "",
        _markdown_table([default_row], RESULT_COLUMNS),
        "",
        "Baseline default params:",
        "",
        f"`{_params_json(DEFAULT_PARAMS)}`",
        "",
        "## Model panel",
        "",
        _markdown_table(_model_panel_rows(), MODEL_PANEL_COLUMNS),
        "",
        "## Stage 1 grid definition",
        "",
        "- Fixed: `learning_rate=0.1`, `n_estimators=100`, `subsample=1.0`, `ccp_alpha=0.0`.",
        "- Grid: `max_depth=[1,2,3]`, `min_samples_leaf=[1,3,5,10]`, `min_samples_split=[2,5,10]`, `max_features=[None,\"sqrt\"]`.",
        f"- Candidate count: `{len(stage1_rows)}`.",
        "",
        "## Stage 1 full results table",
        "",
        _markdown_table(stage1_rows, RESULT_COLUMNS),
        "",
        "## Stage 1 top 10 selected structures",
        "",
        "Selection sort: higher OOF accuracy, lower CV std, lower changed rows vs default, then simpler structure.",
        "",
        _markdown_table(_stage1_selection_rows(selected_stage1_rows), STAGE1_SELECTION_COLUMNS),
        "",
        "## Stage 2 paired learning_rate / n_estimators definition",
        "",
        "- Stage 2 uses only the selected top 10 Stage 1 structures.",
        "- Paired combinations only: `(0.03,300)`, `(0.05,200)`, `(0.07,150)`, `(0.10,100)`, `(0.15,70)`, `(0.20,50)`.",
        "- No Cartesian product of `learning_rate` and `n_estimators`.",
        f"- Candidate count: `{len(stage2_rows)}`.",
        "",
        "## Stage 2 full results table",
        "",
        _markdown_table(stage2_rows, RESULT_COLUMNS),
        "",
        "## Final train-side candidate shortlist",
        "",
        _markdown_table(shortlist_rows, SHORTLIST_COLUMNS),
        "",
        "## Test prediction audit for top candidates",
        "",
        "The shortlist table includes full-fit test prediction audit columns: changed rows vs default full-fit prediction, predicted survivor count, and predicted survivor rate. Test labels are not used.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        f"Best OOF candidate in Stage 1 + Stage 2: `{best_oof['candidate_id']}` with OOF accuracy `{best_oof['oof_accuracy']}`, delta `{best_oof['oof_accuracy_delta_vs_default_raw_tabular']}`, rescue/kill/net `{best_oof['rescue']} / {best_oof['kill']} / {best_oof['net']}`.",
        "",
        "Diagnostic statuses are train-side diagnostics only:",
        "",
        "- `OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN`: OOF delta > 0 and net > 0.",
        "- `OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN`: OOF delta < 0 or net < 0.",
        "- `OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN`: otherwise.",
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


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].astype(int)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(np.zeros(len(train)), y))
    oof_splits = splits[:5]

    baseline_spec = CandidateSpec("default", BASELINE_CANDIDATE_ID, dict(DEFAULT_PARAMS), _structure_key(DEFAULT_PARAMS))
    baseline_result = _evaluate_oof(train, y, splits, DEFAULT_PARAMS)
    baseline_test_pred = _fit_full_predict(train, test, DEFAULT_PARAMS)
    baseline_oof = baseline_result["oof"]
    baseline_accuracy = float(baseline_result["oof_accuracy"])
    default_row = _row_from_result(
        baseline_spec,
        baseline_result,
        baseline_test_pred,
        y,
        baseline_oof,
        baseline_accuracy,
        baseline_test_pred,
    )

    stage1_rows: list[dict[str, object]] = []
    for spec in _make_stage1_specs():
        result = _evaluate_oof(train, y, splits, spec.params)
        test_pred = _fit_full_predict(train, test, spec.params)
        stage1_rows.append(
            _row_from_result(spec, result, test_pred, y, baseline_oof, baseline_accuracy, baseline_test_pred)
        )

    selected_stage1_rows = _select_stage1_top10(stage1_rows)
    stage2_rows: list[dict[str, object]] = []
    for spec in _make_stage2_specs(selected_stage1_rows):
        result = _evaluate_oof(train, y, splits, spec.params)
        test_pred = _fit_full_predict(train, test, spec.params)
        stage2_rows.append(
            _row_from_result(spec, result, test_pred, y, baseline_oof, baseline_accuracy, baseline_test_pred)
        )

    shortlist_rows = _shortlist_rows(default_row, stage1_rows, stage2_rows)
    REPORT_PATH.write_text(
        "\n".join(_report_lines(default_row, stage1_rows, selected_stage1_rows, stage2_rows, shortlist_rows)),
        encoding="utf-8",
    )

    all_rows = [*stage1_rows, *stage2_rows]
    best_oof = sorted(all_rows, key=_best_oof_sort_key)[0]
    positive_count = sum(
        1
        for row in all_rows
        if float(row["oof_accuracy_delta_vs_default_raw_tabular"]) > 0 and int(row["net"]) > 0
    )

    print(f"wrote {_relative(REPORT_PATH)}")
    print(f"stage1_candidates={len(stage1_rows)}")
    print(f"stage2_candidates={len(stage2_rows)}")
    print(
        "best_oof={candidate_id} stage={stage} oof={oof} delta={delta} rescue={rescue} kill={kill} net={net}".format(
            candidate_id=best_oof["candidate_id"],
            stage=best_oof["stage"],
            oof=best_oof["oof_accuracy"],
            delta=best_oof["oof_accuracy_delta_vs_default_raw_tabular"],
            rescue=best_oof["rescue"],
            kill=best_oof["kill"],
            net=best_oof["net"],
        )
    )
    print(f"oof_positive_candidates={positive_count}")
    print("no submission was created")


if __name__ == "__main__":
    main()
