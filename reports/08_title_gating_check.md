# 08 Title Gating Check

## Scope Boundary

- train-side OOF only
- only `train.csv` is read
- `Survived` is used only as the target
- `add_clean_features()` is used only to create `Title`; only `Title` is copied from its output
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- same CV protocol and technical/default model parameters as `04_baseline`
- `predict_proba()` is used only for OOF probability blending
- fixed weight blending values only
- no submission generation
- no Kaggle/public leaderboard lookup
- no `test.csv` scoring or inference
- no test labels or row-level correctness checks
- no `gender_submission.csv` as truth
- no feature other than `Title` is added
- no hyperparameter tuning, model parameter tuning, PassengerId overrides, manual correction rules, or target-derived features

## Input Evidence

- `raw_tabular / GradientBoostingClassifier` is the current clean public baseline leader from step 05.
- unrestricted `raw_plus_title / GradientBoostingClassifier` was rejected for public transfer in step 07.
- this step checks conservative Title blending train-side only.

## CV / OOF Protocol

- splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- identical folds are used for base and title models
- preprocessing is fitted inside each train fold through an sklearn `Pipeline`
- base model: `raw_tabular / GradientBoostingClassifier`
- title model: `raw_plus_title / GradientBoostingClassifier`
- class-1 probabilities are aligned to `Survived == 1` before blending
- rows: `891` from `train.csv`

## Feature Sets

| feature_set | features |
| --- | --- |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare |
| raw_plus_title | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, Title |

## Model Used

| model | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Weight Grid

| weight |
| --- |
| 0.00 |
| 0.05 |
| 0.10 |
| 0.15 |
| 0.20 |
| 0.25 |
| 0.30 |
| 0.35 |
| 0.40 |
| 0.45 |
| 0.50 |
| 1.00 |

## OOF Results

- overall status: `PASS`

| weight | accuracy | accuracy_delta_vs_base | changed_predictions_vs_base | changed_pct_vs_base | base_pred_1_count | blend_pred_1_count | base_pred_1_rate | blend_pred_1_rate | delta_pred_1_rate | rescue_count | kill_count | net_correct_delta | flip_0_to_1 | flip_1_to_0 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.82716 | 0.0 | 0 | 0.0 | 294 | 294 | 0.329966 | 0.329966 | 0.0 | 0 | 0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| 0.05 | 0.828283 | 0.001122 | 3 | 0.3367 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 2 | 1 | 1 | 0 | 3 | CONSERVATIVE_CANDIDATE |
| 0.10 | 0.830527 | 0.003367 | 5 | 0.561167 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 4 | 1 | 3 | 1 | 4 | CONSERVATIVE_CANDIDATE |
| 0.15 | 0.830527 | 0.003367 | 5 | 0.561167 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 4 | 1 | 3 | 1 | 4 | CONSERVATIVE_CANDIDATE |
| 0.20 | 0.829405 | 0.002245 | 6 | 0.673401 | 294 | 292 | 0.329966 | 0.327722 | -0.002245 | 4 | 2 | 2 | 2 | 4 | CONSERVATIVE_CANDIDATE |
| 0.25 | 0.832772 | 0.005612 | 9 | 1.010101 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 7 | 2 | 5 | 3 | 6 | CAUTION_ZONE |
| 0.30 | 0.832772 | 0.005612 | 9 | 1.010101 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 7 | 2 | 5 | 3 | 6 | CAUTION_ZONE |
| 0.35 | 0.830527 | 0.003367 | 11 | 1.234568 | 294 | 293 | 0.329966 | 0.328844 | -0.001122 | 7 | 4 | 3 | 5 | 6 | CAUTION_ZONE |
| 0.40 | 0.83165 | 0.004489 | 12 | 1.346801 | 294 | 294 | 0.329966 | 0.329966 | 0.0 | 8 | 4 | 4 | 6 | 6 | CAUTION_ZONE |
| 0.45 | 0.832772 | 0.005612 | 15 | 1.683502 | 294 | 295 | 0.329966 | 0.331089 | 0.001122 | 10 | 5 | 5 | 8 | 7 | HIGH_TRANSFER_RISK |
| 0.50 | 0.832772 | 0.005612 | 17 | 1.907969 | 294 | 297 | 0.329966 | 0.333333 | 0.003367 | 11 | 6 | 5 | 10 | 7 | HIGH_TRANSFER_RISK |
| 1.00 | 0.839506 | 0.012346 | 33 | 3.703704 | 294 | 303 | 0.329966 | 0.340067 | 0.010101 | 22 | 11 | 11 | 21 | 12 | REJECTED_FULL_STRENGTH |

## Best OOF row by `accuracy`

| weight | accuracy | accuracy_delta_vs_base | changed_predictions_vs_base | changed_pct_vs_base | base_pred_1_count | blend_pred_1_count | base_pred_1_rate | blend_pred_1_rate | delta_pred_1_rate | rescue_count | kill_count | net_correct_delta | flip_0_to_1 | flip_1_to_0 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.839506 | 0.012346 | 33 | 3.703704 | 294 | 303 | 0.329966 | 0.340067 | 0.010101 | 22 | 11 | 11 | 21 | 12 | REJECTED_FULL_STRENGTH |

## Conservative candidates

| weight | accuracy | accuracy_delta_vs_base | changed_predictions_vs_base | changed_pct_vs_base | base_pred_1_count | blend_pred_1_count | base_pred_1_rate | blend_pred_1_rate | delta_pred_1_rate | rescue_count | kill_count | net_correct_delta | flip_0_to_1 | flip_1_to_0 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.828283 | 0.001122 | 3 | 0.3367 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 2 | 1 | 1 | 0 | 3 | CONSERVATIVE_CANDIDATE |
| 0.10 | 0.830527 | 0.003367 | 5 | 0.561167 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 4 | 1 | 3 | 1 | 4 | CONSERVATIVE_CANDIDATE |
| 0.15 | 0.830527 | 0.003367 | 5 | 0.561167 | 294 | 291 | 0.329966 | 0.326599 | -0.003367 | 4 | 1 | 3 | 1 | 4 | CONSERVATIVE_CANDIDATE |
| 0.20 | 0.829405 | 0.002245 | 6 | 0.673401 | 294 | 292 | 0.329966 | 0.327722 | -0.002245 | 4 | 2 | 2 | 2 | 4 | CONSERVATIVE_CANDIDATE |

## Transfer-risk notes

- This is train-side only.
- Flip count is treated as a risk heuristic, not hard truth.
- No submission was created.
- No public score was used.

## Short interpretation

- At least one conservative Title weight improves OOF over baseline; the best conservative row is `w=0.10` with a small flip count.
- The best OOF accuracy row is `w=1.00`, with status `REJECTED_FULL_STRENGTH`.
- A next frozen checkpoint is train-side justified only for the conservative `w=0.10` lane, subject to keeping it separate from final model selection.
- This is not final model selection.
