# 06 Title Feature Check

## Scope Boundary

- train-side CV only
- `train.csv` only
- `Survived` is used only as the target
- `add_clean_features()` is used only to create `Title`
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- same CV protocol and technical model parameters as `04_baseline`
- OOF diagnostics are train-side only
- no submission generation
- no Kaggle/public leaderboard use
- no `test.csv` scoring
- no test labels or row-level correctness checks
- no `gender_submission.csv` as truth
- no hyperparameter tuning
- no threshold tuning
- no final model selection
- no derived features other than `Title` are included in any feature set
- no gating, probability threshold changes, PassengerId overrides, or manual correction rules

## Active Model Lanes

| model | reason |
| --- | --- |
| GradientBoostingClassifier | active/deferred after `05_baseline_frozen_checkpoint` |
| SVC | active/deferred after `05_baseline_frozen_checkpoint` |
| CatBoostClassifier | active/deferred after `05_baseline_frozen_checkpoint` |

## Excluded Models

| model | reason |
| --- | --- |
| RandomForestClassifier | rejected by `05_baseline_frozen_checkpoint` public transfer |
| LGBMClassifier | rejected by `05_baseline_frozen_checkpoint` public transfer |
| HistGradientBoostingClassifier | rejected by `05_baseline_frozen_checkpoint` public transfer |
| XGBClassifier | not active after baseline checkpoint; no raw_tabular checkpoint lane |
| ExtraTreesClassifier | excluded before baseline checkpoint or not active baseline lanes |
| DecisionTreeClassifier | excluded before baseline checkpoint or not active baseline lanes |
| AdaBoostClassifier | excluded before baseline checkpoint or not active baseline lanes |
| LinearSVC | excluded before baseline checkpoint or not active baseline lanes |
| KNeighborsClassifier | excluded before baseline checkpoint or not active baseline lanes |
| GaussianNB | excluded before baseline checkpoint or not active baseline lanes |
| DummyClassifier | excluded before baseline checkpoint or not active baseline lanes |

## Feature Sets

| feature_set | features |
| --- | --- |
| f00_core | Sex, Pclass, Embarked |
| f00_core_plus_title | Sex, Pclass, Embarked, Title |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare |
| raw_plus_title | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, Title |

## CV Protocol

- splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- metric: `accuracy`
- identical precomputed CV split indices are reused for every model and feature set
- preprocessing is fitted inside each train fold through an sklearn `Pipeline`
- rows: `891` from `train.csv`

## Model Panel Used

| model | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| CatBoostClassifier | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |

## All Results

- overall status: `PASS`

| status | feature_set | model | preprocessing_mode | cv_mean | cv_std | cv_min | cv_max | fold_scores | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | f00_core | SVC | scaled_linear | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | GradientBoostingClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | CatBoostClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core_plus_title | SVC | scaled_linear | 0.820426 | 0.006167 | 0.808989 | 0.825843 | [0.821229, 0.820225, 0.808989, 0.825843, 0.825843] |  |
| ok | f00_core_plus_title | GradientBoostingClassifier | unscaled_tree | 0.81595 | 0.007933 | 0.804469 | 0.825843 | [0.804469, 0.820225, 0.808989, 0.825843, 0.820225] |  |
| ok | f00_core_plus_title | CatBoostClassifier | unscaled_tree | 0.817074 | 0.008809 | 0.804469 | 0.825843 | [0.804469, 0.820225, 0.808989, 0.825843, 0.825843] |  |
| ok | raw_tabular | SVC | scaled_linear | 0.821518 | 0.019504 | 0.792135 | 0.849162 | [0.849162, 0.808989, 0.792135, 0.831461, 0.825843] |  |
| ok | raw_tabular | GradientBoostingClassifier | unscaled_tree | 0.827161 | 0.020218 | 0.803371 | 0.859551 | [0.826816, 0.859551, 0.808989, 0.803371, 0.837079] |  |
| ok | raw_tabular | CatBoostClassifier | unscaled_tree | 0.835001 | 0.018107 | 0.803371 | 0.849162 | [0.849162, 0.848315, 0.803371, 0.825843, 0.848315] |  |
| ok | raw_plus_title | SVC | scaled_linear | 0.837254 | 0.018175 | 0.814607 | 0.865169 | [0.843575, 0.820225, 0.814607, 0.842697, 0.865169] |  |
| ok | raw_plus_title | GradientBoostingClassifier | unscaled_tree | 0.839489 | 0.010651 | 0.825843 | 0.854749 | [0.854749, 0.837079, 0.831461, 0.825843, 0.848315] |  |
| ok | raw_plus_title | CatBoostClassifier | unscaled_tree | 0.84286 | 0.010835 | 0.825843 | 0.854749 | [0.854749, 0.853933, 0.825843, 0.837079, 0.842697] |  |

## Paired comparison: `f00_core_plus_title` vs `f00_core`

| model | base_feature_set | candidate_feature_set | base_cv_mean | candidate_cv_mean | delta_candidate_minus_base | result |
| --- | --- | --- | --- | --- | --- | --- |
| SVC | f00_core | f00_core_plus_title | 0.811443 | 0.820426 | 0.008983 | improved |
| GradientBoostingClassifier | f00_core | f00_core_plus_title | 0.811443 | 0.81595 | 0.004507 | improved |
| CatBoostClassifier | f00_core | f00_core_plus_title | 0.811443 | 0.817074 | 0.005631 | improved |

## Paired comparison: `raw_plus_title` vs `raw_tabular`

| model | base_feature_set | candidate_feature_set | base_cv_mean | candidate_cv_mean | delta_candidate_minus_base | result |
| --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_tabular | raw_plus_title | 0.821518 | 0.837254 | 0.015736 | improved |
| GradientBoostingClassifier | raw_tabular | raw_plus_title | 0.827161 | 0.839489 | 0.012328 | improved |
| CatBoostClassifier | raw_tabular | raw_plus_title | 0.835001 | 0.84286 | 0.007859 | improved |

## OOF diagnostics: `f00_core_plus_title` vs `f00_core`

| model | base_feature_set | candidate_feature_set | changed_predictions | changed_pct | base_pred_1_count | candidate_pred_1_count | base_pred_1_rate | candidate_pred_1_rate | delta_pred_1_rate | rescue_count | kill_count | net_correct_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | f00_core | f00_core_plus_title | 32 | 3.59147 | 226 | 256 | 0.253648 | 0.287318 | 0.03367 | 20 | 12 | 8 |
| GradientBoostingClassifier | f00_core | f00_core_plus_title | 32 | 3.59147 | 226 | 242 | 0.253648 | 0.271605 | 0.017957 | 18 | 14 | 4 |
| CatBoostClassifier | f00_core | f00_core_plus_title | 31 | 3.479237 | 226 | 241 | 0.253648 | 0.270483 | 0.016835 | 18 | 13 | 5 |

## OOF diagnostics: `raw_plus_title` vs `raw_tabular`

| model | base_feature_set | candidate_feature_set | changed_predictions | changed_pct | base_pred_1_count | candidate_pred_1_count | base_pred_1_rate | candidate_pred_1_rate | delta_pred_1_rate | rescue_count | kill_count | net_correct_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_tabular | raw_plus_title | 20 | 2.244669 | 291 | 301 | 0.326599 | 0.337823 | 0.011223 | 17 | 3 | 14 |
| GradientBoostingClassifier | raw_tabular | raw_plus_title | 33 | 3.703704 | 294 | 303 | 0.329966 | 0.340067 | 0.010101 | 22 | 11 | 11 |
| CatBoostClassifier | raw_tabular | raw_plus_title | 23 | 2.581369 | 295 | 298 | 0.331089 | 0.334456 | 0.003367 | 15 | 8 | 7 |

## Summary

`f00_core_plus_title` vs `f00_core`:

- improved: `3`
- worsened: `0`
- tied: `0`
- best positive delta: `SVC (0.008983)`
- worst negative delta: `n/a`
- OOF changed prediction range: `3.479237%` to `3.59147%`
- best net_correct_delta: `SVC (8)`

`raw_plus_title` vs `raw_tabular`:

- improved: `3`
- worsened: `0`
- tied: `0`
- best positive delta: `SVC (0.015736)`
- worst negative delta: `n/a`
- OOF changed prediction range: `2.244669%` to `3.703704%`
- best net_correct_delta: `SVC (14)`

## Short interpretation

- On the core layer, `Title` helps in this controlled check (3 improved, 0 worsened, 0 tied).
- On the raw_tabular layer, `Title` helps in this controlled check (3 improved, 0 worsened, 0 tied).
- The OOF effect looks moderate by changed-prediction share across active lanes.
- This is feature-check and OOF diagnostic evidence only.
- This is not final model selection.
- No gating was applied.
