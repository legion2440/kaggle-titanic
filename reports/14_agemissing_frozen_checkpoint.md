# 14 AgeMissing Frozen Checkpoint

## Scope

- frozen checkpoint for AgeMissing handling after step 13
- candidates are fixed before any public score
- full `train.csv` fitting is allowed
- `test.csv` is used only for inference
- no public score is used by the script

## Method boundary

- This is a frozen checkpoint, not feature acceptance.
- No post-score tuning.
- No micro-variants.
- AgeBucket is not changed or reopened.
- Broad `Title` remains closed.
- No Master, Old, Mrs/Miss, Surname, target-derived group survival, or PassengerId corrections.
- CatBoost / LightGBM / XGBoost are not reopened.
- `gender_submission.csv` is not used as truth.
- Test target is not used.

## Fixed candidate list

| candidate_id | model | features | output_file | reason |
| --- | --- | --- | --- | --- |
| gb_raw_plus_agemissing_median | GradientBoostingClassifier | Sex, Pclass, Embarked, Age, AgeMissing, SibSp, Parch, Fare | submissions/submission_14a_gb_raw_plus_agemissing_median.csv | step 13 KEEP_CANDIDATE: cv_mean 0.828284, net OOF +1 |
| gb_raw_age_sentinel_plus_agemissing | GradientBoostingClassifier | Sex, Pclass, Embarked, Age, AgeMissing, SibSp, Parch, Fare | submissions/submission_14b_gb_age_sentinel_plus_agemissing.csv | step 13 KEEP_CANDIDATE: cv_mean 0.833890, net OOF +6 |
| histgb_raw_tabular_nan_native | HistGradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | submissions/submission_14c_histgb_raw_tabular_nan_native.csv | step 13 diagnostic native-NaN lane: cv_mean 0.838359, net OOF +10 vs GB baseline |

## Why checkpoint is allowed

- `gb_raw_plus_agemissing_median` was `KEEP_CANDIDATE` in step 13 with cv_mean `0.828284`, 9 changed rows, rescue 5, kill 4, net +1.
- `gb_raw_age_sentinel_plus_agemissing` was `KEEP_CANDIDATE` in step 13 with cv_mean `0.833890`, 34 changed rows, rescue 20, kill 14, net +6.
- `histgb_raw_tabular_nan_native` is included once as a diagnostic native-NaN model lane with cv_mean `0.838359` and net +10 vs the GB baseline.
- The candidate list is fixed before public score and must not be changed post-score.

## Why excluded candidates are excluded

- No `histgb_raw_plus_agemissing_nan_native` because it produced identical predictions to `histgb_raw_tabular_nan_native` in step 13.
- No AgeBucket reopen.
- No CatBoost/LGBM/XGB reopen.
- No broad Title, Master fallback, Old buckets, Mrs/Miss, or Surname.

## Exact preprocessing logic per candidate

| candidate_id | preprocessing_logic |
| --- | --- |
| gb_raw_plus_agemissing_median | AgeMissing created before imputation; Age remains median-imputed through existing unscaled_tree preprocessing; categorical most_frequent + one-hot |
| gb_raw_age_sentinel_plus_agemissing | AgeMissing created before fill; missing Age replaced with sentinel -1; Age has no missing values after sentinel replacement; other numeric missing handling remains safe; categorical most_frequent + one-hot |
| histgb_raw_tabular_nan_native | Age kept as NaN; no Age median imputation; Sex, Pclass, and Embarked converted to pandas categorical dtype for categorical_features="from_dtype" |

- HistGB used `categorical_features="from_dtype"` with pandas categorical dtypes.

## Model panel

| candidate_id | model_class | package | package_version | preprocessing_logic | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gb_raw_plus_agemissing_median | GradientBoostingClassifier | scikit-learn | 1.8.0 | AgeMissing created before imputation; Age remains median-imputed through existing unscaled_tree preprocessing; categorical most_frequent + one-hot | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| gb_raw_age_sentinel_plus_agemissing | GradientBoostingClassifier | scikit-learn | 1.8.0 | AgeMissing created before fill; missing Age replaced with sentinel -1; Age has no missing values after sentinel replacement; other numeric missing handling remains safe; categorical most_frequent + one-hot | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| histgb_raw_tabular_nan_native | HistGradientBoostingClassifier | scikit-learn | 1.8.0 | Age kept as NaN; no Age median imputation; Sex, Pclass, and Embarked converted to pandas categorical dtype for categorical_features="from_dtype" | {"categorical_features": "from_dtype", "random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Submission diagnostics

- overall status: `PASS`
- train rows: `891`

| candidate_id | output_file | rows | pred_0_count | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- |
| gb_raw_plus_agemissing_median | submissions/submission_14a_gb_raw_plus_agemissing_median.csv | 418 | 271 | 147 | 0.351675 | PASS |
| gb_raw_age_sentinel_plus_agemissing | submissions/submission_14b_gb_age_sentinel_plus_agemissing.csv | 418 | 279 | 139 | 0.332536 | PASS |
| histgb_raw_tabular_nan_native | submissions/submission_14c_histgb_raw_tabular_nan_native.csv | 418 | 272 | 146 | 0.349282 | PASS |

## Diff vs raw GB baseline submission

| baseline_candidate_id | candidate_id | baseline_file | changed_predictions | changed_pct | baseline_pred_1_count | candidate_pred_1_count | baseline_pred_1_rate | candidate_pred_1_rate | delta_pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular__GradientBoostingClassifier | gb_raw_plus_agemissing_median | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 6 | 1.435407 | 141 | 147 | 0.337321 | 0.351675 | 0.014354 | PASS |
| raw_tabular__GradientBoostingClassifier | gb_raw_age_sentinel_plus_agemissing | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 16 | 3.827751 | 141 | 139 | 0.337321 | 0.332536 | -0.004785 | PASS |
| raw_tabular__GradientBoostingClassifier | histgb_raw_tabular_nan_native | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 31 | 7.416268 | 141 | 146 | 0.337321 | 0.349282 | 0.011962 | PASS |

## Sanity checks

| check | status | detail |
| --- | --- | --- |
| exactly 3 submission_14 files generated | PASS | count=3; extra=none; missing=none |
| gb_raw_plus_agemissing_median: 418 rows | PASS | rows=418 |
| gb_raw_plus_agemissing_median: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| gb_raw_plus_agemissing_median: PassengerId order matches data/test.csv | PASS | order checked |
| gb_raw_plus_agemissing_median: Survived values only 0/1 | PASS | values=[0, 1] |
| gb_raw_plus_agemissing_median: no duplicate PassengerId | PASS | duplicates checked |
| gb_raw_age_sentinel_plus_agemissing: 418 rows | PASS | rows=418 |
| gb_raw_age_sentinel_plus_agemissing: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| gb_raw_age_sentinel_plus_agemissing: PassengerId order matches data/test.csv | PASS | order checked |
| gb_raw_age_sentinel_plus_agemissing: Survived values only 0/1 | PASS | values=[0, 1] |
| gb_raw_age_sentinel_plus_agemissing: no duplicate PassengerId | PASS | duplicates checked |
| histgb_raw_tabular_nan_native: 418 rows | PASS | rows=418 |
| histgb_raw_tabular_nan_native: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| histgb_raw_tabular_nan_native: PassengerId order matches data/test.csv | PASS | order checked |
| histgb_raw_tabular_nan_native: Survived values only 0/1 | PASS | values=[0, 1] |
| histgb_raw_tabular_nan_native: no duplicate PassengerId | PASS | duplicates checked |
| all candidate predictions succeeded | PASS | failed=0 |

## Public score placeholder

Public score:
- submission_14a_gb_raw_plus_agemissing_median.csv: TBD
- submission_14b_gb_age_sentinel_plus_agemissing.csv: TBD
- submission_14c_histgb_raw_tabular_nan_native.csv: TBD

| output_file | public_score |
| --- | --- |
| submission_14a_gb_raw_plus_agemissing_median.csv | TBD |
| submission_14b_gb_age_sentinel_plus_agemissing.csv | TBD |
| submission_14c_histgb_raw_tabular_nan_native.csv | TBD |

## Decision rule after public score

- If none beats current public baseline `raw_tabular / GradientBoostingClassifier = 0.79665`, close AgeMissing handling as public-transfer failed.
- If one beats baseline, mark as checkpoint leader/candidate, but do not do row-level tuning.
- If HistGB wins, mark it as new model-lane candidate, not as proof that GB feature engineering succeeded.
- Do not use public result to create micro-variants.
