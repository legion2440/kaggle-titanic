# 13 AgeMissing Handling Check

## Scope

- controlled train-side CV/OOF check only
- only `train.csv` is read
- no submission was created
- no public score or Kaggle leaderboard use
- no `gender_submission.csv` as truth
- no test target or row-level test correctness

## Method boundary

- This is not feature acceptance.
- No post-score tuning.
- Frozen checkpoint reports are not changed.
- AgeBucket v1 is not being reopened here.
- Broad `Title` remains closed.
- No AgeBucket, Master, Old, Mrs/Miss, Surname, or PassengerId corrections are added.
- CatBoost / LightGBM / XGBoost are not reopened in this protocol.
- The main question is whether median-imputed `Age` is hurting GB.

## Why this follows AgeBucket v1

- AgeBucket v1 was checked honestly and improved the SVC lane on public transfer, but it did not beat the primary GB public baseline.
- Step 11 showed that removing raw `Age` made the model lanes noticeably worse.
- This check narrows the hypothesis: keep real known `Age`, then test whether explicit missingness or sentinel missing handling improves the GB lane.

## Variants and preprocessing strategy

| variant | model_name | feature_set | features | preprocessing_strategy | purpose |
| --- | --- | --- | --- | --- | --- |
| gb_raw_tabular_median | GradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | existing unscaled_tree preprocessing: numeric median imputation; categorical most_frequent + one-hot | baseline reference for current raw_tabular GB lane |
| gb_raw_plus_agemissing_median | GradientBoostingClassifier | raw_plus_agemissing | Sex, Pclass, Embarked, Age, AgeMissing, SibSp, Parch, Fare | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | check explicit missingness signal on top of median-imputed Age |
| gb_raw_age_sentinel_plus_agemissing | GradientBoostingClassifier | raw_plus_agemissing | Sex, Pclass, Embarked, Age, AgeMissing, SibSp, Parch, Fare | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | check whether median pseudo-age is hurting GB while preserving known Age |
| histgb_raw_tabular_nan_native | HistGradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | diagnostic native-NaN reference without explicit AgeMissing |
| histgb_raw_plus_agemissing_nan_native | HistGradientBoostingClassifier | raw_plus_agemissing | Sex, Pclass, Embarked, Age, AgeMissing, SibSp, Parch, Fare | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | diagnostic check for AgeMissing on top of native NaN handling |

## Model panel

| variant | model_class | package | package_version | preprocessing_strategy | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gb_raw_tabular_median | GradientBoostingClassifier | scikit-learn | 1.8.0 | existing unscaled_tree preprocessing: numeric median imputation; categorical most_frequent + one-hot | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| gb_raw_plus_agemissing_median | GradientBoostingClassifier | scikit-learn | 1.8.0 | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| gb_raw_age_sentinel_plus_agemissing | GradientBoostingClassifier | scikit-learn | 1.8.0 | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| histgb_raw_tabular_nan_native | HistGradientBoostingClassifier | scikit-learn | 1.8.0 | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | {"categorical_features": "from_dtype", "random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| histgb_raw_plus_agemissing_nan_native | HistGradientBoostingClassifier | scikit-learn | 1.8.0 | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | {"categorical_features": "from_dtype", "random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## CV/OOF summary table

- overall status: `PASS`
- rows: `891`
- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`
- metric: `accuracy`

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | gb_raw_tabular_median | raw_tabular | existing unscaled_tree preprocessing: numeric median imputation; categorical most_frequent + one-hot | 0.825701 | 0.021513 | 0.82716 | 0.329966 | gb_raw_tabular_median | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | gb_raw_plus_agemissing_median | raw_plus_agemissing | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | 0.822893 | 0.021439 | 0.828283 | 0.328844 | gb_raw_tabular_median | 9 | 1.010101 | 5 | 4 | 1 | DEFERRED |
| GradientBoostingClassifier | gb_raw_age_sentinel_plus_agemissing | raw_plus_agemissing | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | 0.818854 | 0.022204 | 0.832772 | 0.328844 | gb_raw_tabular_median | 33 | 3.703704 | 19 | 14 | 5 | DEFERRED |
| HistGradientBoostingClassifier | histgb_raw_tabular_nan_native | raw_tabular | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_BASELINE_REFERENCE |
| HistGradientBoostingClassifier | histgb_raw_plus_agemissing_nan_native | raw_plus_agemissing | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_ONLY |
| HistGradientBoostingClassifier | histgb_raw_tabular_nan_native | raw_tabular | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | 0.818743 | 0.024026 | 0.838384 | 0.356902 | gb_raw_tabular_median | 70 | 7.856341 | 40 | 30 | 10 | CROSS_REFERENCE_ONLY |

## Best rows

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | gb_raw_tabular_median | raw_tabular | existing unscaled_tree preprocessing: numeric median imputation; categorical most_frequent + one-hot | 0.825701 | 0.021513 | 0.82716 | 0.329966 | gb_raw_tabular_median | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | gb_raw_plus_agemissing_median | raw_plus_agemissing | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | 0.822893 | 0.021439 | 0.828283 | 0.328844 | gb_raw_tabular_median | 9 | 1.010101 | 5 | 4 | 1 | DEFERRED |
| GradientBoostingClassifier | gb_raw_age_sentinel_plus_agemissing | raw_plus_agemissing | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | 0.818854 | 0.022204 | 0.832772 | 0.328844 | gb_raw_tabular_median | 33 | 3.703704 | 19 | 14 | 5 | DEFERRED |
| HistGradientBoostingClassifier | histgb_raw_tabular_nan_native | raw_tabular | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_BASELINE_REFERENCE |
| HistGradientBoostingClassifier | histgb_raw_plus_agemissing_nan_native | raw_plus_agemissing | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_ONLY |

## Diagnostics

### GradientBoostingClassifier primary lane

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | gb_raw_tabular_median | raw_tabular | existing unscaled_tree preprocessing: numeric median imputation; categorical most_frequent + one-hot | 0.825701 | 0.021513 | 0.82716 | 0.329966 | gb_raw_tabular_median | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | gb_raw_plus_agemissing_median | raw_plus_agemissing | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | 0.822893 | 0.021439 | 0.828283 | 0.328844 | gb_raw_tabular_median | 9 | 1.010101 | 5 | 4 | 1 | DEFERRED |
| GradientBoostingClassifier | gb_raw_age_sentinel_plus_agemissing | raw_plus_agemissing | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | 0.818854 | 0.022204 | 0.832772 | 0.328844 | gb_raw_tabular_median | 33 | 3.703704 | 19 | 14 | 5 | DEFERRED |

### HistGradientBoostingClassifier native-missing diagnostic lane

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoostingClassifier | histgb_raw_tabular_nan_native | raw_tabular | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_BASELINE_REFERENCE |
| HistGradientBoostingClassifier | histgb_raw_plus_agemissing_nan_native | raw_plus_agemissing | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_ONLY |

### HistGB best vs GB median baseline cross-reference

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoostingClassifier | histgb_raw_tabular_nan_native | raw_tabular | native NaN lane: Age kept as NaN; pandas categorical dtypes with categorical_features="from_dtype" | 0.818743 | 0.024026 | 0.838384 | 0.356902 | gb_raw_tabular_median | 70 | 7.856341 | 40 | 30 | 10 | CROSS_REFERENCE_ONLY |

The cross-reference row is diagnostic only and is not automatic acceptance.

## Decision

| model_name | variant | feature_set | preprocessing_strategy | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_variant | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | gb_raw_plus_agemissing_median | raw_plus_agemissing | AgeMissing created before imputation; Age remains median-imputed; categorical handling matches baseline | 0.822893 | 0.021439 | 0.828283 | 0.328844 | gb_raw_tabular_median | 9 | 1.010101 | 5 | 4 | 1 | DEFERRED |
| GradientBoostingClassifier | gb_raw_age_sentinel_plus_agemissing | raw_plus_agemissing | AgeMissing created before fill; missing Age replaced with sentinel -1; other numeric missing handling remains safe | 0.818854 | 0.022204 | 0.832772 | 0.328844 | gb_raw_tabular_median | 33 | 3.703704 | 19 | 14 | 5 | DEFERRED |
| HistGradientBoostingClassifier | histgb_raw_plus_agemissing_nan_native | raw_plus_agemissing | native NaN lane: Age kept as NaN; AgeMissing added; categorical handling matches HistGB native-NaN reference | 0.818743 | 0.024026 | 0.838384 | 0.356902 | histgb_raw_tabular_nan_native | 0 | 0.0 | 0 | 0 | 0 | DIAGNOSTIC_ONLY |

Decision status definitions:

- `BASELINE_REFERENCE`: current comparison anchor.
- `KEEP_CANDIDATE`: beats its base by CV/OOF and has positive net OOF correctness delta.
- `DEFERRED`: close or mixed diagnostics, not enough for acceptance.
- `REJECTED`: CV and OOF diagnostics are worse than the base.
- `DIAGNOSTIC_ONLY`: native-missing diagnostic evidence, not a primary GB decision.

## Next step recommendation

- Do not checkpoint Age missing handling from this report alone. The primary GB missing-handling variants did not produce a clear keep signal.
- No submission should be created from this step.
- If a later frozen checkpoint is considered, it needs explicit review and a fixed candidate list before any public score.
