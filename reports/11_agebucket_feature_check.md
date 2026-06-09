# 11 AgeBucket Feature Check

## Scope

- train-side CV/OOF check only
- only `train.csv` is read
- no submission generation
- no Kaggle/public leaderboard use
- no `gender_submission.csv` as truth
- no test target or row-level test correctness
- no target-derived family/group survival

## Method boundary

- This is not feature acceptance.
- Broad Title remains closed.
- Master fallback was intentionally skipped.
- Old buckets were intentionally skipped.
- `AgeBucket` v1 is built locally for this controlled check only.
- `scripts/preprocessing.py` only recognizes `AgeBucket` as a categorical column for this check.

## Feature sets

| feature_set | features |
| --- | --- |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare |
| raw_no_age | Sex, Pclass, Embarked, SibSp, Parch, Fare |
| raw_no_age_no_sex_plus_agebucket_v1 | Pclass, Embarked, SibSp, Parch, Fare, AgeBucket |

## AgeBucket v1 mapping

```python
if Age is missing and Sex == "female":
    AgeBucket = "AgeMissingFemale"
elif Age is missing and Sex == "male":
    AgeBucket = "AgeMissingMale"
elif Sex == "female" and Age < 14:
    AgeBucket = "ChildFemale"
elif Sex == "male" and Age < 14:
    AgeBucket = "ChildMale"
elif Sex == "female":
    AgeBucket = "AdultFemale"
else:
    AgeBucket = "AdultMale"
```

Skipped by design: `OldFemale`, `OldMale`, `Master`, `Mrs`, `Miss`, `Surname`, broad `Title`, and PassengerId corrections.

## Model panel

| model | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| CatBoostClassifier | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |

## CV/OOF summary table

- overall status: `PASS`
- rows: `891`
- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_tabular | 0.821549 | 0.024288 | 0.821549 | 0.326599 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| SVC | raw_no_age | 0.799445 | 0.025304 | 0.805836 | 0.315376 | raw_tabular | 40 | 4.489338 | 13 | 27 | -14 | RAW_AGE_REMOVAL_HURTS |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_tabular | 22 | 2.469136 | 14 | 8 | 6 | KEEP_CANDIDATE |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_no_age | 38 | 4.264871 | 29 | 9 | 20 | KEEP_CANDIDATE |
| GradientBoostingClassifier | raw_tabular | 0.825701 | 0.021513 | 0.82716 | 0.329966 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | raw_no_age | 0.806179 | 0.024157 | 0.808081 | 0.333333 | raw_tabular | 71 | 7.968575 | 27 | 44 | -17 | RAW_AGE_REMOVAL_HURTS |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_tabular | 70 | 7.856341 | 33 | 37 | -4 | DEFERRED |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_no_age | 79 | 8.866442 | 46 | 33 | 13 | DEFERRED |
| CatBoostClassifier | raw_tabular | 0.828402 | 0.02515 | 0.835017 | 0.331089 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| CatBoostClassifier | raw_no_age | 0.802252 | 0.027531 | 0.805836 | 0.317621 | raw_tabular | 70 | 7.856341 | 22 | 48 | -26 | RAW_AGE_REMOVAL_HURTS |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_tabular | 47 | 5.274972 | 20 | 27 | -7 | DEFERRED |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_no_age | 55 | 6.17284 | 37 | 18 | 19 | DEFERRED |

## Best rows by CV mean

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoostClassifier | raw_tabular | 0.828402 | 0.02515 | 0.835017 | 0.331089 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | raw_tabular | 0.825701 | 0.021513 | 0.82716 | 0.329966 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_tabular | 22 | 2.469136 | 14 | 8 | 6 | KEEP_CANDIDATE |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_tabular | 47 | 5.274972 | 20 | 27 | -7 | DEFERRED |
| SVC | raw_tabular | 0.821549 | 0.024288 | 0.821549 | 0.326599 | raw_tabular | 0 | 0.0 | 0 | 0 | 0 | BASELINE_REFERENCE |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_tabular | 70 | 7.856341 | 33 | 37 | -4 | DEFERRED |

## Diagnostics by model

### SVC

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_no_age | 0.799445 | 0.025304 | 0.805836 | 0.315376 | raw_tabular | 40 | 4.489338 | 13 | 27 | -14 | RAW_AGE_REMOVAL_HURTS |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_tabular | 22 | 2.469136 | 14 | 8 | 6 | KEEP_CANDIDATE |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_no_age | 38 | 4.264871 | 29 | 9 | 20 | KEEP_CANDIDATE |

### GradientBoostingClassifier

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_no_age | 0.806179 | 0.024157 | 0.808081 | 0.333333 | raw_tabular | 71 | 7.968575 | 27 | 44 | -17 | RAW_AGE_REMOVAL_HURTS |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_tabular | 70 | 7.856341 | 33 | 37 | -4 | DEFERRED |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_no_age | 79 | 8.866442 | 46 | 33 | 13 | DEFERRED |

### CatBoostClassifier

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoostClassifier | raw_no_age | 0.802252 | 0.027531 | 0.805836 | 0.317621 | raw_tabular | 70 | 7.856341 | 22 | 48 | -26 | RAW_AGE_REMOVAL_HURTS |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_tabular | 47 | 5.274972 | 20 | 27 | -7 | DEFERRED |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_no_age | 55 | 6.17284 | 37 | 18 | 19 | DEFERRED |

## Decision

| model_name | feature_set | cv_mean | cv_std | oof_accuracy | pred_1_rate | base_feature_set | changed_predictions_vs_base | changed_pct_vs_base | rescue_count | kill_count | net_correct_delta | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_no_age_no_sex_plus_agebucket_v1 | 0.825028 | 0.022134 | 0.828283 | 0.328844 | raw_tabular | 22 | 2.469136 | 14 | 8 | 6 | KEEP_CANDIDATE |
| GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.817389 | 0.022425 | 0.822671 | 0.334456 | raw_tabular | 70 | 7.856341 | 33 | 37 | -4 | DEFERRED |
| CatBoostClassifier | raw_no_age_no_sex_plus_agebucket_v1 | 0.82301 | 0.023725 | 0.82716 | 0.323232 | raw_tabular | 47 | 5.274972 | 20 | 27 | -7 | DEFERRED |

- `KEEP_CANDIDATE` lanes: `1`
- `DEFERRED` lanes: `2`
- `REJECTED` lanes: `0`
- Primary lane status: `DEFERRED`

## Next step recommendation

- AgeBucket v1 improves the raw-no-age branch but does not beat raw_tabular in the primary lane. Do not checkpoint without further review.
- Do not mark AgeBucket v1 as accepted from this report alone.
- Next possible step after review: frozen checkpoint only if AgeBucket v1 becomes a clear candidate.
