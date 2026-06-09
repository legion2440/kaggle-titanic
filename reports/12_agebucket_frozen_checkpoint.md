# 12 AgeBucket Frozen Checkpoint

## Scope

- frozen checkpoint for AgeBucket v1 transfer probe
- candidates are fixed before any public score
- full `train.csv` fitting is allowed
- `test.csv` is used only for inference
- no submission logic uses public score

## Method boundary

- This is a frozen checkpoint, not feature acceptance.
- No post-score tuning.
- No new features.
- AgeBucket mapping is unchanged from step 11.
- Broad `Title` remains closed.
- No Master fallback, Old buckets, Mrs/Miss, Surname, target-derived family/group survival, or PassengerId corrections.
- `gender_submission.csv` is not used as truth.
- Test target is not used.

## Fixed candidate list

| candidate_id | model | feature_set | features | preprocessing_mode | output_file | reason |
| --- | --- | --- | --- | --- | --- | --- |
| agebucket_v1__SVC | SVC | raw_no_age_no_sex_plus_agebucket_v1 | Pclass, Embarked, SibSp, Parch, Fare, AgeBucket | scaled_linear | submissions/submission_12a_svc_agebucket_v1.csv | train-side KEEP_CANDIDATE in step 11 |
| agebucket_v1__GradientBoostingClassifier | GradientBoostingClassifier | raw_no_age_no_sex_plus_agebucket_v1 | Pclass, Embarked, SibSp, Parch, Fare, AgeBucket | unscaled_tree | submissions/submission_12b_gb_agebucket_v1.csv | primary-lane transfer probe by explicit review override |

## Why checkpoint is allowed

- SVC is train-side `KEEP_CANDIDATE` in `reports/11_agebucket_feature_check.md`.
- GradientBoostingClassifier is a primary-lane transfer probe by explicit review override.
- The candidate list is fixed before public score and must not be changed post-score.

## Why CatBoost is excluded

- CatBoost was `DEFERRED` in step 11.
- This checkpoint is intentionally limited to SVC and the primary GB transfer probe.
- CatBoost is not included to avoid broadening the transfer probe after review.

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

## Model panel used

| candidate_id | model | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agebucket_v1__SVC | SVC | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| agebucket_v1__GradientBoostingClassifier | GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Submission diagnostics

- overall status: `PASS`

| candidate_id | model | output_file | rows | pred_0_count | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agebucket_v1__SVC | SVC | submissions/submission_12a_svc_agebucket_v1.csv | 418 | 269 | 149 | 0.356459 | PASS |
| agebucket_v1__GradientBoostingClassifier | GradientBoostingClassifier | submissions/submission_12b_gb_agebucket_v1.csv | 418 | 273 | 145 | 0.34689 | PASS |

## Diff vs raw GB baseline submission

| baseline_candidate_id | candidate_id | baseline_file | changed_predictions | changed_pct | baseline_pred_1_count | candidate_pred_1_count | baseline_pred_1_rate | candidate_pred_1_rate | delta_pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular__GradientBoostingClassifier | agebucket_v1__SVC | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 32 | 7.655502 | 141 | 149 | 0.337321 | 0.356459 | 0.019139 | PASS |
| raw_tabular__GradientBoostingClassifier | agebucket_v1__GradientBoostingClassifier | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 22 | 5.263158 | 141 | 145 | 0.337321 | 0.34689 | 0.009569 | PASS |

## Sanity checks

| check | status | detail |
| --- | --- | --- |
| exactly 2 submission files generated | PASS | 2 of 2 files |
| agebucket_v1__SVC: 418 rows | PASS | rows=418 |
| agebucket_v1__SVC: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| agebucket_v1__SVC: PassengerId order matches data/test.csv | PASS | order checked |
| agebucket_v1__SVC: Survived values only 0/1 | PASS | values=[0, 1] |
| agebucket_v1__SVC: no duplicate PassengerId | PASS | duplicates checked |
| agebucket_v1__GradientBoostingClassifier: 418 rows | PASS | rows=418 |
| agebucket_v1__GradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| agebucket_v1__GradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| agebucket_v1__GradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| agebucket_v1__GradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |

## Public score placeholder

Public score:

- submission_12a_svc_agebucket_v1.csv: TBD
- submission_12b_gb_agebucket_v1.csv: TBD

| output_file | public_score |
| --- | --- |
| submission_12a_svc_agebucket_v1.csv | TBD |
| submission_12b_gb_agebucket_v1.csv | TBD |

## Decision rule after public score

- If neither beats current public baseline `raw_tabular / GradientBoostingClassifier = 0.79665`, close AgeBucket v1 as public-transfer failed.
- If one beats baseline, mark as checkpoint leader/candidate, but do not do row-level tuning.
- Do not use public result to create micro-variants.
