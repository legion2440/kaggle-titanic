# 07 Title Frozen Checkpoint

## Scope Boundary

- full `train.csv` model fitting is allowed for frozen Title checkpoint file generation
- `test.csv` is used only for inference
- only `raw_plus_title` is used
- `raw_plus_title` is defined as `RAW_TABULAR + ["Title"]`
- `add_clean_features()` is used only to create `Title`; only `Title` is copied into the working frames
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- active model lanes are taken from `06_title_feature_check`
- technical model parameters and package version logic match `04_baseline` / `06_title_feature_check`
- no `gender_submission.csv` as truth
- no test labels or row-level correctness checks
- public scores are recorded only as post-generation checkpoint metadata
- public scores are not used for training, inference, candidate selection, or row-level logic
- no derived features other than `Title` are included
- no hyperparameter tuning, threshold tuning, gating, probability threshold changes, PassengerId overrides, or manual correction rules
- no target-derived features
- candidates are fixed before public checkpointing and are not selected by public score

## Candidate Selection

Only active `raw_plus_title` lanes are promoted to frozen submission files.

| candidate_id | feature_set | model | output_file |
| --- | --- | --- | --- |
| raw_plus_title__SVC | raw_plus_title | SVC | submissions/submission_07_title_raw_plus_title_svc.csv |
| raw_plus_title__GradientBoostingClassifier | raw_plus_title | GradientBoostingClassifier | submissions/submission_07_title_raw_plus_title_gradient_boosting.csv |
| raw_plus_title__CatBoostClassifier | raw_plus_title | CatBoostClassifier | submissions/submission_07_title_raw_plus_title_catboost.csv |

## Training / Inference Protocol

1. Load `train.csv` and `test.csv`.
2. Create clean features with `add_clean_features()` for train and test.
3. Copy only `Title` into the train/test working frames.
4. Select only `RAW_TABULAR + ["Title"]`.
5. Build an sklearn `Pipeline` with `make_preprocessor(preprocessing_mode, feature_names)` and the model.
6. Fit on full `train.csv`.
7. Predict `Survived` for `test.csv` using model `.predict()` output.
8. Write submission CSV with exactly `PassengerId` and `Survived`.

## Model Panel Used

| model | feature_set | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | raw_plus_title | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| GradientBoostingClassifier | raw_plus_title | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| CatBoostClassifier | raw_plus_title | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |

## Generated Submissions

- overall status: `PASS`

| candidate_id | feature_set | model | output_file | rows | pred_0_count | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_plus_title__SVC | raw_plus_title | SVC | submissions/submission_07_title_raw_plus_title_svc.csv | 418 | 260 | 158 | 0.37799 | PASS |
| raw_plus_title__GradientBoostingClassifier | raw_plus_title | GradientBoostingClassifier | submissions/submission_07_title_raw_plus_title_gradient_boosting.csv | 418 | 269 | 149 | 0.356459 | PASS |
| raw_plus_title__CatBoostClassifier | raw_plus_title | CatBoostClassifier | submissions/submission_07_title_raw_plus_title_catboost.csv | 418 | 268 | 150 | 0.358852 | PASS |

## Title vs raw_tabular baseline prediction difference

| base_candidate_id | title_candidate_id | changed_predictions | changed_pct | base_pred_1_count | title_pred_1_count | base_pred_1_rate | title_pred_1_rate | delta_pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular__SVC | raw_plus_title__SVC | 10 | 2.392344 | 150 | 158 | 0.358852 | 0.37799 | 0.019139 | PASS |
| raw_tabular__GradientBoostingClassifier | raw_plus_title__GradientBoostingClassifier | 22 | 5.263158 | 141 | 149 | 0.337321 | 0.356459 | 0.019139 | PASS |
| raw_tabular__CatBoostClassifier | raw_plus_title__CatBoostClassifier | 22 | 5.263158 | 140 | 150 | 0.334928 | 0.358852 | 0.023923 | PASS |

## Pairwise prediction difference among Title submissions

| candidate_a | candidate_b | changed_predictions | changed_pct |
| --- | --- | --- | --- |
| raw_plus_title__SVC | raw_plus_title__GradientBoostingClassifier | 33 | 7.894737 |
| raw_plus_title__SVC | raw_plus_title__CatBoostClassifier | 36 | 8.61244 |
| raw_plus_title__GradientBoostingClassifier | raw_plus_title__CatBoostClassifier | 17 | 4.066986 |

## Sanity Checks

| check | status | detail |
| --- | --- | --- |
| exactly 3 new files generated | PASS | 3 of 3 files generated |
| raw_plus_title__SVC: every file has 418 rows | PASS | rows=418 |
| raw_plus_title__SVC: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_plus_title__SVC: PassengerId order matches data/test.csv | PASS | order checked |
| raw_plus_title__SVC: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_plus_title__SVC: no duplicate PassengerId | PASS | duplicates checked |
| raw_plus_title__GradientBoostingClassifier: every file has 418 rows | PASS | rows=418 |
| raw_plus_title__GradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_plus_title__GradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_plus_title__GradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_plus_title__GradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_plus_title__CatBoostClassifier: every file has 418 rows | PASS | rows=418 |
| raw_plus_title__CatBoostClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_plus_title__CatBoostClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_plus_title__CatBoostClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_plus_title__CatBoostClassifier: no duplicate PassengerId | PASS | duplicates checked |

## Diagnostic-only `f00_core_plus_title`

- `f00_core_plus_title` was checked in `06_title_feature_check`.
- It is not promoted to public checkpoint.
- Reason: raw/title lanes are stronger and baseline `f00_core` is already a weak reference.

## Excluded models

| model | reason |
| --- | --- |
| RandomForestClassifier | rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane |
| LGBMClassifier | rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane |
| HistGradientBoostingClassifier | rejected by the frozen raw_tabular baseline checkpoint; not an active Title checkpoint lane |
| XGBClassifier | not an active Title checkpoint lane from `06_title_feature_check` |
| ExtraTreesClassifier | not an active Title checkpoint lane from `06_title_feature_check` |
| DecisionTreeClassifier | not an active Title checkpoint lane from `06_title_feature_check` |
| AdaBoostClassifier | not an active Title checkpoint lane from `06_title_feature_check` |
| LinearSVC | not an active Title checkpoint lane from `06_title_feature_check` |
| KNeighborsClassifier | not an active Title checkpoint lane from `06_title_feature_check` |
| GaussianNB | not an active Title checkpoint lane from `06_title_feature_check` |
| DummyClassifier | not an active Title checkpoint lane from `06_title_feature_check` |

## Public score checkpoint table

| output_file | public_score | note |
| --- | --- | --- |
| submissions/submission_07_title_raw_plus_title_svc.csv | 0.77990 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_07_title_raw_plus_title_gradient_boosting.csv | 0.76794 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_07_title_raw_plus_title_catboost.csv | 0.76076 | Recorded after file generation; external checkpoint evidence only. |

## Public checkpoint result summary

- best public score in this checkpoint: `0.77990`
- best candidate in this checkpoint: `raw_plus_title__SVC`
- `raw_plus_title__SVC` matched its raw baseline public score and produced no public gain
- `raw_plus_title__GradientBoostingClassifier` dropped from raw baseline `0.79665` to `0.76794`
- `raw_plus_title__CatBoostClassifier` dropped from raw baseline `0.77990` to `0.76076`
- unrestricted `Title` did not transfer as a full-strength direct feature
- public score is checkpoint evidence only, not tuning feedback

## Public checkpoint status

| candidate_id | public_score | baseline_public_score | public_delta | status | note |
| --- | --- | --- | --- | --- | --- |
| raw_plus_title__SVC | 0.77990 | 0.77990 | 0.00000 | CHECKPOINTED_NO_GAIN | matched raw_tabular SVC baseline; no public gain |
| raw_plus_title__GradientBoostingClassifier | 0.76794 | 0.79665 | -0.02871 | REJECTED_PUBLIC_TRANSFER | train-side Title gain did not transfer; unrestricted Title damaged public score |
| raw_plus_title__CatBoostClassifier | 0.76076 | 0.77990 | -0.01914 | REJECTED_PUBLIC_TRANSFER | train-side Title gain did not transfer; unrestricted Title damaged public score |

## Transfer-risk observation

- `raw_plus_title` changed 10 predictions for SVC and produced no public gain.
- `raw_plus_title` changed 22 predictions for GradientBoostingClassifier and CatBoostClassifier and both public scores dropped materially.
- On a 418-row test set, larger flip counts can be a transfer-risk signal for overlapping derived features.
- This is an empirical checkpoint observation, not a hard rule.
- `Title` should not be used as an unrestricted full-strength feature in the next clean lane.
- `Title` signal remains eligible for a separate gated/conservative check.

## Short interpretation

- These are frozen Title checkpoint files.
- Public score was recorded after file generation.
- No tuning, threshold change, gating, model parameter change, feature change, or PassengerId correction was made after public results.
- Unrestricted `raw_plus_title` is rejected as a direct full-strength feature.
- `raw_plus_title / SVC` is checkpointed with no gain.
- `raw_plus_title / GradientBoostingClassifier` and `raw_plus_title / CatBoostClassifier` are rejected for public transfer.
- `Title` signal is moved to `RETEST_FOR_GATING`.
- This does not use old repo history and does not compare against old repo results.
