# 05 Baseline Frozen Checkpoint

## Scope Boundary

- full `train.csv` model fitting is allowed for frozen checkpoint file generation
- `test.csv` is used only for inference
- only baseline feature sets from `04_baseline` are used: `f00_core` and `raw_tabular`
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- the same model panel technical parameters and package version logic as `04_baseline` are used
- no `gender_submission.csv` as truth
- no test labels or row-level correctness checks
- no public leaderboard score is used for training, inference, tuning, thresholding, or candidate selection
- fixed public scores are recorded only after frozen file generation as checkpoint metadata
- no feature engineering, `Title`, derived features, deferred features, or target-derived features
- no hyperparameter tuning, threshold tuning, PassengerId overrides, or manual correction rules
- candidates are fixed by the checkpoint batch specification, not by public score

## Candidate Selection

### f00_core layer

These candidates are included because they share the top `f00_core` CV level in `04_baseline`.

| candidate_id | feature_set | model | output_file |
| --- | --- | --- | --- |
| f00_core__SVC | f00_core | SVC | submissions/submission_05_baseline_f00_core_svc.csv |
| f00_core__RandomForestClassifier | f00_core | RandomForestClassifier | submissions/submission_05_baseline_f00_core_random_forest.csv |
| f00_core__GradientBoostingClassifier | f00_core | GradientBoostingClassifier | submissions/submission_05_baseline_f00_core_gradient_boosting.csv |
| f00_core__HistGradientBoostingClassifier | f00_core | HistGradientBoostingClassifier | submissions/submission_05_baseline_f00_core_hist_gradient_boosting.csv |
| f00_core__XGBClassifier | f00_core | XGBClassifier | submissions/submission_05_baseline_f00_core_xgboost.csv |
| f00_core__LGBMClassifier | f00_core | LGBMClassifier | submissions/submission_05_baseline_f00_core_lgbm.csv |
| f00_core__CatBoostClassifier | f00_core | CatBoostClassifier | submissions/submission_05_baseline_f00_core_catboost.csv |

### raw_tabular layer

These candidates are included as strong raw baseline candidates from `04_baseline`.

| candidate_id | feature_set | model | output_file |
| --- | --- | --- | --- |
| raw_tabular__CatBoostClassifier | raw_tabular | CatBoostClassifier | submissions/submission_05_baseline_raw_tabular_catboost.csv |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular | HistGradientBoostingClassifier | submissions/submission_05_baseline_raw_tabular_hist_gradient_boosting.csv |
| raw_tabular__LGBMClassifier | raw_tabular | LGBMClassifier | submissions/submission_05_baseline_raw_tabular_lgbm.csv |
| raw_tabular__GradientBoostingClassifier | raw_tabular | GradientBoostingClassifier | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv |
| raw_tabular__SVC | raw_tabular | SVC | submissions/submission_05_baseline_raw_tabular_svc.csv |
| raw_tabular__RandomForestClassifier | raw_tabular | RandomForestClassifier | submissions/submission_05_baseline_raw_tabular_random_forest.csv |

## Training / Inference Protocol

1. Load `train.csv` and `test.csv`.
2. Select only the fixed feature columns for the candidate feature set.
3. Build an sklearn `Pipeline` with `make_preprocessor(preprocessing_mode, feature_names)` and the model.
4. Fit on full `train.csv`.
5. Predict `Survived` for `test.csv` using model `.predict()` output.
6. Write submission CSV with exactly `PassengerId` and `Survived`.

## Model Panel Used

| model | feature_set | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVC | f00_core | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| RandomForestClassifier | f00_core | scikit-learn | 1.8.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42} | {"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 100, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false} |  |  |
| GradientBoostingClassifier | f00_core | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| HistGradientBoostingClassifier | f00_core | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| XGBClassifier | f00_core | xgboost | 3.2.0 | unscaled_tree | {"eval_metric": "logloss", "n_jobs": 1, "random_state": 42, "verbosity": 0} | {"base_score": null, "booster": null, "callbacks": null, "colsample_bylevel": null, "colsample_bynode": null, "colsample_bytree": null, "device": null, "early_stopping_rounds": null, "enable_categorical": false, "eval_metric": "logloss", "feature_types": null, "feature_weights": null, "gamma": null, "grow_policy": null, "importance_type": null, "interaction_constraints": null, "learning_rate": null, "max_bin": null, "max_cat_threshold": null, "max_cat_to_onehot": null, "max_delta_step": null, "max_depth": null, "max_leaves": null, "min_child_weight": null, "missing": NaN, "monotone_constraints": null, "multi_strategy": null, "n_estimators": null, "n_jobs": 1, "num_parallel_tree": null, "objective": "binary:logistic", "random_state": 42, "reg_alpha": null, "reg_lambda": null, "sampling_method": null, "scale_pos_weight": null, "subsample": null, "tree_method": null, "validate_parameters": null, "verbosity": 0} |  |  |
| LGBMClassifier | f00_core | lightgbm | 4.6.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42, "verbosity": -1} | {"boosting_type": "gbdt", "class_weight": null, "colsample_bytree": 1.0, "importance_type": "split", "learning_rate": 0.1, "max_depth": -1, "min_child_samples": 20, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 100, "n_jobs": 1, "num_leaves": 31, "objective": null, "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 0.0, "subsample": 1.0, "subsample_for_bin": 200000, "subsample_freq": 0, "verbosity": -1} |  |  |
| CatBoostClassifier | f00_core | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |
| CatBoostClassifier | raw_tabular | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |
| HistGradientBoostingClassifier | raw_tabular | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| LGBMClassifier | raw_tabular | lightgbm | 4.6.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42, "verbosity": -1} | {"boosting_type": "gbdt", "class_weight": null, "colsample_bytree": 1.0, "importance_type": "split", "learning_rate": 0.1, "max_depth": -1, "min_child_samples": 20, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 100, "n_jobs": 1, "num_leaves": 31, "objective": null, "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 0.0, "subsample": 1.0, "subsample_for_bin": 200000, "subsample_freq": 0, "verbosity": -1} |  |  |
| GradientBoostingClassifier | raw_tabular | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| SVC | raw_tabular | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| RandomForestClassifier | raw_tabular | scikit-learn | 1.8.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42} | {"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 100, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false} |  |  |

## Generated Submissions

- overall status: `PASS`

| candidate_id | feature_set | model | output_file | rows | pred_0_count | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f00_core__SVC | f00_core | SVC | submissions/submission_05_baseline_f00_core_svc.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__RandomForestClassifier | f00_core | RandomForestClassifier | submissions/submission_05_baseline_f00_core_random_forest.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__GradientBoostingClassifier | f00_core | GradientBoostingClassifier | submissions/submission_05_baseline_f00_core_gradient_boosting.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__HistGradientBoostingClassifier | f00_core | HistGradientBoostingClassifier | submissions/submission_05_baseline_f00_core_hist_gradient_boosting.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__XGBClassifier | f00_core | XGBClassifier | submissions/submission_05_baseline_f00_core_xgboost.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__LGBMClassifier | f00_core | LGBMClassifier | submissions/submission_05_baseline_f00_core_lgbm.csv | 418 | 307 | 111 | 0.26555 | PASS |
| f00_core__CatBoostClassifier | f00_core | CatBoostClassifier | submissions/submission_05_baseline_f00_core_catboost.csv | 418 | 307 | 111 | 0.26555 | PASS |
| raw_tabular__CatBoostClassifier | raw_tabular | CatBoostClassifier | submissions/submission_05_baseline_raw_tabular_catboost.csv | 418 | 278 | 140 | 0.334928 | PASS |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular | HistGradientBoostingClassifier | submissions/submission_05_baseline_raw_tabular_hist_gradient_boosting.csv | 418 | 273 | 145 | 0.34689 | PASS |
| raw_tabular__LGBMClassifier | raw_tabular | LGBMClassifier | submissions/submission_05_baseline_raw_tabular_lgbm.csv | 418 | 272 | 146 | 0.349282 | PASS |
| raw_tabular__GradientBoostingClassifier | raw_tabular | GradientBoostingClassifier | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 418 | 277 | 141 | 0.337321 | PASS |
| raw_tabular__SVC | raw_tabular | SVC | submissions/submission_05_baseline_raw_tabular_svc.csv | 418 | 268 | 150 | 0.358852 | PASS |
| raw_tabular__RandomForestClassifier | raw_tabular | RandomForestClassifier | submissions/submission_05_baseline_raw_tabular_random_forest.csv | 418 | 267 | 151 | 0.361244 | PASS |

## Pairwise prediction difference

| candidate_a | candidate_b | changed_predictions | changed_pct |
| --- | --- | --- | --- |
| f00_core__SVC | f00_core__RandomForestClassifier | 0 | 0.0 |
| f00_core__SVC | f00_core__GradientBoostingClassifier | 0 | 0.0 |
| f00_core__SVC | f00_core__HistGradientBoostingClassifier | 0 | 0.0 |
| f00_core__SVC | f00_core__XGBClassifier | 0 | 0.0 |
| f00_core__SVC | f00_core__LGBMClassifier | 0 | 0.0 |
| f00_core__SVC | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__SVC | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__SVC | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__SVC | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__SVC | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__SVC | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__SVC | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__RandomForestClassifier | f00_core__GradientBoostingClassifier | 0 | 0.0 |
| f00_core__RandomForestClassifier | f00_core__HistGradientBoostingClassifier | 0 | 0.0 |
| f00_core__RandomForestClassifier | f00_core__XGBClassifier | 0 | 0.0 |
| f00_core__RandomForestClassifier | f00_core__LGBMClassifier | 0 | 0.0 |
| f00_core__RandomForestClassifier | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__RandomForestClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__RandomForestClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__RandomForestClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__RandomForestClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__RandomForestClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__RandomForestClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__GradientBoostingClassifier | f00_core__HistGradientBoostingClassifier | 0 | 0.0 |
| f00_core__GradientBoostingClassifier | f00_core__XGBClassifier | 0 | 0.0 |
| f00_core__GradientBoostingClassifier | f00_core__LGBMClassifier | 0 | 0.0 |
| f00_core__GradientBoostingClassifier | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__GradientBoostingClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__GradientBoostingClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__GradientBoostingClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__GradientBoostingClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__GradientBoostingClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__GradientBoostingClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__HistGradientBoostingClassifier | f00_core__XGBClassifier | 0 | 0.0 |
| f00_core__HistGradientBoostingClassifier | f00_core__LGBMClassifier | 0 | 0.0 |
| f00_core__HistGradientBoostingClassifier | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__HistGradientBoostingClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__XGBClassifier | f00_core__LGBMClassifier | 0 | 0.0 |
| f00_core__XGBClassifier | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__XGBClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__XGBClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__XGBClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__XGBClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__XGBClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__XGBClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__LGBMClassifier | f00_core__CatBoostClassifier | 0 | 0.0 |
| f00_core__LGBMClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__LGBMClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__LGBMClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__LGBMClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__LGBMClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__LGBMClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| f00_core__CatBoostClassifier | raw_tabular__CatBoostClassifier | 43 | 10.287081 |
| f00_core__CatBoostClassifier | raw_tabular__HistGradientBoostingClassifier | 54 | 12.91866 |
| f00_core__CatBoostClassifier | raw_tabular__LGBMClassifier | 47 | 11.244019 |
| f00_core__CatBoostClassifier | raw_tabular__GradientBoostingClassifier | 38 | 9.090909 |
| f00_core__CatBoostClassifier | raw_tabular__SVC | 41 | 9.808612 |
| f00_core__CatBoostClassifier | raw_tabular__RandomForestClassifier | 60 | 14.354067 |
| raw_tabular__CatBoostClassifier | raw_tabular__HistGradientBoostingClassifier | 27 | 6.45933 |
| raw_tabular__CatBoostClassifier | raw_tabular__LGBMClassifier | 30 | 7.177033 |
| raw_tabular__CatBoostClassifier | raw_tabular__GradientBoostingClassifier | 23 | 5.502392 |
| raw_tabular__CatBoostClassifier | raw_tabular__SVC | 36 | 8.61244 |
| raw_tabular__CatBoostClassifier | raw_tabular__RandomForestClassifier | 41 | 9.808612 |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular__LGBMClassifier | 19 | 4.545455 |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular__GradientBoostingClassifier | 28 | 6.698565 |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular__SVC | 47 | 11.244019 |
| raw_tabular__HistGradientBoostingClassifier | raw_tabular__RandomForestClassifier | 34 | 8.133971 |
| raw_tabular__LGBMClassifier | raw_tabular__GradientBoostingClassifier | 27 | 6.45933 |
| raw_tabular__LGBMClassifier | raw_tabular__SVC | 50 | 11.961722 |
| raw_tabular__LGBMClassifier | raw_tabular__RandomForestClassifier | 35 | 8.373206 |
| raw_tabular__GradientBoostingClassifier | raw_tabular__SVC | 31 | 7.416268 |
| raw_tabular__GradientBoostingClassifier | raw_tabular__RandomForestClassifier | 42 | 10.047847 |
| raw_tabular__SVC | raw_tabular__RandomForestClassifier | 63 | 15.07177 |

## Sanity Checks

| check | status | detail |
| --- | --- | --- |
| expected submission file count | PASS | 13 of 13 files generated |
| f00_core__SVC: 418 rows | PASS | rows=418 |
| f00_core__SVC: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__SVC: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__SVC: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__SVC: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__RandomForestClassifier: 418 rows | PASS | rows=418 |
| f00_core__RandomForestClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__RandomForestClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__RandomForestClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__RandomForestClassifier: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__GradientBoostingClassifier: 418 rows | PASS | rows=418 |
| f00_core__GradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__GradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__GradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__GradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__HistGradientBoostingClassifier: 418 rows | PASS | rows=418 |
| f00_core__HistGradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__HistGradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__HistGradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__HistGradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__XGBClassifier: 418 rows | PASS | rows=418 |
| f00_core__XGBClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__XGBClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__XGBClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__XGBClassifier: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__LGBMClassifier: 418 rows | PASS | rows=418 |
| f00_core__LGBMClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__LGBMClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__LGBMClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__LGBMClassifier: no duplicate PassengerId | PASS | duplicates checked |
| f00_core__CatBoostClassifier: 418 rows | PASS | rows=418 |
| f00_core__CatBoostClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| f00_core__CatBoostClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| f00_core__CatBoostClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| f00_core__CatBoostClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__CatBoostClassifier: 418 rows | PASS | rows=418 |
| raw_tabular__CatBoostClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__CatBoostClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__CatBoostClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__CatBoostClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__HistGradientBoostingClassifier: 418 rows | PASS | rows=418 |
| raw_tabular__HistGradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__HistGradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__HistGradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__HistGradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__LGBMClassifier: 418 rows | PASS | rows=418 |
| raw_tabular__LGBMClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__LGBMClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__LGBMClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__LGBMClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__GradientBoostingClassifier: 418 rows | PASS | rows=418 |
| raw_tabular__GradientBoostingClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__GradientBoostingClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__GradientBoostingClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__GradientBoostingClassifier: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__SVC: 418 rows | PASS | rows=418 |
| raw_tabular__SVC: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__SVC: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__SVC: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__SVC: no duplicate PassengerId | PASS | duplicates checked |
| raw_tabular__RandomForestClassifier: 418 rows | PASS | rows=418 |
| raw_tabular__RandomForestClassifier: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| raw_tabular__RandomForestClassifier: PassengerId order matches data/test.csv | PASS | order checked |
| raw_tabular__RandomForestClassifier: Survived values only 0/1 | PASS | values=[0, 1] |
| raw_tabular__RandomForestClassifier: no duplicate PassengerId | PASS | duplicates checked |

## Excluded models

| model | reason |
| --- | --- |
| DummyClassifier | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| GaussianNB | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| KNeighborsClassifier | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| LinearSVC | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| DecisionTreeClassifier | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| ExtraTreesClassifier | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |
| AdaBoostClassifier | excluded from checkpoint batch by task scope; not selected in the frozen f00_core top layer or strong raw baseline candidate list |

## Public score checkpoint table

| output_file | public_score | note |
| --- | --- | --- |
| submissions/submission_05_baseline_f00_core_svc.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_random_forest.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_gradient_boosting.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_hist_gradient_boosting.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_xgboost.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_lgbm.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_f00_core_catboost.csv | 0.77751 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_catboost.csv | 0.77990 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_hist_gradient_boosting.csv | 0.75358 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_lgbm.csv | 0.76555 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | 0.79665 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_svc.csv | 0.77990 | Recorded after file generation; external checkpoint evidence only. |
| submissions/submission_05_baseline_raw_tabular_random_forest.csv | 0.75837 | Recorded after file generation; external checkpoint evidence only. |

## Public checkpoint result summary

- best public score: `0.79665`
- best candidate: `raw_tabular__GradientBoostingClassifier`
- all `f00_core` submissions scored `0.77751`
- all `f00_core` submissions are identical according to the generated pairwise diff table
- `raw_tabular` submissions differ materially from each other
- public score is checkpoint evidence only, not tuning feedback

## Public checkpoint status

| candidate_id | public_score | status | note |
| --- | --- | --- | --- |
| f00_core__SVC | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__RandomForestClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__GradientBoostingClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__HistGradientBoostingClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__XGBClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__LGBMClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| f00_core__CatBoostClassifier | 0.77751 | CHECKPOINTED_REFERENCE | identical predictions, public score 0.77751 |
| raw_tabular__CatBoostClassifier | 0.77990 | DEFERRED | same public score 0.77990, not leading but kept as diagnostic evidence |
| raw_tabular__HistGradientBoostingClassifier | 0.75358 | REJECTED | weak public transfer in this frozen baseline checkpoint |
| raw_tabular__LGBMClassifier | 0.76555 | REJECTED | weak public transfer in this frozen baseline checkpoint |
| raw_tabular__GradientBoostingClassifier | 0.79665 | CURRENT_PUBLIC_BASELINE_LEADER | best frozen baseline checkpoint, public score 0.79665 |
| raw_tabular__SVC | 0.77990 | DEFERRED | same public score 0.77990, not leading but kept as diagnostic evidence |
| raw_tabular__RandomForestClassifier | 0.75837 | REJECTED | weak public transfer in this frozen baseline checkpoint |

## Short interpretation

- These are frozen baseline checkpoint files.
- Public score was recorded after file generation.
- No tuning, threshold change, model parameter change, feature change, or PassengerId correction was made after public results.
- `raw_tabular / GradientBoostingClassifier` is the current clean public baseline leader.
- RF default is rejected for the next clean feature-check lane based on this checkpoint.
- This does not use old repo history and does not compare against old repo results.
