# 04 Baseline

## Scope Boundary

- train-side CV only
- `train.csv` only
- `Survived` is used only as the target
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- no submission generation
- no Kaggle/public leaderboard use
- no `test.csv` scoring
- no test labels or row-level correctness checks
- no `gender_submission.csv` as truth
- no feature engineering
- no hyperparameter tuning
- no threshold tuning
- no final model selection

## Feature Sets

| feature_set | features |
| --- | --- |
| f00_core | Sex, Pclass, Embarked |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare |

## CV Protocol

- splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- metric: `accuracy`
- identical precomputed CV split indices are reused for every model and feature set
- preprocessing is fitted inside each train fold through an sklearn `Pipeline`
- rows: `891` from `train.csv`

## Model Panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DummyClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"strategy": "most_frequent"} | {"constant": null, "random_state": null, "strategy": "most_frequent"} |  |  |
| LogisticRegression | scikit-learn | 1.8.0 | scaled_linear | {"max_iter": 1000, "random_state": 42} | {"C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true, "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 1000, "n_jobs": null, "penalty": "deprecated", "random_state": 42, "solver": "lbfgs", "tol": 0.0001, "verbose": 0, "warm_start": false} |  |  |
| GaussianNB | scikit-learn | 1.8.0 | unscaled_tree | {} | {"priors": null, "var_smoothing": 1e-09} |  |  |
| KNeighborsClassifier | scikit-learn | 1.8.0 | scaled_linear | {} | {"algorithm": "auto", "leaf_size": 30, "metric": "minkowski", "metric_params": null, "n_jobs": null, "n_neighbors": 5, "p": 2, "weights": "uniform"} |  |  |
| LinearSVC | scikit-learn | 1.8.0 | scaled_linear | {"max_iter": 5000, "random_state": 42} | {"C": 1.0, "class_weight": null, "dual": "auto", "fit_intercept": true, "intercept_scaling": 1, "loss": "squared_hinge", "max_iter": 5000, "multi_class": "ovr", "penalty": "l2", "random_state": 42, "tol": 0.0001, "verbose": 0} |  |  |
| SVC | scikit-learn | 1.8.0 | scaled_linear | {"random_state": 42} | {"C": 1.0, "break_ties": false, "cache_size": 200, "class_weight": null, "coef0": 0.0, "decision_function_shape": "ovr", "degree": 3, "gamma": "scale", "kernel": "rbf", "max_iter": -1, "probability": false, "random_state": 42, "shrinking": true, "tol": 0.001, "verbose": false} |  |  |
| DecisionTreeClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "random_state": 42, "splitter": "best"} |  |  |
| RandomForestClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42} | {"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 100, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false} |  |  |
| ExtraTreesClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42} | {"bootstrap": false, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": null, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 100, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false} |  |  |
| AdaBoostClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"estimator": null, "learning_rate": 1.0, "n_estimators": 50, "random_state": 42} |  |  |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| HistGradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| XGBClassifier | xgboost | 3.2.0 | unscaled_tree | {"eval_metric": "logloss", "n_jobs": 1, "random_state": 42, "verbosity": 0} | {"base_score": null, "booster": null, "callbacks": null, "colsample_bylevel": null, "colsample_bynode": null, "colsample_bytree": null, "device": null, "early_stopping_rounds": null, "enable_categorical": false, "eval_metric": "logloss", "feature_types": null, "feature_weights": null, "gamma": null, "grow_policy": null, "importance_type": null, "interaction_constraints": null, "learning_rate": null, "max_bin": null, "max_cat_threshold": null, "max_cat_to_onehot": null, "max_delta_step": null, "max_depth": null, "max_leaves": null, "min_child_weight": null, "missing": NaN, "monotone_constraints": null, "multi_strategy": null, "n_estimators": null, "n_jobs": 1, "num_parallel_tree": null, "objective": "binary:logistic", "random_state": 42, "reg_alpha": null, "reg_lambda": null, "sampling_method": null, "scale_pos_weight": null, "subsample": null, "tree_method": null, "validate_parameters": null, "verbosity": 0} |  |  |
| LGBMClassifier | lightgbm | 4.6.0 | unscaled_tree | {"n_jobs": 1, "random_state": 42, "verbosity": -1} | {"boosting_type": "gbdt", "class_weight": null, "colsample_bytree": 1.0, "importance_type": "split", "learning_rate": 0.1, "max_depth": -1, "min_child_samples": 20, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 100, "n_jobs": 1, "num_leaves": 31, "objective": null, "random_state": 42, "reg_alpha": 0.0, "reg_lambda": 0.0, "subsample": 1.0, "subsample_for_bin": 200000, "subsample_freq": 0, "verbosity": -1} |  |  |
| CatBoostClassifier | catboost | 1.2.10 | unscaled_tree | {"allow_writing_files": false, "random_seed": 42, "verbose": false} | {"allow_writing_files": false, "random_seed": 42, "verbose": false} |  |  |

## All Results

- overall status: `PASS`

| status | feature_set | model | preprocessing_mode | cv_mean | cv_std | cv_min | cv_max | fold_scores | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | f00_core | DummyClassifier | unscaled_tree | 0.616163 | 0.002325 | 0.61236 | 0.617978 | [0.614525, 0.617978, 0.617978, 0.617978, 0.61236] |  |
| ok | f00_core | LogisticRegression | scaled_linear | 0.777792 | 0.01284 | 0.764045 | 0.797753 | [0.765363, 0.775281, 0.786517, 0.764045, 0.797753] |  |
| ok | f00_core | GaussianNB | unscaled_tree | 0.769933 | 0.009088 | 0.759777 | 0.786517 | [0.759777, 0.769663, 0.769663, 0.764045, 0.786517] |  |
| ok | f00_core | KNeighborsClassifier | scaled_linear | 0.781194 | 0.044226 | 0.724719 | 0.837079 | [0.73743, 0.724719, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | LinearSVC | scaled_linear | 0.786755 | 0.018807 | 0.764045 | 0.820225 | [0.787709, 0.775281, 0.786517, 0.764045, 0.820225] |  |
| ok | f00_core | SVC | scaled_linear | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | DecisionTreeClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | RandomForestClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | ExtraTreesClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | AdaBoostClassifier | unscaled_tree | 0.777792 | 0.01284 | 0.764045 | 0.797753 | [0.765363, 0.775281, 0.786517, 0.764045, 0.797753] |  |
| ok | f00_core | GradientBoostingClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | HistGradientBoostingClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | XGBClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | LGBMClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | f00_core | CatBoostClassifier | unscaled_tree | 0.811443 | 0.017667 | 0.786517 | 0.837079 | [0.815642, 0.797753, 0.786517, 0.837079, 0.820225] |  |
| ok | raw_tabular | DummyClassifier | unscaled_tree | 0.616163 | 0.002325 | 0.61236 | 0.617978 | [0.614525, 0.617978, 0.617978, 0.617978, 0.61236] |  |
| ok | raw_tabular | LogisticRegression | scaled_linear | 0.796874 | 0.014567 | 0.780899 | 0.820225 | [0.782123, 0.803371, 0.797753, 0.780899, 0.820225] |  |
| ok | raw_tabular | GaussianNB | unscaled_tree | 0.783416 | 0.019963 | 0.759777 | 0.814607 | [0.759777, 0.775281, 0.769663, 0.797753, 0.814607] |  |
| ok | raw_tabular | KNeighborsClassifier | scaled_linear | 0.805844 | 0.020204 | 0.780899 | 0.842697 | [0.798883, 0.803371, 0.780899, 0.842697, 0.803371] |  |
| ok | raw_tabular | LinearSVC | scaled_linear | 0.79575 | 0.018819 | 0.775281 | 0.825843 | [0.782123, 0.786517, 0.808989, 0.775281, 0.825843] |  |
| ok | raw_tabular | SVC | scaled_linear | 0.821518 | 0.019504 | 0.792135 | 0.849162 | [0.849162, 0.808989, 0.792135, 0.831461, 0.825843] |  |
| ok | raw_tabular | DecisionTreeClassifier | unscaled_tree | 0.782261 | 0.031095 | 0.735955 | 0.814607 | [0.787709, 0.814607, 0.735955, 0.758427, 0.814607] |  |
| ok | raw_tabular | RandomForestClassifier | unscaled_tree | 0.818153 | 0.022834 | 0.792135 | 0.843575 | [0.843575, 0.792135, 0.792135, 0.820225, 0.842697] |  |
| ok | raw_tabular | ExtraTreesClassifier | unscaled_tree | 0.803584 | 0.022247 | 0.769663 | 0.837079 | [0.810056, 0.808989, 0.769663, 0.792135, 0.837079] |  |
| ok | raw_tabular | AdaBoostClassifier | unscaled_tree | 0.80698 | 0.022431 | 0.775281 | 0.837079 | [0.787709, 0.820225, 0.814607, 0.775281, 0.837079] |  |
| ok | raw_tabular | GradientBoostingClassifier | unscaled_tree | 0.827161 | 0.020218 | 0.803371 | 0.859551 | [0.826816, 0.859551, 0.808989, 0.803371, 0.837079] |  |
| ok | raw_tabular | HistGradientBoostingClassifier | unscaled_tree | 0.830507 | 0.026563 | 0.780899 | 0.853933 | [0.849162, 0.842697, 0.780899, 0.825843, 0.853933] |  |
| ok | raw_tabular | XGBClassifier | unscaled_tree | 0.813671 | 0.025111 | 0.769663 | 0.842697 | [0.832402, 0.808989, 0.769663, 0.814607, 0.842697] |  |
| ok | raw_tabular | LGBMClassifier | unscaled_tree | 0.829377 | 0.021095 | 0.792135 | 0.854749 | [0.854749, 0.831461, 0.792135, 0.825843, 0.842697] |  |
| ok | raw_tabular | CatBoostClassifier | unscaled_tree | 0.835001 | 0.018107 | 0.803371 | 0.849162 | [0.849162, 0.848315, 0.803371, 0.825843, 0.848315] |  |

## Best row by `cv_mean`

| status | feature_set | model | preprocessing_mode | cv_mean | cv_std | cv_min | cv_max | fold_scores | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | raw_tabular | CatBoostClassifier | unscaled_tree | 0.835001 | 0.018107 | 0.803371 | 0.849162 | [0.849162, 0.848315, 0.803371, 0.825843, 0.848315] |  |

## Paired feature-set comparison

This compares `raw_tabular` against `f00_core` within each model class.

| model | f00_core_cv_mean | raw_tabular_cv_mean | delta_raw_minus_f00 | result |
| --- | --- | --- | --- | --- |
| DummyClassifier | 0.616163 | 0.616163 | 0.0 | tied |
| LogisticRegression | 0.777792 | 0.796874 | 0.019082 | improved |
| GaussianNB | 0.769933 | 0.783416 | 0.013483 | improved |
| KNeighborsClassifier | 0.781194 | 0.805844 | 0.02465 | improved |
| LinearSVC | 0.786755 | 0.79575 | 0.008995 | improved |
| SVC | 0.811443 | 0.821518 | 0.010075 | improved |
| DecisionTreeClassifier | 0.811443 | 0.782261 | -0.029182 | worsened |
| RandomForestClassifier | 0.811443 | 0.818153 | 0.00671 | improved |
| ExtraTreesClassifier | 0.811443 | 0.803584 | -0.007859 | worsened |
| AdaBoostClassifier | 0.777792 | 0.80698 | 0.029188 | improved |
| GradientBoostingClassifier | 0.811443 | 0.827161 | 0.015718 | improved |
| HistGradientBoostingClassifier | 0.811443 | 0.830507 | 0.019064 | improved |
| XGBClassifier | 0.811443 | 0.813671 | 0.002228 | improved |
| LGBMClassifier | 0.811443 | 0.829377 | 0.017934 | improved |
| CatBoostClassifier | 0.811443 | 0.835001 | 0.023558 | improved |

- improved: `12`
- worsened: `2`
- tied: `1`
- best positive delta: `AdaBoostClassifier (0.029188)`
- worst negative delta: `DecisionTreeClassifier (-0.029182)`

## Comparison summary

- best model on `f00_core`: SVC (0.811443)
- best model on `raw_tabular`: CatBoostClassifier (0.835001)
- whether `raw_tabular` improves over `f00_core`: yes (delta=0.023558)

## Short interpretation

- This is baseline evidence only.
- This is not final model selection.
- No feature engineering or tuning was used.
