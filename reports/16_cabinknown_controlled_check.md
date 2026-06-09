# 16 CabinKnown Controlled Check

## Scope

- controlled train-side CV/OOF check first
- primary model only: `GradientBoostingClassifier`
- full `train.csv` fitting is used only after train-side status is assigned
- `test.csv` is used only for inference and prediction-rate diagnostics
- no public score or Kaggle leaderboard use
- no `gender_submission.csv` as truth
- no test target or row-level test correctness
- no raw Cabin, Deck, Ticket, TicketPrefix, FareLog, FarePerPerson, or Family features

## Current context

- current clean public leader: `raw_tabular / GradientBoostingClassifier`
- current clean public score: `0.79665`
- raw_tabular: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`
- `CabinKnown = df["Cabin"].notna().astype(int)`
- train_survival_rate: `0.383838`
- current_leader_pred_1_rate source: `submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv`

## Fixed variants

| variant | candidate_id | features | submission_candidate | output_file | purpose |
| --- | --- | --- | --- | --- | --- |
| raw_tabular | raw_tabular__GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | False |  | current clean GB baseline reference |
| raw_plus_cabinknown | raw_plus_cabinknown__GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, CabinKnown | True | submissions/submission_16a_gb_raw_plus_cabinknown.csv | controlled check of simple binary CabinKnown on top of raw_tabular |

## Survival rule

- `raw_plus_cabinknown` can become `KEEP_CANDIDATE` only with positive train-side evidence and positive rescue/kill net.
- If evidence is negative or near-zero, status is `REJECTED_TRAIN_SIDE` and no submission is created.
- If evidence is mixed, status is `HOLD_FOR_MANUAL_REVIEW` and no submission is created.
- Meaningful-worse tolerance: `0.0025`.
- Test prediction-rate sanity risk threshold: train_survival_rate + `0.02`.
- Suspicious test-diff threshold without strong train support: `5.0%`.

## Model panel

| variant | model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular | GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| raw_plus_cabinknown | GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Train-side and test-diff diagnostics

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_current_leader | train_survival_rate | current_leader_pred_1_rate | calibration_sanity_flag | status | submission_file | submission_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_tabular | raw_tabular__GradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.827161 | 0.020218 | 0.826816 | 0.859551 | 0.808989 | 0.803371 | 0.837079 | 0.82716 | 0.0 | 0 | 0.0 | 0 | 0 | 0 | 294 | 0.329966 | 0.0 | 0 | 0.0 | 141 | 0.337321 | 0.0 | 0.383838 | 0.337321 | OK | BASELINE_REFERENCE | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | existing leader reference |
| GradientBoostingClassifier | raw_plus_cabinknown | raw_plus_cabinknown__GradientBoostingClassifier | raw_plus_cabinknown | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, CabinKnown | 0.838359 | 0.022455 | 0.860335 | 0.853933 | 0.797753 | 0.831461 | 0.848315 | 0.838384 | 0.011223 | 30 | 3.367003 | 20 | 10 | 10 | 296 | 0.332211 | 0.002245 | 19 | 4.545455 | 138 | 0.330144 | -0.007177 | 0.383838 | 0.337321 | OK | KEEP_CANDIDATE | submissions/submission_16a_gb_raw_plus_cabinknown.csv | generated |

## Candidate decision

- KEEP_CANDIDATE: `1`
- HOLD_FOR_MANUAL_REVIEW: `0`
- REJECTED_TRAIN_SIDE: `0`

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_current_leader | train_survival_rate | current_leader_pred_1_rate | calibration_sanity_flag | status | submission_file | submission_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_plus_cabinknown | raw_plus_cabinknown__GradientBoostingClassifier | raw_plus_cabinknown | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, CabinKnown | 0.838359 | 0.022455 | 0.860335 | 0.853933 | 0.797753 | 0.831461 | 0.848315 | 0.838384 | 0.011223 | 30 | 3.367003 | 20 | 10 | 10 | 296 | 0.332211 | 0.002245 | 19 | 4.545455 | 138 | 0.330144 | -0.007177 | 0.383838 | 0.337321 | OK | KEEP_CANDIDATE | submissions/submission_16a_gb_raw_plus_cabinknown.csv | generated |

## Row-level diff diagnostics

- `reports/16_cabinknown_diff_rows.csv` generated
- includes OOF train rows where candidate prediction changed vs raw_tabular
- includes test full-fit rows where candidate prediction changed vs raw_tabular full-fit prediction
- no decision changed by this diagnostic

## Conditional submission generation

| variant | output_file | rows | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- |
| raw_plus_cabinknown | submissions/submission_16a_gb_raw_plus_cabinknown.csv | 418 | 138 | 0.330144 | generated |

## Sanity checks

- overall status: `PASS`

| check | status | detail |
| --- | --- | --- |
| submission_16a existence follows train-side status | PASS | status=KEEP_CANDIDATE; exists=True; file=submissions/submission_16a_gb_raw_plus_cabinknown.csv |
| submission_16a: 418 rows | PASS | rows=418 |
| submission_16a: columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| submission_16a: PassengerId order matches data/test.csv | PASS | order checked |
| submission_16a: Survived values only 0/1 | PASS | values=[0, 1] |
| submission_16a: no duplicate PassengerId | PASS | duplicates checked |

## Reading boundary

- This report does not use public score.
- No post-public tuning or micro-variants are allowed.
- CabinKnown is checked only as a simple binary feature.
- Deck remains diagnostic only and is not promoted here.
- Family, Title, Age, FareLog, TicketPrefix, and Ticket are not reopened.
