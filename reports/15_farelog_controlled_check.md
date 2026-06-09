# 15 FareLog Controlled Check

## Scope

- controlled train-side CV/OOF check first
- full `train.csv` fitting is used only after train-side status is assigned
- `test.csv` is used only for inference and prediction-rate diagnostics
- no public score or Kaggle leaderboard use
- no `gender_submission.csv` as truth
- no test target or row-level test correctness
- no gated FareLog variant in this step
- no Family / FamilySize / FamilySizeBucket next-step work

## Current context

- current clean public leader: `raw_tabular / GradientBoostingClassifier`
- current clean public score: `0.79665`
- raw_tabular already contains raw `Fare`: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`
- train_survival_rate: `0.383838`
- current_leader_pred_1_rate source: `submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv`

## Fixed variants

| variant | candidate_id | features | submission_candidate | output_file | purpose |
| --- | --- | --- | --- | --- | --- |
| raw_tabular | raw_tabular__GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | False |  | current clean GB baseline reference; raw Fare is already included |
| raw_plus_farelog | raw_plus_farelog__GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, FareLog | True | submissions/submission_15a_gb_raw_plus_farelog.csv | controlled check of FareLog as a transformation added to raw Fare |
| raw_farelog_replace_fare | raw_farelog_replace_fare__GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, FareLog | True | submissions/submission_15c_gb_farelog_replace_fare.csv | controlled check of replacing raw Fare with FareLog |

## Excluded from this step

- `submission_15b_gb_raw_plus_farelog_gated.csv` is not created.
- Gated FareLog requires manual inspection and tuning after OOF/test diff review.
- No gating is improvised inside this first controlled check.

## Survival rule

- A variant can become `KEEP_CANDIDATE` only with positive train-side evidence: CV/OOF better than raw_tabular, positive rescue/kill net, and no obvious OOF predicted-survival inflation.
- If evidence is mixed, the status is `HOLD_FOR_MANUAL_REVIEW` or `REJECTED_TRAIN_SIDE`; no forced submission is created.
- Meaningful-worse tolerance: `0.0025`.
- Test prediction-rate sanity risk threshold: train_survival_rate + `0.02`.
- A test prediction-rate sanity risk is marked clearly and is not used as the only rejection rule.

## Model panel

| variant | model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular | GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| raw_plus_farelog | GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| raw_farelog_replace_fare | GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Train-side and test-diff diagnostics

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_current_leader | train_survival_rate | current_leader_pred_1_rate | calibration_sanity_flag | status | submission_file | submission_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_tabular | raw_tabular__GradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.827161 | 0.020218 | 0.826816 | 0.859551 | 0.808989 | 0.803371 | 0.837079 | 0.82716 | 0.0 | 0 | 0.0 | 0 | 0 | 0 | 294 | 0.329966 | 0.0 | 0 | 0.0 | 141 | 0.337321 | 0.0 | 0.383838 | 0.337321 | OK | BASELINE_REFERENCE | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv | existing leader reference |
| GradientBoostingClassifier | raw_plus_farelog | raw_plus_farelog__GradientBoostingClassifier | raw_plus_farelog | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, FareLog | 0.826037 | 0.021616 | 0.826816 | 0.859551 | 0.808989 | 0.797753 | 0.837079 | 0.826038 | -0.001122 | 1 | 0.112233 | 0 | 1 | -1 | 293 | 0.328844 | -0.001122 | 1 | 0.239234 | 140 | 0.334928 | -0.002392 | 0.383838 | 0.337321 | OK | REJECTED_TRAIN_SIDE | submissions/submission_15a_gb_raw_plus_farelog.csv | not generated: REJECTED_TRAIN_SIDE |
| GradientBoostingClassifier | raw_farelog_replace_fare | raw_farelog_replace_fare__GradientBoostingClassifier | raw_farelog_replace_fare | Sex, Pclass, Embarked, Age, SibSp, Parch, FareLog | 0.826037 | 0.021616 | 0.826816 | 0.859551 | 0.808989 | 0.797753 | 0.837079 | 0.826038 | -0.001122 | 1 | 0.112233 | 0 | 1 | -1 | 293 | 0.328844 | -0.001122 | 1 | 0.239234 | 140 | 0.334928 | -0.002392 | 0.383838 | 0.337321 | OK | REJECTED_TRAIN_SIDE | submissions/submission_15c_gb_farelog_replace_fare.csv | not generated: REJECTED_TRAIN_SIDE |

## Candidate decision

- KEEP_CANDIDATE: `0`
- HOLD_FOR_MANUAL_REVIEW: `0`
- REJECTED_TRAIN_SIDE: `2`

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_current_leader | train_survival_rate | current_leader_pred_1_rate | calibration_sanity_flag | status | submission_file | submission_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_plus_farelog | raw_plus_farelog__GradientBoostingClassifier | raw_plus_farelog | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, FareLog | 0.826037 | 0.021616 | 0.826816 | 0.859551 | 0.808989 | 0.797753 | 0.837079 | 0.826038 | -0.001122 | 1 | 0.112233 | 0 | 1 | -1 | 293 | 0.328844 | -0.001122 | 1 | 0.239234 | 140 | 0.334928 | -0.002392 | 0.383838 | 0.337321 | OK | REJECTED_TRAIN_SIDE | submissions/submission_15a_gb_raw_plus_farelog.csv | not generated: REJECTED_TRAIN_SIDE |
| GradientBoostingClassifier | raw_farelog_replace_fare | raw_farelog_replace_fare__GradientBoostingClassifier | raw_farelog_replace_fare | Sex, Pclass, Embarked, Age, SibSp, Parch, FareLog | 0.826037 | 0.021616 | 0.826816 | 0.859551 | 0.808989 | 0.797753 | 0.837079 | 0.826038 | -0.001122 | 1 | 0.112233 | 0 | 1 | -1 | 293 | 0.328844 | -0.001122 | 1 | 0.239234 | 140 | 0.334928 | -0.002392 | 0.383838 | 0.337321 | OK | REJECTED_TRAIN_SIDE | submissions/submission_15c_gb_farelog_replace_fare.csv | not generated: REJECTED_TRAIN_SIDE |

## Conditional submission generation

| variant | output_file | rows | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- |
| raw_plus_farelog | submissions/submission_15a_gb_raw_plus_farelog.csv |  |  |  | not generated: REJECTED_TRAIN_SIDE |
| raw_farelog_replace_fare | submissions/submission_15c_gb_farelog_replace_fare.csv |  |  |  | not generated: REJECTED_TRAIN_SIDE |

## Row-level diff diagnostics

- `reports/15_farelog_diff_rows.csv` generated
- compare whether raw_plus_farelog and raw_farelog_replace_fare changed the same PassengerId(s)
- no decision changed by this diagnostic

## Sanity checks

- overall status: `PASS`

| check | status | detail |
| --- | --- | --- |
| forbidden gated submission_15b absent | PASS | submissions/submission_15b_gb_raw_plus_farelog_gated.csv |
| raw_plus_farelog: submission existence follows train-side status | PASS | status=REJECTED_TRAIN_SIDE; exists=False; file=submissions/submission_15a_gb_raw_plus_farelog.csv |
| raw_farelog_replace_fare: submission existence follows train-side status | PASS | status=REJECTED_TRAIN_SIDE; exists=False; file=submissions/submission_15c_gb_farelog_replace_fare.csv |

## Reading boundary

- This report does not accept or reject FareLog as a general feature.
- It only records whether the two fixed FareLog candidates survive this controlled GB check.
- No public-score tuning or micro-variants are allowed after failed transfer.
- The gated FareLog branch remains separate and requires manual review before any public-facing file.
