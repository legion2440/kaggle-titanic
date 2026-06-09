# 16b CabinKnown Subgroup Gate Check

## Scope

- Frozen subgroup gate for the pre-identified CabinKnown diff subgroup.
- Model is `GradientBoostingClassifier` only.
- No test labels, PassengerId corrections, post-public tuning, weight/blend branch, or additional subgroup variants.
- Deck, TicketPrefix, Ticket, Family, FareLog, Age, and Title are not changed or reopened.

## EDA / OOF rationale

- Full `raw_plus_cabinknown` survived train-side but failed public transfer.
- Step 16 full CabinKnown public score: `0.77990`.
- Direction-only Step 16b downshift/upshift outputs are diagnostic only and are not final subgroup gates.
- The selected subgroup was identified before this frozen checkpoint from OOF/group diagnostics:
  `raw_tabular_pred == 1`, `raw_plus_cabinknown_pred == 0`, `Sex == "male"`, `Pclass == 1`, `CabinKnown == 0`.
- This is a model-diff subgroup, not a raw PassengerId correction.
- Current leader source for prediction-rate comparison: `submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv`.

## Step 16 reference reproduction

- The script stops before submission generation if these reference metrics do not match.

| variant | oof_accuracy | pred_1_count | pred_1_rate | changed_rows_vs_raw_tabular | rescue | kill | net | test_changed_rows_vs_raw_tabular | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular | 0.82716 | 294 | 0.329966 | 0 | 0 | 0 | 0 | 0 | REFERENCE_REPRODUCED |
| raw_plus_cabinknown | 0.839506 | 297 | 0.333333 | 31 | 21 | 10 | 11 | 19 | REFERENCE_REPRODUCED |

## Subgroup rule

```python
gated_pred = raw_tabular_pred
if (
    raw_tabular_pred == 1
    and raw_plus_cabinknown_pred == 0
    and Sex == "male"
    and Pclass == 1
    and CabinKnown == 0
):
    gated_pred = raw_plus_cabinknown_pred
```

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## OOF table

| model_name | variant | rule | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_current_leader | train_survival_rate | current_leader_pred_1_rate | calibration_sanity_flag | train_changed_passenger_ids | test_changed_passenger_ids | submission_file | frozen_public_score | public_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | male_pclass1_cabin_unknown_downshift | raw_tabular_pred == 1 and raw_plus_cabinknown_pred == 0 and Sex == "male" and Pclass == 1 and CabinKnown == 0 | 0.83165 | 0.004489 | 6 | 0.673401 | 5 | 1 | 4 | 288 | 0.323232 | -0.006734 | 3 | 0.717703 | 138 | 0.330144 | -0.007177 | 0.383838 | 0.337321 | OK | 31 35 156 296 448 794 | 915 1040 1215 | submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv | 0.79904 | CURRENT_CLEAN_LEADER |

## Row-level OOF diff table

| split | variant | PassengerId | Survived | raw_tabular_pred | raw_plus_cabinknown_pred | gated_pred | raw_tabular_correct | gated_correct | diff_type | direction | Sex | Pclass | Age | SibSp | Parch | Fare | Cabin | CabinKnown | Embarked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_oof | male_pclass1_cabin_unknown_downshift | 31 | 0 | 1 | 0 | 0 | False | True | rescue | downshift_1_to_0 | male | 1 | 40.0 | 0 | 0 | 27.7208 |  | 0 | C |
| train_oof | male_pclass1_cabin_unknown_downshift | 35 | 0 | 1 | 0 | 0 | False | True | rescue | downshift_1_to_0 | male | 1 | 28.0 | 1 | 0 | 82.1708 |  | 0 | C |
| train_oof | male_pclass1_cabin_unknown_downshift | 156 | 0 | 1 | 0 | 0 | False | True | rescue | downshift_1_to_0 | male | 1 | 51.0 | 0 | 1 | 61.3792 |  | 0 | C |
| train_oof | male_pclass1_cabin_unknown_downshift | 296 | 0 | 1 | 0 | 0 | False | True | rescue | downshift_1_to_0 | male | 1 |  | 0 | 0 | 27.7208 |  | 0 | C |
| train_oof | male_pclass1_cabin_unknown_downshift | 448 | 1 | 1 | 0 | 0 | True | False | kill | downshift_1_to_0 | male | 1 | 34.0 | 0 | 0 | 26.55 |  | 0 | S |
| train_oof | male_pclass1_cabin_unknown_downshift | 794 | 0 | 1 | 0 | 0 | False | True | rescue | downshift_1_to_0 | male | 1 |  | 0 | 0 | 30.6958 |  | 0 | C |

## Test diff table

| split | variant | PassengerId | Survived | raw_tabular_pred | raw_plus_cabinknown_pred | gated_pred | raw_tabular_correct | gated_correct | diff_type | direction | Sex | Pclass | Age | SibSp | Parch | Fare | Cabin | CabinKnown | Embarked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_full_fit | male_pclass1_cabin_unknown_downshift | 915 |  | 1 | 0 | 0 |  |  | test_changed | downshift_1_to_0 | male | 1 | 21.0 | 0 | 1 | 61.3792 |  | 0 | C |
| test_full_fit | male_pclass1_cabin_unknown_downshift | 1040 |  | 1 | 0 | 0 |  |  | test_changed | downshift_1_to_0 | male | 1 |  | 0 | 0 | 26.55 |  | 0 | S |
| test_full_fit | male_pclass1_cabin_unknown_downshift | 1215 |  | 1 | 0 | 0 |  |  | test_changed | downshift_1_to_0 | male | 1 | 33.0 | 0 | 0 | 26.55 |  | 0 | S |

## Submission sanity checks

- overall status: `PASS`

| check | status | detail |
| --- | --- | --- |
| submission exists | PASS | submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv |
| 418 rows | PASS | rows=418 |
| columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| PassengerId order matches data/test.csv | PASS | order checked |
| Survived values only 0/1 | PASS | values=[0, 1] |
| no duplicate PassengerId | PASS | duplicates checked |
| changed PassengerIds exactly expected | PASS | changed=[915, 1040, 1215] |

## Frozen public checkpoint

- submission: `submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv`
- public score: `0.79904`
- status: `CURRENT_CLEAN_LEADER`
- Public score is recorded only after this subgroup was already identified by OOF/group diagnostics.

## Reading boundary

- No further subgroup gates are introduced here.
- No public tuning from this result.
- Weight/blend branch, if pursued, is separate and predeclared.
- Direction-only downshift/upshift remains diagnostic/deprecated, not current candidate logic.
