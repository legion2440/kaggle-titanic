# 17B FamilySurnameSizeBand Replacement Controlled Check

## Purpose

This is corrected Step 17B: a controlled `GradientBoostingClassifier` check of plain `FamilySurnameSizeBand` as a replacement for raw `SibSp` and `Parch`.

This is not additive on top of `SibSp/Parch`, not overlap-aware, and not target encoding. Mismatch-aware features are explicitly out of scope. No submission was created.

## Feature boundary

- `FamilySize = SibSp + Parch + 1`.
- `Surname` is the substring before the comma in `Name`.
- `SurnameCount` is counted over the combined train/test passenger manifest without `Survived`.
- `FamilySurnameSize = max(FamilySize, SurnameCount)`.
- `FamilySurnameSizeBand`: `alone=1`, `small=2-4`, `medium=5-6`, `large=7+`.
- Sex and Pclass remain separate raw features and are not baked into the size band.

## Baseline vs candidate feature sets

| variant | features | purpose |
| --- | --- | --- |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | baseline GB reference with raw SibSp and Parch |
| raw_family_surname_size_band | Sex, Pclass, Embarked, Age, Fare, FamilySurnameSizeBand | controlled replacement of SibSp/Parch with one plain FamilySurnameSizeBand categorical feature |

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## OOF result

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | pred_1_count | pred_1_rate | pred_1_rate_delta_vs_raw_tabular | test_changed_rows_vs_raw_tabular_full_fit | test_changed_pct_vs_raw_tabular_full_fit | test_0_to_1 | test_1_to_0 | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_raw_tabular | train_survival_rate | prediction_rate_sanity_flag | final_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_tabular | raw_tabular__GradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.825701 | 0.021513 | 0.82716 | 0.0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 294 | 0.329966 | 0.0 | 0 | 0.0 | 0 | 0 | 141 | 0.337321 | 0.0 | 0.383838 | OK |  |
| GradientBoostingClassifier | raw_family_surname_size_band | raw_family_surname_size_band__GradientBoostingClassifier | raw_family_surname_size_band | Sex, Pclass, Embarked, Age, Fare, FamilySurnameSizeBand | 0.823119 | 0.025799 | 0.820426 | -0.006734 | 28 | 3.142536 | 16 | 12 | 11 | 17 | -6 | 298 | 0.334456 | 0.004489 | 20 | 4.784689 | 10 | 10 | 141 | 0.337321 | 0.0 | 0.383838 | OK | REJECTED_TRAIN_SIDE |

## OOF changed-row audit

- changed rows: `28`
- 0 -> 1: `16`
- 1 -> 0: `12`
- rescue / kill / net: `11` / `17` / `-6`
- changed PassengerIds: `35 39 65 75 98 166 234 242 296 348 375 387 446 475 535 545 555 568 584 613 633 672 804 824 851 868 870 883`

OOF changed rows by plain size band:

| FamilySurnameSizeBand | diff_direction | rescue_or_kill | count | PassengerIds |
| --- | --- | --- | --- | --- |
| alone | upshift_0_to_1 | kill | 5 | 65 475 584 868 883 |
| small | upshift_0_to_1 | rescue | 4 | 98 242 348 613 |
| small | downshift_1_to_0 | kill | 3 | 166 446 824 |
| small | downshift_1_to_0 | rescue | 3 | 35 39 545 |
| alone | downshift_1_to_0 | kill | 2 | 75 633 |
| large | upshift_0_to_1 | kill | 2 | 387 851 |
| medium | downshift_1_to_0 | kill | 2 | 804 870 |
| small | upshift_0_to_1 | kill | 2 | 535 672 |
| alone | downshift_1_to_0 | rescue | 1 | 296 |
| alone | upshift_0_to_1 | rescue | 1 | 555 |
| large | upshift_0_to_1 | rescue | 1 | 234 |
| medium | downshift_1_to_0 | rescue | 1 | 568 |
| medium | upshift_0_to_1 | kill | 1 | 375 |

OOF row-level audit:

| split | variant | PassengerId | Survived | raw_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | FamilySize | Surname | SurnameCount | FamilySurnameSize | FamilySurnameSizeBand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_oof | raw_family_surname_size_band | 35 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 28.0 | 1 | 0 | 82.1708 | 2 | Meyer | 3 | 3 | small |
| train_oof | raw_family_surname_size_band | 39 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 18.0 | 2 | 0 | 18.0 | 3 | Vander Planke | 4 | 4 | small |
| train_oof | raw_family_surname_size_band | 65 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 |  | 0 | 0 | 27.7208 | 1 | Stewart | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 75 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 32.0 | 0 | 0 | 56.4958 | 1 | Bing | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 98 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 1 | 23.0 | 0 | 1 | 63.3583 | 2 | Greenfield | 2 | 2 | small |
| train_oof | raw_family_surname_size_band | 166 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 9.0 | 0 | 2 | 20.525 | 3 | Goldsmith | 4 | 4 | small |
| train_oof | raw_family_surname_size_band | 234 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 5.0 | 4 | 2 | 31.3875 | 7 | Asplund | 8 | 8 | large |
| train_oof | raw_family_surname_size_band | 242 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 15.5 | 2 | Murphy | 3 | 3 | small |
| train_oof | raw_family_surname_size_band | 296 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 |  | 0 | 0 | 27.7208 | 1 | Lewy | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 348 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 16.1 | 2 | Davison | 2 | 2 | small |
| train_oof | raw_family_surname_size_band | 375 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 3.0 | 3 | 1 | 21.075 | 5 | Palsson | 5 | 5 | medium |
| train_oof | raw_family_surname_size_band | 387 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 1.0 | 5 | 2 | 46.9 | 8 | Goodwin | 8 | 8 | large |
| train_oof | raw_family_surname_size_band | 446 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 4.0 | 0 | 2 | 81.8583 | 3 | Dodge | 3 | 3 | small |
| train_oof | raw_family_surname_size_band | 475 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 22.0 | 0 | 0 | 9.8375 | 1 | Strandberg | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 535 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 30.0 | 0 | 0 | 8.6625 | 1 | Cacic | 4 | 4 | small |
| train_oof | raw_family_surname_size_band | 545 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 50.0 | 1 | 0 | 106.425 | 2 | Douglas | 3 | 3 | small |
| train_oof | raw_family_surname_size_band | 555 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 22.0 | 0 | 0 | 7.775 | 1 | Ohman | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | 5 | Palsson | 5 | 5 | medium |
| train_oof | raw_family_surname_size_band | 584 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 36.0 | 0 | 0 | 40.125 | 1 | Ross | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 613 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 15.5 | 2 | Murphy | 3 | 3 | small |
| train_oof | raw_family_surname_size_band | 633 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 32.0 | 0 | 0 | 30.5 | 1 | Stahelin-Maeglin | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 672 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 31.0 | 1 | 0 | 52.0 | 2 | Davidson | 2 | 2 | small |
| train_oof | raw_family_surname_size_band | 804 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 0.42 | 0 | 1 | 8.5167 | 2 | Thomas | 5 | 5 | medium |
| train_oof | raw_family_surname_size_band | 824 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 3 | 27.0 | 0 | 1 | 12.475 | 2 | Moor | 2 | 2 | small |
| train_oof | raw_family_surname_size_band | 851 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 4.0 | 4 | 2 | 31.275 | 7 | Andersson | 11 | 11 | large |
| train_oof | raw_family_surname_size_band | 868 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 31.0 | 0 | 0 | 50.4958 | 1 | Roebling | 1 | 1 | alone |
| train_oof | raw_family_surname_size_band | 870 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 4.0 | 1 | 1 | 11.1333 | 3 | Johnson | 6 | 6 | medium |
| train_oof | raw_family_surname_size_band | 883 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 22.0 | 0 | 0 | 10.5167 | 1 | Dahlberg | 1 | 1 | alone |

## Test full-fit diff audit

- changed rows: `20`
- 0 -> 1: `10`
- 1 -> 0: `10`
- changed PassengerIds: `911 915 920 931 933 956 982 986 1010 1036 1050 1084 1098 1175 1205 1239 1251 1275 1284 1297`
- test pred_1_count: `141`
- test pred_1_rate: `0.337321`
- pred_1_rate delta vs raw_tabular: `0.0`

Test changed rows by plain size band:

| FamilySurnameSizeBand | diff_direction | rescue_or_kill | count | PassengerIds |
| --- | --- | --- | --- | --- |
| alone | upshift_0_to_1 | test_changed | 5 | 920 986 1010 1036 1050 |
| small | upshift_0_to_1 | test_changed | 5 | 1084 1175 1251 1275 1284 |
| alone | downshift_1_to_0 | test_changed | 4 | 911 931 1239 1297 |
| small | downshift_1_to_0 | test_changed | 4 | 933 982 1098 1205 |
| medium | downshift_1_to_0 | test_changed | 2 | 915 956 |

Test row-level audit:

| split | variant | PassengerId | Survived | raw_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | FamilySize | Surname | SurnameCount | FamilySurnameSize | FamilySurnameSizeBand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_full_fit | raw_family_surname_size_band | 911 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 45.0 | 0 | 0 | 7.225 | 1 | Assaf Khalil | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 915 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 | 21.0 | 0 | 1 | 61.3792 | 2 | Williams | 5 | 5 | medium |
| test_full_fit | raw_family_surname_size_band | 920 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 41.0 | 0 | 0 | 30.5 | 1 | Brady | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 931 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 3 |  | 0 | 0 | 56.4958 | 1 | Hee | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 933 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 |  | 0 | 0 | 26.55 | 1 | Franklin | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 956 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 | 13.0 | 2 | 2 | 262.375 | 5 | Ryerson | 5 | 5 | medium |
| test_full_fit | raw_family_surname_size_band | 982 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 22.0 | 1 | 0 | 13.9 | 2 | Dyker | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 986 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 25.0 | 0 | 0 | 26.0 | 1 | Birnbaum | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 1010 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 36.0 | 0 | 0 | 75.2417 | 1 | Beattie | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 1036 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 42.0 | 0 | 0 | 26.55 | 1 | Lindeberg-Lind | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 1050 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 42.0 | 0 | 0 | 26.55 | 1 | Borebank | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 1084 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 3 | 11.5 | 1 | 1 | 14.5 | 3 | van Billiard | 3 | 3 | small |
| test_full_fit | raw_family_surname_size_band | 1098 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 35.0 | 0 | 0 | 7.75 | 1 | McGowan | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 1175 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 9.0 | 1 | 1 | 15.2458 | 3 | Touma | 3 | 3 | small |
| test_full_fit | raw_family_surname_size_band | 1205 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 37.0 | 0 | 0 | 7.75 | 1 | Carr | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 1239 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 38.0 | 0 | 0 | 7.2292 | 1 | Whabee | 1 | 1 | alone |
| test_full_fit | raw_family_surname_size_band | 1251 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 30.0 | 1 | 0 | 15.55 | 2 | Lindell | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 1275 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 19.0 | 1 | 0 | 16.1 | 2 | McNamee | 2 | 2 | small |
| test_full_fit | raw_family_surname_size_band | 1284 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 3 | 13.0 | 0 | 2 | 20.25 | 3 | Abbott | 3 | 3 | small |
| test_full_fit | raw_family_surname_size_band | 1297 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 2 | 20.0 | 0 | 0 | 13.8625 | 1 | Nourney | 1 | 1 | alone |

## Prediction-rate sanity

- train survival rate: `0.383838`
- raw_tabular test pred_1_rate: `0.337321`
- candidate test pred_1_rate: `0.337321`
- candidate delta vs raw_tabular: `0.0`
- sanity flag: `OK`

## Decision

Final status: **REJECTED_TRAIN_SIDE**

Reason: train-side evidence is negative or weak under the predeclared Step 15/16-style decision rule.

## Validation guard

Active corrected Step 17B files should contain only plain size-band logic. Overlap-aware equality flags and prefixed overlap buckets are out of scope.

## Output files

- metrics CSV: `reports/17_family_surname_size_band_check.csv`
- row-level diff CSV: `reports/17_family_surname_size_band_diff_rows.csv`
