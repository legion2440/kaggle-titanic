# 18C SurnameSurvival Directional Overlay Check

## Purpose

This diagnostic checks whether a narrow surname target-derived correction exists when the baseline `raw_tabular / GradientBoostingClassifier` predictions are kept fixed and only predeclared post-model overlays are applied.

Current public leader remains frozen: `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv`, public score `0.79904`.

## Why Step 18C exists

Step 18 tested `SurnameSurvival` as a normal GB feature. Step 18B moved to a controlled post-model overlay and passed the inactive-row invariant, but the broad active overlay was train-side negative. Step 18C keeps the post-model overlay approach, raises the active-count requirement, and separates broad, upshift-only, downshift-only, and strict directional rules.

## Why min_count = 3

`min_count = 2` is no longer used because two train-fold records are too weak a basis for a target-derived surname overlay. Step 18C uses one predeclared stricter gate, `min_count = 3`, without comparing or tuning against `min_count = 2`.

## Method boundary

- This is diagnostic only and not public tuning.
- No candidate trains a new GB model with `SurnameSurvival` as a feature.
- The baseline model and baseline features are unchanged.
- `GradientBoostingClassifier` hyperparameters are unchanged.
- Existing submissions and the frozen public leader are not altered.
- PassengerId is not used as a rule, feature, lookup key, or tuning input.
- Step 17 structural docs and closed structural lanes are not reopened.

## Anti-leakage / fold-safe notes

- For each CV fold, baseline GB is trained only on train-fold rows.
- Validation-fold baseline predictions are true OOF predictions.
- For each CV fold, the surname encoder is fitted only on train-fold rows.
- Validation-fold labels are never used to build surname encoding.
- Validation rows receive surname stats from the train-fold map.
- Unknown surnames and train-fold surname counts below `min_count = 3` are inactive and keep baseline predictions.
- OOF `global_survival_rate` is fold-specific.
- Test overlay is transformed from a full-train surname map after OOF validation.
- Test labels are never used.

## Candidate overlay rules

| candidate | rule | positive_threshold | negative_threshold |
| --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | active count >= 3; rate >= 0.5 set to 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline | 0.5 | global |
| surname_overlay_upshift_only_min3 | active count >= 3; baseline_pred == 0; rate >= 0.5 set to 1; otherwise keep baseline | 0.5 | none |
| surname_overlay_downshift_only_min3 | active count >= 3; baseline_pred == 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline | none | global |
| surname_overlay_downshift_strict_min3 | active count >= 3; baseline_pred == 1; rate <= 0.30 set to 0; otherwise keep baseline | none | 0.3 |
| surname_overlay_upshift_strict_min3 | active count >= 3; baseline_pred == 0; rate >= 0.60 set to 1; otherwise keep baseline | 0.6 | none |

## Baseline metrics

| variant | model | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | test_pred_1_count | test_pred_1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular | GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.827161 | 0.020218 | 0.826816 | 0.859551 | 0.808989 | 0.803371 | 0.837079 | 0.82716 | 141 | 0.337321 |

## Candidate metrics table

| candidate | rule | baseline_oof_accuracy | candidate_oof_accuracy | delta_vs_raw_tabular | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_changed_rows | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | test_changed_rows | test_0_to_1 | test_1_to_0 | baseline_test_pred_1_count | baseline_test_pred_1_rate | candidate_test_pred_1_count | candidate_test_pred_1_rate | inactive_changed_rows_oof | inactive_changed_rows_test | diagnostic_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | active count >= 3; rate >= 0.5 set to 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline | 0.82716 | 0.820426 | -0.006734 | 0.820444 | 0.025972 | 0.804469 | 0.859551 | 0.792135 | 0.803371 | 0.842697 | 10 | 5 | 5 | 2 | 8 | -6 | 10 | 5 | 5 | 141 | 0.337321 | 141 | 0.337321 | 0 | 0 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_upshift_only_min3 | active count >= 3; baseline_pred == 0; rate >= 0.5 set to 1; otherwise keep baseline | 0.82716 | 0.821549 | -0.005612 | 0.821562 | 0.023298 | 0.810056 | 0.859551 | 0.803371 | 0.797753 | 0.837079 | 5 | 5 | 0 | 0 | 5 | -5 | 5 | 5 | 0 | 141 | 0.337321 | 146 | 0.349282 | 0 | 0 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_downshift_only_min3 | active count >= 3; baseline_pred == 1; rate <= fold/full-train global survival rate set to 0; otherwise keep baseline | 0.82716 | 0.826038 | -0.001122 | 0.826044 | 0.022425 | 0.821229 | 0.859551 | 0.797753 | 0.808989 | 0.842697 | 5 | 0 | 5 | 2 | 3 | -1 | 5 | 0 | 5 | 141 | 0.337321 | 136 | 0.325359 | 0 | 0 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_downshift_strict_min3 | active count >= 3; baseline_pred == 1; rate <= 0.30 set to 0; otherwise keep baseline | 0.82716 | 0.828283 | 0.001122 | 0.828284 | 0.018983 | 0.826816 | 0.859551 | 0.808989 | 0.808989 | 0.837079 | 1 | 0 | 1 | 1 | 0 | 1 | 2 | 0 | 2 | 141 | 0.337321 | 139 | 0.332536 | 0 | 0 | OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_upshift_strict_min3 | active count >= 3; baseline_pred == 0; rate >= 0.60 set to 1; otherwise keep baseline | 0.82716 | 0.822671 | -0.004489 | 0.822679 | 0.022849 | 0.815642 | 0.859551 | 0.803371 | 0.797753 | 0.837079 | 4 | 4 | 0 | 0 | 4 | -4 | 0 | 0 | 0 | 141 | 0.337321 | 141 | 0.337321 | 0 | 0 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |

## Directional comparison table

| candidate | oof_changed_rows | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | test_changed_rows | test_0_to_1 | test_1_to_0 | test_survivor_count | test_survivor_rate | diagnostic_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | 10 | 5 | 5 | 2 | 8 | -6 | 10 | 5 | 5 | 141 | 0.337321 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_upshift_only_min3 | 5 | 5 | 0 | 0 | 5 | -5 | 5 | 5 | 0 | 146 | 0.349282 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_downshift_only_min3 | 5 | 0 | 5 | 2 | 3 | -1 | 5 | 0 | 5 | 136 | 0.325359 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_downshift_strict_min3 | 1 | 0 | 1 | 1 | 0 | 1 | 2 | 0 | 2 | 139 | 0.332536 | OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |
| surname_overlay_upshift_strict_min3 | 4 | 4 | 0 | 0 | 4 | -4 | 0 | 0 | 0 | 141 | 0.337321 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Inactive changed rows invariant

| candidate | inactive_changed_rows_oof | inactive_changed_rows_test | status |
| --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | 0 | 0 | PASS |
| surname_overlay_upshift_only_min3 | 0 | 0 | PASS |
| surname_overlay_downshift_only_min3 | 0 | 0 | PASS |
| surname_overlay_downshift_strict_min3 | 0 | 0 | PASS |
| surname_overlay_upshift_strict_min3 | 0 | 0 | PASS |

Every candidate satisfies: `No inactive row changed prediction.`

## OOF diff audit per candidate

| candidate | PassengerId | Survived | baseline_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvivalRate | SurnameCountFromEncoder | SurnameOverlayActive | SurnameOverlayDirection | SurnameOverlayReason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | 173 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 3 | 1.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 183 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 9.0 | 4 | 2 | 31.3875 | S | Asplund | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 250 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 54.0 | 1 | 0 | 26.0 | S | Carter | 0.546348 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 342 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 1 | 24.0 | 3 | 2 | 263.0 | S | Fortune | 0.364642 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | S | Palsson | 0.240182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 685 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 60.0 | 1 | 1 | 39.0 | S | Brown | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 697 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 44.0 | 0 | 0 | 8.05 | S | Kelly | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 814 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 6.0 | 4 | 2 | 31.275 | S | Andersson | 0.301111 | 8 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 849 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 28.0 | 0 | 1 | 33.0 | S | Harper | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 870 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 4.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_upshift_only_min3 | 183 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 9.0 | 4 | 2 | 31.3875 | S | Asplund | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 250 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 54.0 | 1 | 0 | 26.0 | S | Carter | 0.546348 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 685 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 60.0 | 1 | 1 | 39.0 | S | Brown | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 697 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 44.0 | 0 | 0 | 8.05 | S | Kelly | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 849 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 28.0 | 0 | 1 | 33.0 | S | Harper | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_downshift_only_min3 | 173 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 3 | 1.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 342 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 1 | 24.0 | 3 | 2 | 263.0 | S | Fortune | 0.364642 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | S | Palsson | 0.240182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 814 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 6.0 | 4 | 2 | 31.275 | S | Andersson | 0.301111 | 8 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 870 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 4.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_strict_min3 | 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | S | Palsson | 0.240182 | 3 | 1 | set_to_0 | active_rate_le_0.3 |
| surname_overlay_upshift_strict_min3 | 183 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 9.0 | 4 | 2 | 31.3875 | S | Asplund | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.6 |
| surname_overlay_upshift_strict_min3 | 685 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 60.0 | 1 | 1 | 39.0 | S | Brown | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0.6 |
| surname_overlay_upshift_strict_min3 | 697 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 44.0 | 0 | 0 | 8.05 | S | Kelly | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.6 |
| surname_overlay_upshift_strict_min3 | 849 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 28.0 | 0 | 1 | 33.0 | S | Harper | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0.6 |

## Test diff audit per candidate

| candidate | PassengerId | baseline_pred | candidate_pred | diff_direction | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvivalRate | SurnameCountFromEncoder | SurnameOverlayActive | SurnameOverlayDirection | SurnameOverlayReason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surname_overlay_broad_reference_min3 | 892 | 0 | 1 | upshift_0_to_1 | male | 3 | 34.5 | 0 | 0 | 7.8292 | Q | Kelly | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 913 | 1 | 0 | downshift_1_to_0 | male | 3 | 9.0 | 0 | 1 | 3.1708 | S | Olsen | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 915 | 1 | 0 | downshift_1_to_0 | male | 1 | 21.0 | 0 | 1 | 61.3792 | C | Williams | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 972 | 1 | 0 | downshift_1_to_0 | male | 3 | 6.0 | 1 | 1 | 15.2458 | C | Boulos | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 1046 | 0 | 1 | upshift_0_to_1 | male | 3 | 13.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 1066 | 0 | 1 | upshift_0_to_1 | male | 3 | 40.0 | 1 | 5 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 1118 | 0 | 1 | upshift_0_to_1 | male | 3 | 23.0 | 0 | 0 | 7.7958 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 1222 | 1 | 0 | downshift_1_to_0 | female | 2 | 48.0 | 0 | 2 | 36.75 | S | Davies | 0.364899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_broad_reference_min3 | 1271 | 0 | 1 | upshift_0_to_1 | male | 3 | 5.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_broad_reference_min3 | 1287 | 1 | 0 | downshift_1_to_0 | female | 1 | 18.0 | 1 | 0 | 60.0 | S | Smith | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_upshift_only_min3 | 892 | 0 | 1 | upshift_0_to_1 | male | 3 | 34.5 | 0 | 0 | 7.8292 | Q | Kelly | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 1046 | 0 | 1 | upshift_0_to_1 | male | 3 | 13.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 1066 | 0 | 1 | upshift_0_to_1 | male | 3 | 40.0 | 1 | 5 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 1118 | 0 | 1 | upshift_0_to_1 | male | 3 | 23.0 | 0 | 0 | 7.7958 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_upshift_only_min3 | 1271 | 0 | 1 | upshift_0_to_1 | male | 3 | 5.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0.5 |
| surname_overlay_downshift_only_min3 | 913 | 1 | 0 | downshift_1_to_0 | male | 3 | 9.0 | 0 | 1 | 3.1708 | S | Olsen | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 915 | 1 | 0 | downshift_1_to_0 | male | 1 | 21.0 | 0 | 1 | 61.3792 | C | Williams | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 972 | 1 | 0 | downshift_1_to_0 | male | 3 | 6.0 | 1 | 1 | 15.2458 | C | Boulos | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 1222 | 1 | 0 | downshift_1_to_0 | female | 2 | 48.0 | 0 | 2 | 36.75 | S | Davies | 0.364899 | 3 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_only_min3 | 1287 | 1 | 0 | downshift_1_to_0 | female | 1 | 18.0 | 1 | 0 | 60.0 | S | Smith | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| surname_overlay_downshift_strict_min3 | 913 | 1 | 0 | downshift_1_to_0 | male | 3 | 9.0 | 0 | 1 | 3.1708 | S | Olsen | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_0.3 |
| surname_overlay_downshift_strict_min3 | 972 | 1 | 0 | downshift_1_to_0 | male | 3 | 6.0 | 1 | 1 | 15.2458 | C | Boulos | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_0.3 |

## Interpretation

Step 18C does not make a public-transfer claim.

Among the predeclared `min_count=3` directional overlay diagnostics:

- `broad_reference_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.
- `upshift_only_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.
- `downshift_only_min3` is close to train-side neutral but still OOF-negative by the current criterion; public transfer remains unknown. OOF delta `-0.001122`, net `-1`.
- `downshift_strict_min3` is the only OOF-positive diagnostic candidate, with a small controlled OOF signal; public transfer remains unknown. OOF delta `0.001122`, net `1`.
- `upshift_strict_min3` is OOF-negative under the diagnostic criterion; public transfer remains unknown.

These statuses do not prove public performance. They only describe train-side OOF diagnostics under the predeclared overlay rules.

## Diagnostic status

Diagnostic status is assigned per candidate in the tables above:

- `OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN` if OOF delta > 0 and net > 0.
- `OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN` if OOF delta < 0 or net < 0.
- `OOF_NEUTRAL / NO_SUBMISSION / PUBLIC_UNKNOWN` otherwise.

OOF status is only a diagnostic train-side status. Public transfer remains unknown. Submission was not created. Final project decision is not automatic.

Candidate wording guide:

- OOF-negative candidates: This candidate is OOF-negative under the current predeclared diagnostic criterion. Public transfer remains unknown.
- Near-neutral candidates: This candidate is near-neutral on train-side diagnostics. Public transfer remains unknown.
- OOF-positive candidates: This candidate has a small OOF-positive diagnostic signal under the current predeclared rule. Public transfer remains unknown.

## Submission status

No submission was created.

## Output files

- report: `reports/18c_surname_survival_directional_overlay_check.md`
