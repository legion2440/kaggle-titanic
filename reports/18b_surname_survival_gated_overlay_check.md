# 18B SurnameSurvival Gated Overlay Check

## Purpose

This diagnostic replaces the old Step 18B delta feature check with a controlled post-model overlay. The baseline model remains the unchanged `raw_tabular / GradientBoostingClassifier`; the candidate copies baseline predictions and applies one surname rule only on active repeated-surname rows.

Current public leader remains frozen: `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv`, public score `0.79904`.

## Why old 18B was removed/replaced

The old Step 18B `SurnameSurvivalDelta` fallback-neutral check still trained `GradientBoostingClassifier` with a target-derived surname feature. That did not isolate a surname correction path because the model could reshape its full decision surface. This replacement keeps GB unchanged and applies a bounded overlay after prediction.

## Method boundary

- This is diagnostic only and not public tuning.
- The candidate is not a retrained model with extra features.
- `raw_tabular` baseline features remain `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`.
- `GradientBoostingClassifier` hyperparameters are unchanged.
- Existing submissions and the frozen public leader are not altered.
- PassengerId is not used as a rule, feature, lookup key, or tuning input.
- Step 17 structural docs and closed structural lanes are not reopened.

## Anti-leakage / fold-safe notes

- For each CV fold, the baseline GB model is trained only on train-fold rows.
- Validation-fold baseline predictions are OOF predictions from that fold model.
- For each CV fold, the surname encoder is fitted only on train-fold rows.
- Validation-fold labels are never used to build surname encoding.
- Unknown surnames and train-fold surname counts below `min_count` are inactive and keep baseline predictions.
- OOF `global_survival_rate` is fold-specific.
- Test overlay is transformed from a full-train surname map after OOF validation.
- Test labels are never used.

## Overlay rule

- `Surname` is the substring before comma in `Name`.
- Smoothed surname survival rate: `(survived_sum + alpha * global_survival_rate) / (surname_count + alpha)`.
- Fixed parameters: `min_count=2`, `alpha=5.0`.
- `active = SurnameCountFromEncoder >= min_count`.
- If inactive: `candidate_pred = baseline_pred`.
- If active and `SurnameSurvivalRate >= 0.5`: `candidate_pred = 1`.
- If active and `SurnameSurvivalRate <= global_survival_rate`: `candidate_pred = 0`.
- Otherwise: `candidate_pred = baseline_pred`.
- Thresholds are fixed and not tuned.

## Baseline metrics

| variant | model_or_candidate | features | cv_mean | cv_std | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | test_changed_rows | test_0_to_1 | test_1_to_0 | test_pred_1_count | test_pred_1_rate | diagnostic_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular | GradientBoostingClassifier | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.825701 | 0.021513 | 0.82716 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 141 | 0.337321 |  |

## Overlay candidate metrics

| variant | model_or_candidate | features | cv_mean | cv_std | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | test_changed_rows | test_0_to_1 | test_1_to_0 | test_pred_1_count | test_pred_1_rate | diagnostic_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular_plus_surname_survival_gated_overlay | post-model gated overlay | raw_tabular baseline predictions + SurnameSurvival gated overlay | 0.812579 | 0.021729 | 0.814815 | -0.012346 | 17 | 8 | 9 | 3 | 14 | -11 | 22 | 13 | 9 | 145 | 0.34689 | OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |

## Required metric summary

- baseline OOF accuracy: `0.82716`
- candidate OOF accuracy: `0.814815`
- delta vs raw_tabular: `-0.012346`
- OOF changed rows: `17`
- OOF 0 -> 1: `8`
- OOF 1 -> 0: `9`
- rescue / kill / net: `3` / `14` / `-11`
- test changed rows: `22`
- test 0 -> 1: `13`
- test 1 -> 0: `9`
- baseline test predicted survivors and rate: `141` / `0.337321`
- candidate test predicted survivors and rate: `145` / `0.34689`

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Safety invariant

- OOF changed rows where `SurnameOverlayActive == 0`: `0`
- test changed rows where `SurnameOverlayActive == 0`: `0`
- invariant: `No inactive row changed prediction.`

## OOF diff audit

| PassengerId | Survived | baseline_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvivalRate | SurnameCountFromEncoder | SurnameOverlayActive | SurnameOverlayDirection | SurnameOverlayReason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 149 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 36.5 | 0 | 2 | 26.0 | S | Navratil | 0.559591 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 173 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 3 | 1.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |
| 183 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 9.0 | 4 | 2 | 31.3875 | S | Asplund | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0_5 |
| 250 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 54.0 | 1 | 0 | 26.0 | S | Carter | 0.546348 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 263 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 52.0 | 1 | 1 | 79.65 | S | Taussig | 0.560208 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 342 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 1 | 24.0 | 3 | 2 | 263.0 | S | Fortune | 0.364642 | 3 | 1 | set_to_0 | active_rate_le_global |
| 347 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 2 | 40.0 | 0 | 0 | 13.0 | S | Smith | 0.274494 | 2 | 1 | set_to_0 | active_rate_le_global |
| 376 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 1 |  | 1 | 0 | 82.1708 | C | Meyer | 0.274494 | 2 | 1 | set_to_0 | active_rate_le_global |
| 451 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 36.0 | 1 | 2 | 27.75 | S | West | 0.559207 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | S | Palsson | 0.240182 | 3 | 1 | set_to_0 | active_rate_le_global |
| 573 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 36.0 | 0 | 0 | 26.3875 | S | Flynn | 0.274494 | 2 | 1 | set_to_0 | active_rate_le_global |
| 685 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 60.0 | 1 | 1 | 39.0 | S | Brown | 0.615182 | 3 | 1 | set_to_1 | active_rate_ge_0_5 |
| 697 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 44.0 | 0 | 0 | 8.05 | S | Kelly | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0_5 |
| 814 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 6.0 | 4 | 2 | 31.275 | S | Andersson | 0.301111 | 8 | 1 | set_to_0 | active_rate_le_global |
| 849 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 28.0 | 0 | 1 | 33.0 | S | Harper | 0.614642 | 3 | 1 | set_to_1 | active_rate_ge_0_5 |
| 853 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 9.0 | 1 | 1 | 15.2458 | C | Boulos | 0.274494 | 2 | 1 | set_to_0 | active_rate_le_global |
| 870 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 4.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 | 1 | set_to_0 | active_rate_le_global |

## Test diff audit

| PassengerId | baseline_pred | candidate_pred | diff_direction | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvivalRate | SurnameCountFromEncoder | SurnameOverlayActive | SurnameOverlayDirection | SurnameOverlayReason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 892 | 0 | 1 | upshift_0_to_1 | male | 3 | 34.5 | 0 | 0 | 7.8292 | Q | Kelly | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 899 | 0 | 1 | upshift_0_to_1 | male | 2 | 26.0 | 1 | 1 | 29.0 | S | Caldwell | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 913 | 1 | 0 | downshift_1_to_0 | male | 3 | 9.0 | 0 | 1 | 3.1708 | S | Olsen | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| 915 | 1 | 0 | downshift_1_to_0 | male | 1 | 21.0 | 0 | 1 | 61.3792 | C | Williams | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| 972 | 1 | 0 | downshift_1_to_0 | male | 3 | 6.0 | 1 | 1 | 15.2458 | C | Boulos | 0.239899 | 3 | 1 | set_to_0 | active_rate_le_global |
| 1011 | 1 | 0 | downshift_1_to_0 | female | 2 | 29.0 | 1 | 0 | 26.0 | S | Chapman | 0.27417 | 2 | 1 | set_to_0 | active_rate_le_global |
| 1034 | 0 | 1 | upshift_0_to_1 | male | 1 | 61.0 | 1 | 3 | 262.375 | C | Ryerson | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1046 | 0 | 1 | upshift_0_to_1 | male | 3 | 13.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1066 | 0 | 1 | upshift_0_to_1 | male | 3 | 40.0 | 1 | 5 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1093 | 1 | 0 | downshift_1_to_0 | male | 3 | 0.33 | 0 | 2 | 14.4 | S | Danbom | 0.27417 | 2 | 1 | set_to_0 | active_rate_le_global |
| 1109 | 0 | 1 | upshift_0_to_1 | male | 1 | 57.0 | 1 | 1 | 164.8667 | S | Wick | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1118 | 0 | 1 | upshift_0_to_1 | male | 3 | 23.0 | 0 | 0 | 7.7958 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1176 | 1 | 0 | downshift_1_to_0 | female | 3 | 2.0 | 1 | 1 | 20.2125 | S | Rosblom | 0.27417 | 2 | 1 | set_to_0 | active_rate_le_global |
| 1183 | 0 | 1 | upshift_0_to_1 | female | 3 | 30.0 | 0 | 0 | 6.95 | Q | Daly | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1200 | 0 | 1 | upshift_0_to_1 | male | 1 | 55.0 | 1 | 1 | 93.5 | S | Hays | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1206 | 1 | 0 | downshift_1_to_0 | female | 1 | 55.0 | 0 | 0 | 135.6333 | C | White | 0.27417 | 2 | 1 | set_to_0 | active_rate_le_global |
| 1222 | 1 | 0 | downshift_1_to_0 | female | 2 | 48.0 | 0 | 2 | 36.75 | S | Davies | 0.364899 | 3 | 1 | set_to_0 | active_rate_le_global |
| 1245 | 0 | 1 | upshift_0_to_1 | male | 2 | 49.0 | 1 | 2 | 65.0 | S | Herman | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1271 | 0 | 1 | upshift_0_to_1 | male | 3 | 5.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1287 | 1 | 0 | downshift_1_to_0 | female | 1 | 18.0 | 1 | 0 | 60.0 | S | Smith | 0.324355 | 4 | 1 | set_to_0 | active_rate_le_global |
| 1296 | 0 | 1 | upshift_0_to_1 | male | 1 | 43.0 | 1 | 0 | 27.7208 | C | Frauenthal | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |
| 1309 | 0 | 1 | upshift_0_to_1 | male | 3 |  | 1 | 1 | 22.3583 | C | Peter | 0.559885 | 2 | 1 | set_to_1 | active_rate_ge_0_5 |

## Diff-by-active diagnostic

| split | active_changed_rows | inactive_changed_rows | active_0_to_1 | active_1_to_0 | active_rescue | active_kill | active_net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OOF | 17 | 0 | 8 | 9 | 3 | 14 | -11 |
| test | 22 | 0 | 13 | 9 |  |  |  |

## Decision wording

Diagnostic status: **OOF_NEGATIVE / NO_SUBMISSION / PUBLIC_UNKNOWN**

This does not claim public transfer, does not claim public score is worse or better, and does not auto-close the SurnameSurvival lane. Final project decision is not automatic.

## Deleted old artifact note

- `scripts/18b_surname_survival_delta_fallback_neutral_check.py`: `absent / removed`
- `reports/18b_surname_survival_delta_fallback_neutral_check.md`: `absent / removed`

## Submission status

No submission was created.

## Output files

- report: `reports/18b_surname_survival_gated_overlay_check.md`
