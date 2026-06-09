# 18 SurnameSurvival Fold-Safe Check

## Purpose

This is one fixed, predeclared `SurnameSurvival` validation check on top of the unchanged `raw_tabular / GradientBoostingClassifier` baseline.

Current public leader remains frozen: `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv`, public score `0.79904`. No submission was created.

## Feature definition

- `Surname` is the substring before the comma in `Name`.
- `SurnameSurvival` is a smoothed surname survival rate computed from labels.
- Formula: `(survived_sum + alpha * global_survival_rate) / (surname_count + alpha)`.
- Fixed parameters: `min_count=2`, `alpha=5.0`.
- Fallback is the train-fold global survival rate for OOF and the full-train global survival rate for test transform.

## Method boundary / anti-leakage notes

- This is a target-derived train-side validation lane, not part of the closed Family/Surname structural lane.
- For each CV fold, the surname map is fitted only on train-fold rows.
- Validation-fold labels are never used to build validation encodings.
- Validation rows receive train-fold map values; unknown surnames and train-fold counts below `min_count` receive the train-fold global survival rate.
- After OOF validation, the test audit fits the encoder on full train only and transforms test from that full-train map.
- Test labels are never used.
- PassengerId is not used as a predictive feature, lookup key, rule, or tuning input.
- No GradientBoostingClassifier hyperparameters, thresholds, gates, or public leaderboard choices are tuned here.
- FamilySurnameSizeBand, mismatch-only, Fare/FareLog, Title, Age, Ticket, Deck, CabinCount, and broad CabinKnown are not reopened.

## Baseline vs candidate feature sets

| variant | features | purpose |
| --- | --- | --- |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | unchanged raw_tabular GB baseline |
| raw_tabular_plus_surname_survival | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, SurnameSurvival | one fixed fold-safe smoothed surname survival-rate check |

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## OOF / test metrics

| model_name | variant | candidate_id | feature_set | features | cv_mean | cv_std | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | oof_accuracy | oof_accuracy_delta_vs_raw_tabular | oof_changed_rows | oof_changed_pct | oof_0_to_1 | oof_1_to_0 | rescue | kill | net | oof_pred_1_count | oof_pred_1_rate | oof_pred_1_rate_delta_vs_raw_tabular | test_changed_rows | test_changed_pct | test_0_to_1 | test_1_to_0 | test_pred_1_count | test_pred_1_rate | test_pred_1_rate_delta_vs_raw_tabular | train_survival_rate | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | raw_tabular | raw_tabular__GradientBoostingClassifier | raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | 0.827161 | 0.020218 | 0.826816 | 0.859551 | 0.808989 | 0.803371 | 0.837079 | 0.82716 | 0.0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 294 | 0.329966 | 0.0 | 0 | 0.0 | 0 | 0 | 141 | 0.337321 | 0.0 | 0.383838 |  |
| GradientBoostingClassifier | raw_tabular_plus_surname_survival | raw_tabular_plus_surname_survival__GradientBoostingClassifier | raw_tabular_plus_surname_survival | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, SurnameSurvival | 0.814808 | 0.02492 | 0.821229 | 0.831461 | 0.792135 | 0.780899 | 0.848315 | 0.814815 | -0.012346 | 79 | 8.866442 | 56 | 23 | 34 | 45 | -11 | 327 | 0.367003 | 0.037037 | 40 | 9.569378 | 27 | 13 | 155 | 0.370813 | 0.033493 | 0.383838 | REJECT_TRAIN_SIDE |

## Required metric summary

- baseline OOF accuracy: `0.82716`
- candidate OOF accuracy: `0.814815`
- delta vs raw_tabular: `-0.012346`
- candidate CV mean/std: `0.814808` / `0.02492`
- OOF changed rows count: `79`
- OOF 0 -> 1 count: `56`
- OOF 1 -> 0 count: `23`
- rescue / kill / net: `34` / `45` / `-11`
- test changed rows count: `40`
- test 0 -> 1 count: `27`
- test 1 -> 0 count: `13`
- baseline test predicted survivors and rate: `141` / `0.337321`
- candidate test predicted survivors and rate: `155` / `0.370813`

## OOF changed-row audit

- changed PassengerIds: `3 9 26 35 41 50 80 101 114 115 120 133 140 142 143 149 156 166 173 183 217 234 242 255 262 263 296 316 329 340 347 348 363 374 376 405 416 424 431 432 435 448 451 475 485 490 504 506 513 531 535 542 543 545 555 565 568 573 579 600 605 611 613 622 633 672 685 697 808 814 831 840 849 853 856 868 883 889 890`

| diff_direction | rescue_or_kill | count | PassengerIds |
| --- | --- | --- | --- |
| upshift_0_to_1 | kill | 33 | 41 50 101 114 115 120 133 140 149 183 255 263 363 405 416 424 451 475 504 535 542 543 565 579 611 672 685 697 808 849 868 883 889 |
| upshift_0_to_1 | rescue | 23 | 3 9 26 80 142 143 217 234 242 262 316 329 348 431 432 485 531 555 605 613 622 831 856 |
| downshift_1_to_0 | kill | 12 | 166 173 347 376 448 490 513 573 600 633 840 890 |
| downshift_1_to_0 | rescue | 11 | 35 156 296 340 374 435 506 545 568 814 853 |

OOF row-level audit:

| split | PassengerId | Survived | raw_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvival | SurnameCountFromEncoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_oof | 3 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 26.0 | 0 | 0 | 7.925 | S | Heikkinen | 0.384292 | 0 |
| train_oof | 9 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 27.0 | 0 | 2 | 11.1333 | S | Johnson | 0.434938 | 4 |
| train_oof | 26 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 38.0 | 1 | 5 | 31.3875 | S | Asplund | 0.416349 | 2 |
| train_oof | 35 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 28.0 | 1 | 0 | 82.1708 | C | Meyer | 0.416734 | 2 |
| train_oof | 41 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 40.0 | 1 | 0 | 9.475 | S | Ahlin | 0.384292 | 0 |
| train_oof | 50 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 18.0 | 1 | 0 | 17.8 | S | Arnold-Franchi | 0.384292 | 1 |
| train_oof | 80 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 30.0 | 0 | 0 | 12.475 | S | Dowdell | 0.383427 | 0 |
| train_oof | 101 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 28.0 | 0 | 0 | 7.8958 | S | Petranec | 0.382889 | 0 |
| train_oof | 114 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 20.0 | 1 | 0 | 9.825 | S | Jussila | 0.416734 | 2 |
| train_oof | 115 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 17.0 | 0 | 0 | 14.4583 | C | Attalah | 0.383427 | 1 |
| train_oof | 120 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 2.0 | 4 | 2 | 31.275 | S | Andersson | 0.392146 | 5 |
| train_oof | 133 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 47.0 | 1 | 0 | 14.5 | S | Robins | 0.382889 | 0 |
| train_oof | 140 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 24.0 | 0 | 0 | 79.2 | C | Giglio | 0.382889 | 0 |
| train_oof | 142 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 22.0 | 0 | 0 | 7.75 | S | Nysten | 0.383427 | 0 |
| train_oof | 143 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 24.0 | 1 | 0 | 15.85 | S | Hakkarainen | 0.382889 | 0 |
| train_oof | 149 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 36.5 | 0 | 2 | 26.0 | S | Navratil | 0.559591 | 2 |
| train_oof | 156 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 51.0 | 0 | 1 | 61.3792 | C | Williams | 0.416734 | 2 |
| train_oof | 166 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 9.0 | 0 | 2 | 20.525 | S | Goldsmith | 0.384292 | 1 |
| train_oof | 173 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 3 | 1.0 | 1 | 1 | 11.1333 | S | Johnson | 0.365182 | 3 |
| train_oof | 183 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 9.0 | 4 | 2 | 31.3875 | S | Asplund | 0.615182 | 3 |
| train_oof | 217 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 27.0 | 0 | 0 | 7.925 | S | Honkanen | 0.382889 | 0 |
| train_oof | 234 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 5.0 | 4 | 2 | 31.3875 | S | Asplund | 0.490182 | 3 |
| train_oof | 242 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 15.5 | Q | Murphy | 0.384292 | 0 |
| train_oof | 255 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 41.0 | 0 | 2 | 20.2125 | S | Rosblom | 0.382889 | 1 |
| train_oof | 262 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 3 | 3.0 | 4 | 2 | 31.3875 | S | Asplund | 0.416349 | 2 |
| train_oof | 263 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 52.0 | 1 | 1 | 79.65 | S | Taussig | 0.560208 | 2 |
| train_oof | 296 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 |  | 0 | 0 | 27.7208 | C | Lewy | 0.382889 | 0 |
| train_oof | 316 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 26.0 | 0 | 0 | 7.8542 | S | Nilsson | 0.384292 | 0 |
| train_oof | 329 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 31.0 | 1 | 1 | 20.525 | S | Goldsmith | 0.416349 | 2 |
| train_oof | 340 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 45.0 | 0 | 0 | 35.5 | S | Blackwell | 0.384292 | 0 |
| train_oof | 347 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 2 | 40.0 | 0 | 0 | 13.0 | S | Smith | 0.274494 | 2 |
| train_oof | 348 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 16.1 | S | Davison | 0.383427 | 0 |
| train_oof | 363 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 45.0 | 0 | 1 | 14.4542 | C | Barbara | 0.383427 | 1 |
| train_oof | 374 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 22.0 | 0 | 0 | 135.6333 | C | Ringhini | 0.384292 | 0 |
| train_oof | 376 | 1 | 1 | 0 | downshift_1_to_0 | kill | female | 1 |  | 1 | 0 | 82.1708 | C | Meyer | 0.274494 | 2 |
| train_oof | 405 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 20.0 | 0 | 0 | 8.6625 | S | Oreskovic | 0.384292 | 1 |
| train_oof | 416 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 |  | 0 | 0 | 8.05 | S | Meek | 0.384292 | 0 |
| train_oof | 424 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 28.0 | 1 | 1 | 14.4 | S | Danbom | 0.382889 | 1 |
| train_oof | 431 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 1 | 28.0 | 0 | 0 | 26.55 | S | Bjornstrom-Steffansson | 0.384292 | 0 |
| train_oof | 432 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 16.1 | S | Thorneycroft | 0.382889 | 1 |
| train_oof | 435 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 50.0 | 1 | 0 | 55.9 | S | Silvey | 0.384292 | 1 |
| train_oof | 448 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 34.0 | 0 | 0 | 26.55 | S | Seward | 0.382889 | 0 |
| train_oof | 451 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 36.0 | 1 | 2 | 27.75 | S | West | 0.559207 | 2 |
| train_oof | 475 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 22.0 | 0 | 0 | 9.8375 | S | Strandberg | 0.384292 | 0 |
| train_oof | 485 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 1 | 25.0 | 1 | 0 | 91.0792 | C | Bishop | 0.382889 | 0 |
| train_oof | 490 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 3 | 9.0 | 1 | 1 | 15.9 | S | Coutts | 0.384292 | 1 |
| train_oof | 504 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 37.0 | 0 | 0 | 9.5875 | S | Laitinen | 0.384292 | 0 |
| train_oof | 506 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 18.0 | 1 | 0 | 108.9 | C | Penasco y Castellana | 0.384292 | 1 |
| train_oof | 513 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 36.0 | 0 | 0 | 26.2875 | S | McGough | 0.382889 | 0 |
| train_oof | 531 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 2 | 2.0 | 1 | 1 | 26.0 | S | Quick | 0.384292 | 1 |
| train_oof | 535 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 30.0 | 0 | 0 | 8.6625 | S | Cacic | 0.384292 | 0 |
| train_oof | 542 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 9.0 | 4 | 2 | 31.275 | S | Andersson | 0.392146 | 5 |
| train_oof | 543 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 11.0 | 4 | 2 | 31.275 | S | Andersson | 0.392146 | 5 |
| train_oof | 545 | 0 | 1 | 0 | downshift_1_to_0 | rescue | male | 1 | 50.0 | 1 | 0 | 106.425 | C | Douglas | 0.384292 | 0 |
| train_oof | 555 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 22.0 | 0 | 0 | 7.775 | S | Ohman | 0.382889 | 0 |
| train_oof | 565 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 |  | 0 | 0 | 8.05 | S | Meanwell | 0.382889 | 0 |
| train_oof | 568 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 29.0 | 0 | 4 | 21.075 | S | Palsson | 0.240182 | 3 |
| train_oof | 573 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 36.0 | 0 | 0 | 26.3875 | S | Flynn | 0.274494 | 2 |
| train_oof | 579 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 |  | 1 | 0 | 14.4583 | C | Caram | 0.384292 | 0 |
| train_oof | 600 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 49.0 | 1 | 0 | 56.9292 | C | Duff Gordon | 0.384292 | 1 |
| train_oof | 605 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 1 | 35.0 | 0 | 0 | 26.55 | C | Homer | 0.384292 | 0 |
| train_oof | 611 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 39.0 | 1 | 5 | 31.275 | S | Andersson | 0.392146 | 5 |
| train_oof | 613 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 |  | 1 | 0 | 15.5 | Q | Murphy | 0.384292 | 0 |
| train_oof | 622 | 1 | 0 | 1 | upshift_0_to_1 | rescue | male | 1 | 42.0 | 1 | 0 | 52.5542 | S | Kimball | 0.384292 | 0 |
| train_oof | 633 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 32.0 | 0 | 0 | 30.5 | C | Stahelin-Maeglin | 0.384292 | 0 |
| train_oof | 672 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 31.0 | 1 | 0 | 52.0 | S | Davidson | 0.384292 | 0 |
| train_oof | 685 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 60.0 | 1 | 1 | 39.0 | S | Brown | 0.615182 | 3 |
| train_oof | 697 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 3 | 44.0 | 0 | 0 | 8.05 | S | Kelly | 0.614642 | 3 |
| train_oof | 808 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 18.0 | 0 | 0 | 7.775 | S | Pettersson | 0.382889 | 0 |
| train_oof | 814 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 6.0 | 4 | 2 | 31.275 | S | Andersson | 0.301111 | 8 |
| train_oof | 831 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 15.0 | 1 | 0 | 14.4542 | C | Yasbeck | 0.383427 | 1 |
| train_oof | 840 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 |  | 0 | 0 | 29.7 | C | Marechal | 0.383427 | 0 |
| train_oof | 849 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 2 | 28.0 | 0 | 1 | 33.0 | S | Harper | 0.614642 | 3 |
| train_oof | 853 | 0 | 1 | 0 | downshift_1_to_0 | rescue | female | 3 | 9.0 | 1 | 1 | 15.2458 | C | Boulos | 0.274494 | 2 |
| train_oof | 856 | 1 | 0 | 1 | upshift_0_to_1 | rescue | female | 3 | 18.0 | 0 | 1 | 9.35 | S | Aks | 0.382889 | 0 |
| train_oof | 868 | 0 | 0 | 1 | upshift_0_to_1 | kill | male | 1 | 31.0 | 0 | 0 | 50.4958 | S | Roebling | 0.384292 | 0 |
| train_oof | 883 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 | 22.0 | 0 | 0 | 10.5167 | S | Dahlberg | 0.384292 | 0 |
| train_oof | 889 | 0 | 0 | 1 | upshift_0_to_1 | kill | female | 3 |  | 1 | 2 | 23.45 | S | Johnston | 0.384292 | 1 |
| train_oof | 890 | 1 | 1 | 0 | downshift_1_to_0 | kill | male | 1 | 26.0 | 0 | 0 | 30.0 | C | Behr | 0.384292 | 0 |

## Test full-fit diff audit

- changed PassengerIds: `896 899 913 915 920 926 928 933 960 972 979 1010 1011 1034 1036 1040 1045 1050 1051 1057 1091 1093 1109 1134 1141 1160 1175 1176 1200 1205 1206 1245 1251 1259 1271 1274 1275 1296 1297 1309`

| diff_direction | rescue_or_kill | count | PassengerIds |
| --- | --- | --- | --- |
| upshift_0_to_1 | test_changed | 27 | 896 899 920 928 960 979 1010 1034 1036 1045 1050 1051 1057 1091 1109 1141 1160 1175 1200 1245 1251 1259 1271 1274 1275 1296 1309 |
| downshift_1_to_0 | test_changed | 13 | 913 915 926 933 972 1011 1040 1093 1134 1176 1205 1206 1297 |

Test row-level audit:

| split | PassengerId | Survived | raw_pred | candidate_pred | diff_direction | rescue_or_kill | Sex | Pclass | Age | SibSp | Parch | Fare | Embarked | Surname | SurnameSurvival | SurnameCountFromEncoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_full_fit | 896 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 22.0 | 1 | 1 | 12.2875 | S | Hirvonen | 0.383838 | 1 |
| test_full_fit | 899 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 2 | 26.0 | 1 | 1 | 29.0 | S | Caldwell | 0.559885 | 2 |
| test_full_fit | 913 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 3 | 9.0 | 0 | 1 | 3.1708 | S | Olsen | 0.239899 | 3 |
| test_full_fit | 915 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 | 21.0 | 0 | 1 | 61.3792 | C | Williams | 0.324355 | 4 |
| test_full_fit | 920 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 41.0 | 0 | 0 | 30.5 | S | Brady | 0.383838 | 0 |
| test_full_fit | 926 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 | 30.0 | 1 | 0 | 57.75 | C | Mock | 0.383838 | 0 |
| test_full_fit | 928 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 |  | 0 | 0 | 8.05 | S | Roth | 0.383838 | 0 |
| test_full_fit | 933 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 |  | 0 | 0 | 26.55 | S | Franklin | 0.383838 | 0 |
| test_full_fit | 960 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 31.0 | 0 | 0 | 28.5375 | C | Tucker | 0.383838 | 0 |
| test_full_fit | 972 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 3 | 6.0 | 1 | 1 | 15.2458 | C | Boulos | 0.239899 | 3 |
| test_full_fit | 979 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 18.0 | 0 | 0 | 8.05 | S | Badman | 0.383838 | 0 |
| test_full_fit | 1010 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 36.0 | 0 | 0 | 75.2417 | C | Beattie | 0.383838 | 0 |
| test_full_fit | 1011 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 2 | 29.0 | 1 | 0 | 26.0 | S | Chapman | 0.27417 | 2 |
| test_full_fit | 1034 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 61.0 | 1 | 3 | 262.375 | C | Ryerson | 0.559885 | 2 |
| test_full_fit | 1036 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 42.0 | 0 | 0 | 26.55 | S | Lindeberg-Lind | 0.383838 | 0 |
| test_full_fit | 1040 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 |  | 0 | 0 | 26.55 | S | Crafton | 0.383838 | 0 |
| test_full_fit | 1045 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 36.0 | 0 | 2 | 12.1833 | S | Klasen | 0.383838 | 1 |
| test_full_fit | 1050 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 42.0 | 0 | 0 | 26.55 | S | Borebank | 0.383838 | 0 |
| test_full_fit | 1051 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 26.0 | 0 | 2 | 13.775 | S | Peacock | 0.383838 | 0 |
| test_full_fit | 1057 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 26.0 | 1 | 1 | 22.025 | S | Kink-Heilmann | 0.383838 | 1 |
| test_full_fit | 1091 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 |  | 0 | 0 | 8.1125 | S | Rasmussen | 0.383838 | 0 |
| test_full_fit | 1093 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 3 | 0.33 | 0 | 2 | 14.4 | S | Danbom | 0.27417 | 2 |
| test_full_fit | 1109 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 57.0 | 1 | 1 | 164.8667 | S | Wick | 0.559885 | 2 |
| test_full_fit | 1134 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 1 | 45.0 | 1 | 1 | 134.5 | C | Spedden | 0.383838 | 1 |
| test_full_fit | 1141 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 |  | 1 | 0 | 14.4542 | C | Khalil | 0.383838 | 0 |
| test_full_fit | 1160 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 |  | 0 | 0 | 8.05 | S | Howard | 0.383838 | 0 |
| test_full_fit | 1175 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 9.0 | 1 | 1 | 15.2458 | C | Touma | 0.383838 | 1 |
| test_full_fit | 1176 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 2.0 | 1 | 1 | 20.2125 | S | Rosblom | 0.27417 | 2 |
| test_full_fit | 1200 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 55.0 | 1 | 1 | 93.5 | S | Hays | 0.559885 | 2 |
| test_full_fit | 1205 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 3 | 37.0 | 0 | 0 | 7.75 | Q | Carr | 0.383838 | 1 |
| test_full_fit | 1206 |  | 1 | 0 | downshift_1_to_0 | test_changed | female | 1 | 55.0 | 0 | 0 | 135.6333 | C | White | 0.27417 | 2 |
| test_full_fit | 1245 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 2 | 49.0 | 1 | 2 | 65.0 | S | Herman | 0.559885 | 2 |
| test_full_fit | 1251 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 30.0 | 1 | 0 | 15.55 | S | Lindell | 0.383838 | 1 |
| test_full_fit | 1259 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 22.0 | 0 | 0 | 39.6875 | S | Riihivouri | 0.383838 | 0 |
| test_full_fit | 1271 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 3 | 5.0 | 4 | 2 | 31.3875 | S | Asplund | 0.546577 | 4 |
| test_full_fit | 1274 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 |  | 0 | 0 | 14.5 | S | Risien | 0.383838 | 1 |
| test_full_fit | 1275 |  | 0 | 1 | upshift_0_to_1 | test_changed | female | 3 | 19.0 | 1 | 0 | 16.1 | S | McNamee | 0.383838 | 1 |
| test_full_fit | 1296 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 1 | 43.0 | 1 | 0 | 27.7208 | C | Frauenthal | 0.559885 | 2 |
| test_full_fit | 1297 |  | 1 | 0 | downshift_1_to_0 | test_changed | male | 2 | 20.0 | 0 | 0 | 13.8625 | C | Nourney | 0.383838 | 0 |
| test_full_fit | 1309 |  | 0 | 1 | upshift_0_to_1 | test_changed | male | 3 |  | 1 | 1 | 22.3583 | C | Peter | 0.559885 | 2 |

## Decision

Decision: **REJECT_TRAIN_SIDE**

Reason: candidate does not satisfy the fixed train-side rule requiring OOF improvement and positive rescue/kill/net evidence.

## Submission status

No submission was created.

## Output files

- report: `reports/18_surname_survival_foldsafe_check.md`
