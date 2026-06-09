# 19B Tuned GB Raw Tabular Submission Review

## Purpose

Step 19B creates one public-check submission for the best Step 19A `raw_tabular / GradientBoostingClassifier` tuning candidate.

This report records the public result for this specific standalone tuned raw_tabular GB public-check candidate.

## Method boundary

- Fit one tuned `GradientBoostingClassifier` on full train only.
- Feature set is pure `raw_tabular`: `Sex, Pclass, Embarked, Age, SibSp, Parch, Fare`.
- No new tuning, threshold changes, post-processing, overlays, SurnameSurvival, CabinKnown gate, or public-score selection are used.
- PassengerId is used only as the required submission row identifier and validation key; it is not used as a rule, feature, lookup key, or tuning input.
- CabinKnown gate re-check is explicitly out of scope for Step 19B.

## Source candidate from Step 19A

| candidate_id | oof_accuracy | delta_vs_default_raw_tabular_gb | rescue_kill_net | test_changed_rows_vs_default_full_fit | test_survivors | test_survivor_rate | diagnostic_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stage2_d3_leaf5_split10_mfnone_lr0p07_n150 | 0.843996 | 0.016835 | 21 / 6 / +15 | 18 | 139 | 0.332536 | OOF_POSITIVE / NO_SUBMISSION / PUBLIC_UNKNOWN |

Source candidate audit match after 19B full-fit prediction: `PASS`.

## Model params

| param | value |
| --- | --- |
| loss | log_loss |
| learning_rate | 0.07 |
| n_estimators | 150 |
| max_depth | 3 |
| min_samples_leaf | 5 |
| min_samples_split | 10 |
| max_features | None |
| subsample | 1.0 |
| ccp_alpha | 0.0 |
| random_state | 42 |

## Feature set

| feature_set | features | preprocessing |
| --- | --- | --- |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | make_preprocessor("unscaled_tree", RAW_TABULAR) |

## Anti-leakage notes

- Train labels are used only to fit the full-train supervised model.
- Test labels are not used.
- Public score is not used.
- PassengerId is not used as a model feature or rule.
- `make_preprocessor("unscaled_tree", RAW_TABULAR)` is the only preprocessing pipeline used.

## Submission file path

`submissions/submission_19b_tuned_gb_raw_tabular.csv`

## Submission validation

| check | status | detail |
| --- | --- | --- |
| file exists | PASS | submissions/submission_19b_tuned_gb_raw_tabular.csv |
| columns exactly PassengerId,Survived | PASS | PassengerId, Survived |
| row count equals test row count | PASS | 418 / 418 |
| PassengerId matches test order | PASS | same order and values |
| Survived has no NaN | PASS | nan_count=0 |
| Survived values are 0/1 | PASS | 0,1 |
| Survived dtype is integer | PASS | int64 |
| PassengerId has no duplicates | PASS | duplicate_count=0 |

## Prediction audit

| metric | value |
| --- | --- |
| test row count | 418 |
| predicted survivors count | 139 |
| predicted survivors rate | 0.332536 |
| predicted died count | 279 |
| default raw_tabular GB predicted survivors count | 141 |
| default raw_tabular GB predicted survivors rate | 0.337321 |
| CabinKnown gate applied | NO |
| SurnameSurvival applied | NO |
| PassengerId rule used | NO |

## Comparison vs default raw_tabular GB full-fit prediction

| comparison | changed_rows | 0_to_1 | 1_to_0 |
| --- | --- | --- | --- |
| submission_19b_tuned_gb_raw_tabular vs default raw_tabular GB full-fit prediction | 18 | 8 | 10 |

This comparison is audit-only. It does not change the selected candidate and does not use public score.

## Public result

Submission: submissions/submission_19b_tuned_gb_raw_tabular.csv

Public score: 0.79186

Status: REJECT_PUBLIC_TRANSFER

## Public comparison

default raw_tabular GB public score: 0.79665

current frozen public leader public score: 0.79904

19B tuned raw_tabular GB public score: 0.79186

## Comparison vs frozen public leader as reference only

| metric | value |
| --- | --- |
| frozen public leader path | submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv |
| frozen public leader public score | 0.79904 |
| used for 19B selection logic | NO |
| modified by 19B | NO |
| frozen public leader file exists | YES |
| frozen public leader shape/order valid | YES |
| frozen public leader predicted survivors count | 138 |
| frozen public leader predicted survivors rate | 0.330144 |

| comparison | changed_rows | 0_to_1 | 1_to_0 |
| --- | --- | --- | --- |
| submission_19b_tuned_gb_raw_tabular vs frozen public leader reference | 19 | 10 | 9 |

The frozen public leader is a benchmark reference only. It is not used in Step 19B selection logic and is not modified.

## Final interpretation

The strong train-side OOF signal from Step 19A did not transfer to public for this standalone tuned raw_tabular GB candidate.

Step 19B is rejected by public transfer.

This does not prove that all GB tuning is useless. It only rejects this specific standalone tuned raw_tabular GB public-check candidate.

Current frozen public leader remains:

submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv

public score 0.79904

## Model panel

| model_class | package | package_version | preprocessing_mode | explicit_tuned_params | actual_resolved_params |
| --- | --- | --- | --- | --- | --- |
| GradientBoostingClassifier | scikit-learn | 1.9.0 | unscaled_tree | {"ccp_alpha": 0.0, "learning_rate": 0.07, "loss": "log_loss", "max_depth": 3, "max_features": null, "min_samples_leaf": 5, "min_samples_split": 10, "n_estimators": 150, "random_state": 42, "subsample": 1.0} | {"ccp_alpha": 0.0, "criterion": "deprecated", "init": null, "learning_rate": 0.07, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 5, "min_samples_split": 10, "min_weight_fraction_leaf": 0.0, "n_estimators": 150, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |

## Submission status

Submission created: `submissions/submission_19b_tuned_gb_raw_tabular.csv`.

Public status: `REJECT_PUBLIC_TRANSFER` with public score `0.79186`.

Predicted survivors: `139` / `418`; rate `0.332536`.

No CabinKnown gate, SurnameSurvival, overlay, post-processing rule, threshold adjustment, or PassengerId rule was applied.

## Next-step boundary

After public result, the next possible separate step is: `re-check CabinKnown subgroup gate on tuned raw_tabular GB`.

Do not do that in Step 19B.

## Output files

- script: `scripts/19b_create_tuned_gb_raw_tabular_submission.py`
- report: `reports/19b_tuned_gb_raw_tabular_submission_review.md`
- submission: `submissions/submission_19b_tuned_gb_raw_tabular.csv`
