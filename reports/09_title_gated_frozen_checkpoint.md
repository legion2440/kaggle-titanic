# 09 Title Gated Frozen Checkpoint

## Scope Boundary

- full `train.csv` model fitting is allowed for frozen checkpoint file generation
- `test.csv` is used only for inference
- only one preselected conservative blend candidate is evaluated
- base feature set is `raw_tabular`
- title feature set is `raw_plus_title`
- blend weight is fixed at `0.10`
- `add_clean_features()` is used only to create `Title`; only `Title` is copied from its output
- existing preprocessing is used through `scripts.preprocessing.make_preprocessor`
- same technical/default model parameters as `04_baseline` and `08_title_gating_check`
- `predict_proba()` is used only for base/title probabilities
- no `gender_submission.csv` as truth
- no test labels or row-level correctness checks
- public scores are recorded only as post-generation checkpoint metadata
- public scores are not used for training, inference, candidate selection, or row-level logic
- no features other than `Title` are added
- no hyperparameter tuning, model parameter tuning, threshold tuning, multiple-weight search, PassengerId overrides, manual correction rules, or target-derived features

## Input Evidence

- `raw_tabular / GradientBoostingClassifier` is current clean public baseline leader from step 05.
- unrestricted `raw_plus_title / GradientBoostingClassifier` was rejected for public transfer in step 07.
- `08_title_gating_check` selected `w=0.10` as conservative train-side candidate.

## Candidate Definition

| candidate_id | base_model | title_model | blend_weight | output_file |
| --- | --- | --- | --- | --- |
| title_gated_w010__GradientBoostingClassifier | raw_tabular / GradientBoostingClassifier | raw_plus_title / GradientBoostingClassifier | 0.10 | submissions/submission_09_title_gated_w010_gradient_boosting.csv |

- `w=0.10` is used because it was the smaller stable conservative weight among the best conservative rows in step 08.
- `w=0.15` is not used because the same OOF/flip result is available with a smaller weight.
- `w=0.25` and `w=0.30` are not used because they are already in the caution zone.
- `w=1.00` is not used because full-strength Title was already rejected.

## Training / Inference Protocol

1. Load `train.csv` and `test.csv`.
2. Create `Title` for train and test using `add_clean_features()`.
3. Copy only `Title` into train/test working frames.
4. Build the base pipeline with `RAW_TABULAR` features.
5. Build the title pipeline with `RAW_TABULAR + ["Title"]` features.
6. Fit both pipelines on full `train.csv`.
7. For `test.csv`, get class-1 probabilities aligned to `Survived == 1`.
8. Blend probabilities as `p_blend = 0.90 * p_base + 0.10 * p_title`.
9. Predict `Survived = 1` when `p_blend >= 0.5`, otherwise `0`.
10. Write submission CSV with exactly `PassengerId` and `Survived`.

## Model Panel Used

| role | feature_set | model | package | package_version | preprocessing_mode | explicit_technical_params | actual_resolved_params | parameter_adjustments | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base model | raw_tabular | GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |
| title model | raw_plus_title | GradientBoostingClassifier | scikit-learn | 1.8.0 | unscaled_tree | {"random_state": 42} | {"ccp_alpha": 0.0, "criterion": "friedman_mse", "init": null, "learning_rate": 0.1, "loss": "log_loss", "max_depth": 3, "max_features": null, "max_leaf_nodes": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 100, "n_iter_no_change": null, "random_state": 42, "subsample": 1.0, "tol": 0.0001, "validation_fraction": 0.1, "verbose": 0, "warm_start": false} |  |  |

## Generated Submission

- overall status: `PASS`

| candidate_id | output_file | rows | pred_0_count | pred_1_count | pred_1_rate | status |
| --- | --- | --- | --- | --- | --- | --- |
| title_gated_w010__GradientBoostingClassifier | submissions/submission_09_title_gated_w010_gradient_boosting.csv | 418 | 275 | 143 | 0.342105 | PASS |

## Baseline comparison

| base_candidate_id | gated_candidate_id | changed_predictions | changed_pct | base_pred_1_count | gated_pred_1_count | base_pred_1_rate | gated_pred_1_rate | delta_pred_1_rate | flip_0_to_1 | flip_1_to_0 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_tabular__GradientBoostingClassifier | title_gated_w010__GradientBoostingClassifier | 6 | 1.435407 | 141 | 143 | 0.337321 | 0.342105 | 0.004785 | 4 | 2 | PASS |

## Sanity Checks

| check | status | detail |
| --- | --- | --- |
| exactly 1 new file generated | PASS | submissions/submission_09_title_gated_w010_gradient_boosting.csv |
| file has 418 rows | PASS | rows=418 |
| columns exactly PassengerId,Survived | PASS | PassengerId,Survived |
| PassengerId order matches data/test.csv | PASS | order checked |
| Survived values are only 0/1 | PASS | values=[0, 1] |
| no duplicate PassengerId | PASS | duplicates checked |
| existing baseline submission file was not modified | PASS | submissions/submission_05_baseline_raw_tabular_gradient_boosting.csv |

## Public score checkpoint table

| output_file | public_score | note |
| --- | --- | --- |
| submissions/submission_09_title_gated_w010_gradient_boosting.csv | 0.78229 | Recorded after file generation; external checkpoint evidence only. |

## Public checkpoint result summary

- baseline public score: `0.79665`
- gated Title public score: `0.78229`
- public delta vs baseline: `-0.01436`
- gated candidate changed 6 predictions vs raw GB baseline:
  - 4 flips `0 -> 1`
  - 2 flips `1 -> 0`
- conservative `w=0.10` improved OOF in step 08, but did not transfer to public
- public score is checkpoint evidence only, not tuning feedback

## Public checkpoint status

| candidate_id | public_score | baseline_public_score | public_delta | status | note |
| --- | --- | --- | --- | --- | --- |
| title_gated_w010__GradientBoostingClassifier | 0.78229 | 0.79665 | -0.01436 | REJECTED_PUBLIC_TRANSFER | conservative Title blend did not beat raw GB baseline despite small flip count |

## Additional manual diagnostic observations

| variant | public_score | note |
| --- | --- | --- |
| no-kill directional variant | 0.78708 | ad-hoc diagnostic; did not beat raw GB baseline |
| kill-only directional variant | 0.79186 | ad-hoc diagnostic; did not beat raw GB baseline |

- These variants are not promoted.
- They are recorded only to explain that directional post-processing also did not beat the raw GB baseline.
- Do not create new submission files in this step.
- Do not use these manual diagnostics to tune row-level rules.

## Title lane conclusion

- Broad `Title` signal is not globally declared useless.
- In the current clean active GB lane:
  - unrestricted `raw_plus_title` failed public transfer in step 07;
  - conservative gated `w=0.10` also failed public transfer in step 09.
- Therefore broad/direct `Title` usage is closed for the current GB lane.
- `Title` may still be retested later only as a narrow, constrained signal inside the Age/AgeBand/Child block, for example `Master` as a child-male proxy.
- Do not continue broad Title tuning in the current lane.

## Short interpretation

- This is a frozen conservative Title checkpoint.
- Public score was recorded after file generation.
- No tuning, threshold change, row-level correction, model parameter change, feature change, or PassengerId correction was made after public results.
- `title_gated_w010 / GradientBoostingClassifier` is rejected for public transfer.
- Current clean public baseline leader remains `raw_tabular / GradientBoostingClassifier` with `0.79665`.
- Broad Title lane is closed for now.
- Narrow Title-derived child signal is deferred to a later Age/AgeBand/Child check.
- This does not use old repo history and does not compare against old repo results.
