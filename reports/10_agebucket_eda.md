# 10 AgeBucket EDA

## Scope

EDA-only consolidation of Age-related checks in `notebooks/02_theory_checks.ipynb`. No preprocessing, feature, modeling, or submission files were changed.

## Notebook Changes

- Added a compact `Title Signal Context` block after rule sanity checks.
- Collected Age-related EDA immediately after the Title block under `Age / AgeMissing / AgeBucket theory checks`.
- Replaced the earlier scattered `AgeMissing Signal`, `Highest Age Missing Groups`, and `Children / Women / Class Signal` sections with one ordered Age section.
- Kept detailed Age logic out of `01_main_eda.ipynb`; it remains a basic dataset overview.

## Age Missingness

Age present vs missing:

| AgeStatus   |   count |   survived_count |   survival_rate |
|:------------|--------:|-----------------:|----------------:|
| Age missing |     177 |               52 |          0.2938 |
| Age present |     714 |              290 |          0.4062 |

Missing `Age` is structured by `Sex`, `Pclass`, and `Title`, so the next candidate should not simply be `AgeMissing` on top of median-imputed raw `Age`. The cleaner next test is an AgeBucket branch where raw `Age` is removed and age information is passed through buckets.

## Child Threshold Check

Male-child threshold rows:

|   threshold | group      |   count |   survival_rate |
|------------:|:-----------|--------:|----------------:|
|          12 | male_child |      36 |          0.5556 |
|          14 | male_child |      37 |          0.5676 |
|          16 | male_child |      40 |          0.525  |
|          18 | male_child |      58 |          0.3966 |

`14` looks like the cleaner child-threshold candidate in the displayed male-child rows; `16` is weaker and `18` materially dilutes the male-child signal.

## Old Threshold Candidates

| Sex    |   threshold |   old_count |   old_survival_rate |   non_old_survival_rate |    delta |
|:-------|------------:|------------:|--------------------:|------------------------:|---------:|
| female |          45 |          36 |              0.8056 |                  0.7467 |   0.0589 |
| female |          50 |          22 |              0.9091 |                  0.7406 |   0.1685 |
| female |          55 |          10 |              0.9    |                  0.749  |   0.151  |
| female |          60 |           4 |              1      |                  0.751  |   0.249  |
| female |          65 |           0 |            nan      |                  0.7548 | nan      |
| male   |          45 |          79 |              0.1772 |                  0.2112 |  -0.034  |
| male   |          50 |          52 |              0.1346 |                  0.2145 |  -0.0798 |
| male   |          55 |          32 |              0.125  |                  0.2114 |  -0.0864 |
| male   |          60 |          22 |              0.1364 |                  0.2088 |  -0.0725 |
| male   |          65 |          11 |              0.0909 |                  0.2081 |  -0.1172 |

The old-threshold rows have small old-count segments, especially at higher thresholds, and do not show a stable enough breakpoint to promote a general `Old` threshold from EDA alone. Any later old threshold should remain sex-specific and controlled.

## Master Fallback

|   Master Age present count |   Master Age present survival_rate |   Master Age missing count |   Master Age missing survival_rate |   Master known age min |   Master known age median |   Master known age max |
|---------------------------:|-----------------------------------:|---------------------------:|-----------------------------------:|-----------------------:|--------------------------:|-----------------------:|
|                         36 |                             0.5833 |                          4 |                                0.5 |                   0.42 |                       3.5 |                     12 |

`Master` has 4 missing-age rows; known `Master` ages run from 0.42 to 12.00 with median 3.50. This supports only a narrow missing-age child-male fallback candidate, not broad `Title` restoration.

## Proposed Next Controlled Feature Check

Next step after EDA is controlled feature check only, not feature acceptance:

- `raw_tabular`
- `raw_no_age`
- `raw_no_age_plus_agebucket`
- optionally later `raw_no_age_plus_agebucket_master_missing_only`

## Boundary

- EDA-only report.
- No model CV was run.
- No submission was created.
- `gender_submission.csv` was not used as truth.
- Broad `Title` is not restored.
- No `Mrs` / `Miss` buckets or surname/family inference are added in this step.
