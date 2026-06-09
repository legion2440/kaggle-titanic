# 17A Family/Surname Structural EDA

## Purpose / hypothesis

This is an EDA-only check for whether `FamilySize` and full-manifest `SurnameCount` are mostly duplicate structural signals or whether their divergence is useful enough to justify one combined structural band.

Current public leader context remains `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv` with public score `0.79904`. No model check, submission, PassengerId rule, public-score tuning, Ticket/Fare expansion, or survival encoding is introduced here.

Pre-check result: the requested stale Step 17 CabinKnown weight/blend search returned no matches before this EDA was added, so removed blend artifacts were not still presented as active candidates/leaders.

## Allowed boundary

Allowed in this step:

- `FamilySize = SibSp + Parch + 1`
- `FamilyBand`
- `Surname`, parsed as text before the comma in `Name`
- `SurnameCount`, counted over the combined train/test passenger manifest without `Survived`
- `SurnameCountBand`
- `FamilySizeEqualsSurnameCount`
- `FamilySizeDiffersSurnameCount`
- combined `FamilySurnameBand`

Band convention reuses the repo `bucket_family_size` cut points for both family and surname size: `alone=1`, `small=2-4`, `medium=5-6`, `large=7+`.

Combined bucket definition:

- matching rows: `match_<FamilyBand>`
- mismatching rows: `mismatch_family_<FamilyBand>_surname_<SurnameCountBand>`

## Feature sanity and distributions

Rows: train `891`, test `418`, combined manifest `1309`.

Surname extraction sanity examples:

| dataset | PassengerId | Name | Surname | FamilySize | SurnameCount |
| --- | --- | --- | --- | --- | --- |
| train | 1 | Braund, Mr. Owen Harris | Braund | 2 | 2 |
| train | 2 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | Cumings | 2 | 2 |
| train | 3 | Heikkinen, Miss. Laina | Heikkinen | 1 | 1 |
| train | 4 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | Futrelle | 2 | 2 |
| train | 5 | Allen, Mr. William Henry | Allen | 1 | 2 |
| train | 6 | Moran, Mr. James | Moran | 1 | 3 |
| train | 7 | McCarthy, Mr. Timothy J | McCarthy | 1 | 2 |
| train | 8 | Palsson, Master. Gosta Leonard | Palsson | 5 | 5 |
| train | 9 | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) | Johnson | 3 | 6 |
| train | 10 | Nasser, Mrs. Nicholas (Adele Achem) | Nasser | 2 | 2 |
| train | 11 | Sandstrom, Miss. Marguerite Rut | Sandstrom | 3 | 3 |
| train | 12 | Bonnell, Miss. Elizabeth | Bonnell | 1 | 2 |

Numeric train/test distributions:

| value | FamilySize train | FamilySize test | SurnameCount train | SurnameCount test |
| --- | --- | --- | --- | --- |
| 1 | 537 (0.6027) | 253 (0.6053) | 447 (0.5017) | 190 (0.4545) |
| 2 | 161 (0.1807) | 74 (0.1770) | 168 (0.1886) | 98 (0.2344) |
| 3 | 102 (0.1145) | 57 (0.1364) | 121 (0.1358) | 68 (0.1627) |
| 4 | 29 (0.0325) | 14 (0.0335) | 62 (0.0696) | 26 (0.0622) |
| 5 | 15 (0.0168) | 7 (0.0167) | 19 (0.0213) | 11 (0.0263) |
| 6 | 22 (0.0247) | 3 (0.0072) | 45 (0.0505) | 9 (0.0215) |
| 7 | 12 (0.0135) | 4 (0.0096) | 3 (0.0034) | 4 (0.0096) |
| 8 | 6 (0.0067) | 2 (0.0048) | 10 (0.0112) | 6 (0.0144) |
| 11 | 7 (0.0079) | 4 (0.0096) | 16 (0.0180) | 6 (0.0144) |

Band distributions:

| band | FamilyBand train | FamilyBand test | SurnameCountBand train | SurnameCountBand test |
| --- | --- | --- | --- | --- |
| alone | 537 (0.6027) | 253 (0.6053) | 447 (0.5017) | 190 (0.4545) |
| small | 292 (0.3277) | 145 (0.3469) | 351 (0.3939) | 192 (0.4593) |
| medium | 37 (0.0415) | 10 (0.0239) | 64 (0.0718) | 20 (0.0478) |
| large | 25 (0.0281) | 10 (0.0239) | 29 (0.0325) | 16 (0.0383) |

Equality diagnostics:

| diagnostic | train count | train rate | test count | test rate |
| --- | --- | --- | --- | --- |
| `FamilySize == SurnameCount` | 665 | 0.7464 | 303 | 0.7249 |
| `FamilySize != SurnameCount` | 226 | 0.2536 | 115 | 0.2751 |

Combined `FamilySurnameBand` train/test distribution:

| FamilySurnameBand | train count | train rate | test count | test rate |
| --- | --- | --- | --- | --- |
| match_alone | 426 | 0.4781 | 184 | 0.4402 |
| match_large | 13 | 0.0146 | 6 | 0.0144 |
| match_medium | 31 | 0.0348 | 8 | 0.0191 |
| match_small | 195 | 0.2189 | 105 | 0.2512 |
| mismatch_family_alone_surname_large | 2 | 0.0022 | 3 | 0.0072 |
| mismatch_family_alone_surname_medium | 16 | 0.0180 | 4 | 0.0096 |
| mismatch_family_alone_surname_small | 93 | 0.1044 | 62 | 0.1483 |
| mismatch_family_large_surname_large | 12 | 0.0135 | 4 | 0.0096 |
| mismatch_family_medium_surname_medium | 4 | 0.0045 | 1 | 0.0024 |
| mismatch_family_medium_surname_small | 2 | 0.0022 | 1 | 0.0024 |
| mismatch_family_small_surname_alone | 21 | 0.0236 | 6 | 0.0144 |
| mismatch_family_small_surname_large | 2 | 0.0022 | 3 | 0.0072 |
| mismatch_family_small_surname_medium | 13 | 0.0146 | 7 | 0.0167 |
| mismatch_family_small_surname_small | 61 | 0.0685 | 24 | 0.0574 |

No combined bucket is train-only or test-only. Several buckets are rare in both splits, especially `mismatch_family_*_surname_large`, `mismatch_family_medium_*`, and `match_large`.

## Overlap tables

Train `FamilySize` vs `SurnameCount`:

| FamilySize | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 11 | All |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 426 | 49 | 29 | 15 | 7 | 9 | 1 | 0 | 1 | 537 |
| 2 | 16 | 107 | 25 | 9 | 2 | 2 | 0 | 0 | 0 | 161 |
| 3 | 5 | 9 | 66 | 15 | 0 | 5 | 2 | 0 | 0 | 102 |
| 4 | 0 | 3 | 0 | 22 | 0 | 4 | 0 | 0 | 0 | 29 |
| 5 | 0 | 0 | 0 | 1 | 10 | 4 | 0 | 0 | 0 | 15 |
| 6 | 0 | 0 | 1 | 0 | 0 | 21 | 0 | 0 | 0 | 22 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 8 | 12 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 6 |
| 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| All | 447 | 168 | 121 | 62 | 19 | 45 | 3 | 10 | 16 | 891 |

Test `FamilySize` vs `SurnameCount`:

| FamilySize | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 11 | All |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 184 | 38 | 17 | 7 | 3 | 1 | 1 | 1 | 1 | 253 |
| 2 | 4 | 52 | 10 | 4 | 2 | 2 | 0 | 0 | 0 | 74 |
| 3 | 2 | 7 | 40 | 2 | 1 | 2 | 3 | 0 | 0 | 57 |
| 4 | 0 | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 14 |
| 5 | 0 | 0 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 7 |
| 6 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 4 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| All | 190 | 98 | 68 | 26 | 11 | 9 | 4 | 6 | 6 | 418 |

Band-level overlap:

| split | FamilyBand | alone | small | medium | large | All |
| --- | --- | --- | --- | --- | --- | --- |
| train | alone | 426 | 93 | 16 | 2 | 537 |
| train | small | 21 | 256 | 13 | 2 | 292 |
| train | medium | 0 | 2 | 35 | 0 | 37 |
| train | large | 0 | 0 | 0 | 25 | 25 |
| train | All | 447 | 351 | 64 | 29 | 891 |
| test | alone | 184 | 62 | 4 | 3 | 253 |
| test | small | 6 | 129 | 7 | 3 | 145 |
| test | medium | 0 | 1 | 9 | 0 | 10 |
| test | large | 0 | 0 | 0 | 10 | 10 |
| test | All | 190 | 192 | 20 | 16 | 418 |

## Train-only survival analysis

Survival by simple bands:

| feature | band | count | survived | survival_rate |
| --- | --- | --- | --- | --- |
| FamilyBand | alone | 537 | 163 | 0.3035 |
| FamilyBand | small | 292 | 169 | 0.5788 |
| FamilyBand | medium | 37 | 6 | 0.1622 |
| FamilyBand | large | 25 | 4 | 0.1600 |
| SurnameCountBand | alone | 447 | 149 | 0.3333 |
| SurnameCountBand | small | 351 | 167 | 0.4758 |
| SurnameCountBand | medium | 64 | 20 | 0.3125 |
| SurnameCountBand | large | 29 | 6 | 0.2069 |

Survival by equality diagnostic:

| diagnostic | count | survived | survival_rate |
| --- | --- | --- | --- |
| `FamilySize == SurnameCount` | 665 | 248 | 0.3729 |
| `FamilySize != SurnameCount` | 226 | 94 | 0.4159 |

Survival by combined `FamilySurnameBand`:

| FamilySurnameBand | count | survived | survival_rate |
| --- | --- | --- | --- |
| match_alone | 426 | 132 | 0.3099 |
| match_large | 13 | 0 | 0.0000 |
| match_medium | 31 | 4 | 0.1290 |
| match_small | 195 | 112 | 0.5744 |
| mismatch_family_alone_surname_large | 2 | 1 | 0.5000 |
| mismatch_family_alone_surname_medium | 16 | 7 | 0.4375 |
| mismatch_family_alone_surname_small | 93 | 23 | 0.2473 |
| mismatch_family_large_surname_large | 12 | 4 | 0.3333 |
| mismatch_family_medium_surname_medium | 4 | 0 | 0.0000 |
| mismatch_family_medium_surname_small | 2 | 2 | 1.0000 |
| mismatch_family_small_surname_alone | 21 | 17 | 0.8095 |
| mismatch_family_small_surname_large | 2 | 1 | 0.5000 |
| mismatch_family_small_surname_medium | 13 | 9 | 0.6923 |
| mismatch_family_small_surname_small | 61 | 30 | 0.4918 |

Reading: the clearest large-count contrast is within `alone`: `match_alone` survival is `0.3099`, while `mismatch_family_alone_surname_small` is lower at `0.2473`. In the other direction, `mismatch_family_small_surname_alone` is high at `0.8095`, but only 21 rows. Several extreme rates are small-n and should not be promoted as rules.

## Segment checks

Match/mismatch survival by `Sex x Pclass`:

| Sex | Pclass | match_status | count | survived | survival_rate |
| --- | --- | --- | --- | --- | --- |
| female | 1 | mismatch | 31 | 31 | 1.0000 |
| female | 1 | match | 63 | 60 | 0.9524 |
| female | 2 | mismatch | 22 | 20 | 0.9091 |
| female | 2 | match | 54 | 50 | 0.9259 |
| female | 3 | mismatch | 40 | 19 | 0.4750 |
| female | 3 | match | 104 | 53 | 0.5096 |
| male | 1 | mismatch | 25 | 9 | 0.3600 |
| male | 1 | match | 97 | 36 | 0.3711 |
| male | 2 | mismatch | 18 | 4 | 0.2222 |
| male | 2 | match | 90 | 13 | 0.1444 |
| male | 3 | mismatch | 90 | 11 | 0.1222 |
| male | 3 | match | 257 | 36 | 0.1401 |

Female Pclass 3 combined buckets:

| FamilySurnameBand | count | survived | survival_rate |
| --- | --- | --- | --- |
| match_alone | 48 | 30 | 0.6250 |
| match_large | 5 | 0 | 0.0000 |
| match_medium | 11 | 0 | 0.0000 |
| match_small | 40 | 23 | 0.5750 |
| mismatch_family_alone_surname_medium | 2 | 2 | 1.0000 |
| mismatch_family_alone_surname_small | 10 | 5 | 0.5000 |
| mismatch_family_large_surname_large | 8 | 3 | 0.3750 |
| mismatch_family_medium_surname_medium | 3 | 0 | 0.0000 |
| mismatch_family_small_surname_alone | 2 | 1 | 0.5000 |
| mismatch_family_small_surname_medium | 2 | 2 | 1.0000 |
| mismatch_family_small_surname_small | 13 | 6 | 0.4615 |

Male Pclass 1 combined buckets:

| FamilySurnameBand | count | survived | survival_rate |
| --- | --- | --- | --- |
| match_alone | 64 | 22 | 0.3438 |
| match_medium | 2 | 0 | 0.0000 |
| match_small | 31 | 14 | 0.4516 |
| mismatch_family_alone_surname_medium | 2 | 0 | 0.0000 |
| mismatch_family_alone_surname_small | 9 | 3 | 0.3333 |
| mismatch_family_small_surname_alone | 1 | 0 | 0.0000 |
| mismatch_family_small_surname_medium | 3 | 2 | 0.6667 |
| mismatch_family_small_surname_small | 10 | 4 | 0.4000 |

Segment reading:

- Female Pclass 3 is structurally heterogeneous: `match_medium` and `match_large` are low-survival family-heavy groups, while `match_alone` and `match_small` remain materially higher.
- Male Pclass 1 does not show a strong broad mismatch split: match `0.3711` vs mismatch `0.3600`. The useful signal, if any, is bounded to small combined buckets and is too small for a rule.
- Sex and Pclass already dominate survival levels. Family/Surname structure should therefore be tested only as one bounded categorical structure, not as separate independent additive columns.

## Duplicate-signal assessment

Is `FamilyBand` mostly duplicating `SibSp`/`Parch` already present in `raw_tabular`?

Yes. `FamilySize` is a deterministic compression of `SibSp + Parch + 1`, and `FamilyBand` is a bucketed version of that same raw information. It may help some model classes through nonlinearity, but it is not new information.

Is `SurnameCountBand` providing genuinely different structure?

Partially. `SurnameCount` overlaps heavily with `FamilySize`, but about one quarter of both splits differ: train `226/891 = 0.2536`, test `115/418 = 0.2751`. This is enough to treat surname count as complementary structural context, not as a standalone independent feature.

Is the useful signal concentrated in `FamilySize != SurnameCount`?

Mostly yes, but not as a simple binary rule. The binary mismatch rate has only modest aggregate train survival lift (`0.4159` vs `0.3729`). The signal appears in specific combined buckets such as `mismatch_family_alone_surname_small`, `mismatch_family_small_surname_alone`, and difficult family-heavy buckets.

Is it safer to test one combined `FamilySurnameBand` rather than adding `FamilyBand` and `SurnameCountBand` as separate independent columns?

Yes. Separate columns would invite duplicate counting of family size and surname size. One combined band explicitly represents the overlap/mismatch structure and keeps the next controlled check bounded.

## Final recommendation

Step 17A EDA conclusion: FamilySize and SurnameCount overlap heavily, but the aggregate equality/difference survival contrast is weak and does not justify a separate mismatch-only diagnostic.

Current status after corrected Step 17B: **Family/Surname structural lane = CLOSED**.

The corrected Step 17B `FamilySurnameSizeBand` replacement check rejected the structural branch train-side. Do not read this EDA as an open recommendation to run Step 17C or `FamilySmallerThanSurnameFlag`.

Warnings / uncertainty:

- Several combined buckets are small-n in train and rare in test, so they should not be converted into PassengerId-style or survival-derived rules.
- This report did not run any model check, did not create a submission, and did not use public score feedback.
- `SurnameCount` was computed on the full train/test manifest structurally and without `Survived`; this is acceptable for EDA here but should be documented if later used in a feature pipeline.
