# CHECK002 Preprocessing Layer

This report records the first clean preprocessing layer implementation.
It does not run model training, benchmarking, CV, OOF, holdout, F01, BASE_1, or submission generation.

## Overall Status

- overall: `PASS`
- train shape raw: `(891, 12)`
- test shape raw: `(418, 11)`
- train shape after `add_clean_features`: `(891, 20)`
- test shape after `add_clean_features`: `(418, 19)`

## Preprocessing Modes

- `scaled_linear`: numeric median imputation + scaling; categorical most_frequent imputation + one-hot encoding
- `unscaled_tree`: numeric median imputation without scaling; categorical most_frequent imputation + one-hot encoding

## Current Feature Typing

Numeric features:

- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `AgeMissing`
- `IsChild12`
- `FamilySize`
- `FareLog`
- `CabinKnown`

Categorical features:

- `Sex`
- `Pclass`
- `Embarked`
- `Title`
- `AgeBin`
- `FamilySizeBucket`

## Agreed Feature Sets For Next Step

| feature_set | features | numeric | categorical | presence_train | presence_test | deferred_absent |
| --- | --- | --- | --- | --- | --- | --- |
| f00_core | Sex, Pclass, Embarked | none | Sex, Pclass, Embarked | PASS | PASS | PASS |
| raw_tabular | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare | Age, SibSp, Parch, Fare | Sex, Pclass, Embarked | PASS | PASS | PASS |
| raw_plus_title | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, Title | Age, SibSp, Parch, Fare | Sex, Pclass, Embarked, Title | PASS | PASS | PASS |
| raw_plus_agemissing | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, AgeMissing | Age, SibSp, Parch, Fare, AgeMissing | Sex, Pclass, Embarked | PASS | PASS | PASS |
| raw_plus_agebin_child12 | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, AgeBin, IsChild12 | Age, SibSp, Parch, Fare, IsChild12 | Sex, Pclass, Embarked, AgeBin | PASS | PASS | PASS |
| raw_plus_family | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, FamilySize, FamilySizeBucket | Age, SibSp, Parch, Fare, FamilySize | Sex, Pclass, Embarked, FamilySizeBucket | PASS | PASS | PASS |
| raw_plus_farelog | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, FareLog | Age, SibSp, Parch, Fare, FareLog | Sex, Pclass, Embarked | PASS | PASS | PASS |
| raw_plus_cabinknown | Sex, Pclass, Embarked, Age, SibSp, Parch, Fare, CabinKnown | Age, SibSp, Parch, Fare, CabinKnown | Sex, Pclass, Embarked | PASS | PASS | PASS |

## Preprocessing Checks

| feature_set | mode | fit_transform | train_shape | test_shape | same_output_columns | finite_train | finite_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f00_core | scaled_linear | PASS | (891, 8) | (418, 8) | PASS | PASS | PASS |
| f00_core | unscaled_tree | PASS | (891, 8) | (418, 8) | PASS | PASS | PASS |
| raw_tabular | scaled_linear | PASS | (891, 12) | (418, 12) | PASS | PASS | PASS |
| raw_tabular | unscaled_tree | PASS | (891, 12) | (418, 12) | PASS | PASS | PASS |
| raw_plus_title | scaled_linear | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_title | unscaled_tree | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_agemissing | scaled_linear | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |
| raw_plus_agemissing | unscaled_tree | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |
| raw_plus_agebin_child12 | scaled_linear | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_agebin_child12 | unscaled_tree | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_family | scaled_linear | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_family | unscaled_tree | PASS | (891, 17) | (418, 17) | PASS | PASS | PASS |
| raw_plus_farelog | scaled_linear | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |
| raw_plus_farelog | unscaled_tree | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |
| raw_plus_cabinknown | scaled_linear | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |
| raw_plus_cabinknown | unscaled_tree | PASS | (891, 13) | (418, 13) | PASS | PASS | PASS |

## Scope Boundary

- no model training
- no benchmark
- no CV / OOF
- no holdout
- no F01
- no BASE_1
- no submissions
