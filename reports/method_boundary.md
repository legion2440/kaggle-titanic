# Method Boundary

This file defines the method boundary for the clean research line. It separates
allowed EDA / validation usage from leakage-prone shortcuts.

## Source Roles

- `train.csv` can use `Survived` only for train-side EDA, CV, OOF, and holdout.
- `test.csv` has no `Survived` and must not be used for row-level correctness.
- `gender_submission.csv` is a baseline prediction file, not truth.

## Forbidden

- Test labels, inferred test labels, or external row-level truth for `test.csv`.
- `gender_submission.csv` as truth.
- `PassengerId` as a predictive feature, lookup key, or manual correction rule for `test.csv`.
- Target-derived group survival rates are forbidden as final/deployable features, lookup rules, or test-side scoring logic. They may be used only in quarantined theory checks to test whether such signals explain a pattern. Such checks must be marked as outside the clean selected solution.
- Local test accuracy claims from `test.csv`.

## Allowed

- Raw clean features from `train.csv` and/or `test.csv`.
- Clean derived features computed from raw passenger fields only, without using `Survived` as an input feature, lookup target, or test-side scoring signal.
- Structural counts without `Survived`, including group sizes, missingness flags, parsed titles, family size, ticket prefix counts, and cabin availability.
- Train-only EDA that uses `Survived` to form hypotheses.
- Train-only CV / OOF / holdout validation.
- Kaggle public/private leaderboard scores as external submission feedback, not local row-level truth.
- Quarantined target-derived theory checks are allowed only for analysis/rejection, not for final pipeline, selected solution, or deployable test prediction logic.

## Practical Rule

If a value or rule requires knowing whether a specific `test.csv` row survived,
it is outside the method boundary. If it can be computed from raw row fields and
train-only validation protocol, it remains eligible for validation.
