# Kaggle Titanic

## О проекте

Репозиторий фиксирует воспроизводимую исследовательскую линию по задаче Kaggle Titanic: от EDA и границы метода до feature checks, frozen checkpoints, проверок переноса на public и текущего лидера.

Основной принцип проекта: сначала train-side evidence, затем отдельная проверка public transfer. `test.csv` используется только для inference и проверки формы submission, без test labels и без row-level correctness.

## Текущий статус

Текущий frozen public leader:

- submission: `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv`
- public score: `0.79904`
- model base: `raw_tabular / GradientBoostingClassifier`
- rule layer: ограниченный CabinKnown subgroup downshift для male, Pclass 1, CabinKnown 0, где `raw_tabular_pred == 1` и `raw_plus_cabinknown_pred == 0`
- changed test rows vs raw_tabular GB full-fit: 3
- status label: `CURRENT_PUBLIC_LEADER`

Последняя проверенная submission:

- submission: `submissions/submission_19b_tuned_gb_raw_tabular.csv`
- public score: `0.79186`
- status: `REJECT_PUBLIC_TRANSFER`
- interpretation: сильный train-side OOF-сигнал Step 19A не перенесся на public для standalone tuned raw_tabular GB candidate

## Граница метода

Источник истины по границе метода: `reports/method_boundary.md`.

Коротко:

- `train.csv` можно использовать для EDA, CV, OOF и full-train fitting для frozen submissions.
- `test.csv` используется только для inference и проверки формы submission.
- `gender_submission.csv` - baseline prediction file, а не truth.
- `PassengerId` допускается только как submission identifier и validation key для проверки row order.
- test labels, row-level public correctness checks, PassengerId rules и public-score tuning запрещены.
- target-derived group survival rates не являются deployable final features; они допустимы только в quarantined diagnostics с явной anti-leakage boundary.

## Структура репозитория

```text
kaggle-titanic/
├── data/
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── notebooks/
│   ├── 01_main_eda.ipynb
│   └── 02_theory_checks.ipynb
├── reports/
│   ├── 03_preprocessing_layer.md
│   ├── 04_baseline.md
│   ├── 05_baseline_frozen_checkpoint.md
│   ├── ...
│   └── 19b_tuned_gb_raw_tabular_submission_review.md
├── scripts/
│   ├── config.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── 04_baseline.py
│   ├── ...
│   └── 19b_create_tuned_gb_raw_tabular_submission.py
└── submissions/
    ├── submission_05_baseline_raw_tabular_gradient_boosting.csv
    ├── ...
    ├── submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv
    ├── ...
    └── submission_19b_tuned_gb_raw_tabular.csv
```

## Слои признаков

### F00_CORE

Минимальный first-order baseline:

- `Sex`
- `Pclass`
- `Embarked`

### RAW_TABULAR

Основной raw baseline:

- `Sex`
- `Pclass`
- `Embarked`
- `Age`
- `SibSp`
- `Parch`
- `Fare`

### Контролируемые feature checks

Проверенные ветки признаков:

- `Title`
- gated `Title`
- `AgeBucket`
- `AgeMissing`
- `FareLog`
- `CabinKnown`
- bounded `CabinKnown subgroup` gate
- `FamilySurnameSizeBand`
- fold-safe `SurnameSurvival` diagnostics
- tuned `raw_tabular / GradientBoostingClassifier`

## Основные результаты

```text
Step  Artifact                                                   Status                  Key result
----  ---------------------------------------------------------  ----------------------  ------------------------------------------------------------
03    reports/03_preprocessing_layer.md                         PASS                    preprocessing layer создан и проверен
04    reports/04_baseline.md                                    PASS                    baseline model panel завершен
05    reports/05_baseline_frozen_checkpoint.md                  BASELINE_LEADER         raw_tabular / GradientBoostingClassifier, public 0.79665
07    reports/07_title_frozen_checkpoint.md                     REJECT_PUBLIC_TRANSFER  unrestricted Title не перенесся на public
09    reports/09_title_gated_frozen_checkpoint.md               REJECT_PUBLIC_TRANSFER  conservative gated Title не побил baseline
12    reports/12_agebucket_frozen_checkpoint.md                 REJECT_PUBLIC_TRANSFER  AgeBucket branch не стал лидером
14    reports/14_agemissing_frozen_checkpoint.md                REJECT_PUBLIC_TRANSFER  AgeMissing handling не стал лидером
15    reports/15_farelog_controlled_check.md                    REJECTED_TRAIN_SIDE     FareLog branch отклонен train-side
16    reports/16_cabinknown_controlled_check.md                 MIXED                   full CabinKnown дал OOF-прирост, но не прошел public transfer
16B   reports/16b_cabinknown_subgroup_gate_check.md             CURRENT_PUBLIC_LEADER   bounded CabinKnown subgroup gate достиг public 0.79904
17    reports/17_family_surname_size_band_check.md              REJECTED_TRAIN_SIDE     Family/Surname structural size band отклонен train-side
18    reports/18_surname_survival_foldsafe_check.md             REJECT_TRAIN_SIDE       fold-safe SurnameSurvival feature отклонен train-side
18B   reports/18b_surname_survival_gated_overlay_check.md       OOF_NEGATIVE            broad overlay diagnostic negative, submission не создавалась
18C   reports/18c_surname_survival_directional_overlay_check.md DIAGNOSTIC_ONLY         strict downshift дал tiny OOF-positive signal, без public claim
19A   reports/19a_gb_raw_tabular_tuning_stage1_stage2.md        OOF_POSITIVE            best tuned GB OOF 0.843996, delta +0.016835
19B   reports/19b_tuned_gb_raw_tabular_submission_review.md     REJECT_PUBLIC_TRANSFER  tuned GB public 0.79186; текущий лидер остается Step 16B
```

## Текущий вывод

Самый сильный текущий public artifact - Step 16B, а не tuned Step 19B model.

Step 19A показал сильное OOF-улучшение для tuned `raw_tabular / GradientBoostingClassifier`, но Step 19B отклонил этот candidate по public transfer. Это аргумент против конкретной standalone tuned raw_tabular GB submission, а не доказательство, что tuning бесполезен в целом.

Активный frozen public leader остается:

```text
submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv
public score: 0.79904
```

## Установка и запуск

Установить зависимости и создать окружение:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Запускать scripts из корня репозитория:

```bash
python scripts/04_baseline.py
python scripts/05_baseline_frozen_checkpoint.py
python scripts/06_title_feature_check.py
python scripts/07_title_frozen_checkpoint.py
python scripts/08_title_gating_check.py
python scripts/09_title_gated_frozen_checkpoint.py
python scripts/11_agebucket_feature_check.py
python scripts/12_agebucket_frozen_checkpoint.py
python scripts/13_agemissing_handling_check.py
python scripts/14_agemissing_frozen_checkpoint.py
python scripts/15_farelog_controlled_check.py
python scripts/16_cabinknown_controlled_check.py
python scripts/16b_cabinknown_subgroup_gate_check.py
python scripts/17_family_surname_size_band_check.py
python scripts/18_surname_survival_foldsafe_check.py
python scripts/18b_surname_survival_gated_overlay_check.py
python scripts/18c_surname_survival_directional_overlay_check.py
python scripts/19a_gb_raw_tabular_tuning_stage1_stage2.py
python scripts/19b_create_tuned_gb_raw_tabular_submission.py
```

Примечания:
- Scripts ожидают стандартную структуру репозитория с `data/`, `reports/`, `scripts/` и `submissions/` в корне.
- Сгенерированные reports и submissions закоммичены как experiment artifacts.
- Часть scripts зависит от `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `pandas` и `numpy`.

## Важные артефакты

### Method и EDA

- `reports/method_boundary.md` - граница метода и anti-leakage rules.
- `notebooks/01_main_eda.ipynb` - основной executable EDA.
- `notebooks/02_theory_checks.ipynb` - quarantined theory/rule checks.

### Основная реализация

- `scripts/config.py` - repository paths.
- `scripts/features.py` - feature definitions и feature construction helpers.
- `scripts/preprocessing.py` - preprocessing pipelines для linear/scaled и tree/unscaled model lanes.

### Текущий лидер и последняя проверка

- `reports/16b_cabinknown_subgroup_gate_check.md` - отчет по текущему public leader.
- `submissions/submission_16b_gb_cabinknown_male_p1_cabin_unknown_downshift.csv` - submission текущего public leader.
- `reports/19b_tuned_gb_raw_tabular_submission_review.md` - последний rejected public-transfer check для tuned GB.
- `submissions/submission_19b_tuned_gb_raw_tabular.csv` - последняя проверенная submission.


## TOC

- [О проекте](#о-проекте)
- [Текущий статус](#текущий-статус)
- [Граница метода](#граница-метода)
- [Структура репозитория](#структура-репозитория)
- [Слои признаков](#слои-признаков)
  - [F00_CORE](#f00_core)
  - [RAW_TABULAR](#raw_tabular)
  - [Контролируемые feature checks](#контролируемые-feature-checks)
- [Основные результаты](#основные-результаты)
- [Текущий вывод](#текущий-вывод)
- [Установка и запуск](#установка-и-запуск)
- [Важные артефакты](#важные-артефакты)
  - [Method и EDA](#method-и-eda)
  - [Основная реализация](#основная-реализация)
  - [Текущий лидер и последняя проверка](#текущий-лидер-и-последняя-проверка)

## Автор

- Nazar Yestayev (@nyestaye / @legion2440)