# Kaggle Titanic

## О проекте

Репозиторий собирает чистую исследовательскую линию для задачи Kaggle Titanic.

Сейчас в репозитории зафиксированы:
- исполняемая EDA в ноутбуках;
- граница допустимого метода;
- разделение между основным обзором данных и theory checks.

Финальный training pipeline и selected solution пока не зафиксированы.

## Что есть сейчас

- `notebooks/01_main_eda.ipynb` - основной исполняемый EDA notebook.
- `notebooks/02_theory_checks.ipynb` - отдельный notebook для theory/rule checks.
- `reports/method_boundary.md` - рабочая граница допустимого метода.
- `scripts/` - зарезервирован под воспроизводимую preprocessing/modeling логику.
- `submissions/` - зарезервирован под артефакты сабмишенов.

## Граница метода

Источник истины по границе метода: `reports/method_boundary.md`.

Коротко:
- `train.csv` можно использовать для train-side EDA и последующей train-only validation.
- `test.csv` не содержит `Survived`; локальная row-level correctness для него недопустима.
- `gender_submission.csv` - baseline prediction file, а не truth.
- `PassengerId`, manual rules, test labels и target-derived group survival rates не должны использоваться как путь к финальному решению.

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
│   └── method_boundary.md
├── scripts/
└── submissions/
```

## Ноутбуки

### 01_main_eda.ipynb

Содержит основной обзор данных:
- source roles;
- column presence;
- target availability;
- missingness;
- raw distributions;
- базовые train-only target patterns;
- raw outlier audit.

### 02_theory_checks.ipynb

Содержит отдельные theory/rule checks:
- quarantined hypothesis checks;
- threshold comparisons;
- rule-style baselines;
- проверки, которые не считаются selected solution и не переходят в final pipeline автоматически.

## Что пока не зафиксировано

### F01

`F01` пока не зафиксирован.

### BASE_1

`BASE_1` пока не зафиксирован.

### Rejected Checks

Раздел будет заполнен после появления validation artifacts.

### Validation Summary

Раздел будет заполнен после появления validation artifacts.

## TOC

- [О проекте](#о-проекте)
- [Что есть сейчас](#что-есть-сейчас)
- [Граница метода](#граница-метода)
- [Структура репозитория](#структура-репозитория)
- [Ноутбуки](#ноутбуки)
- [Что пока не зафиксировано](#что-пока-не-зафиксировано)
- [F01](#f01)
- [BASE_1](#base_1)
- [Rejected Checks](#rejected-checks)
- [Validation Summary](#validation-summary)

## Автор
- Nazar Yestayev (@nyestaye / @legion2440)