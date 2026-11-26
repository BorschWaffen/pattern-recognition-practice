# Розпізнавання образів і машинне навчання

Репозиторій містить практичні роботи та мініпроєкт із дисципліни «Розпізнавання образів і машинне навчання»[cite: 70, 94].

## Зміст репозиторію

- **Практична робота 1–2** (`notebooks/`): налаштування середовища Google Colab, робота з набором даних Iris, очищення та підготовка даних (`data/processed/iris_cleaned.csv`)[cite: 70, 88].
- **Практична робота 3** (`notebooks/`): перша модель класифікації (`DummyClassifier` та `DecisionTreeClassifier` з `max_depth=3`), розрахунок метрик, аналіз матриці помилок і збереження моделі (`models/iris_decision_tree.joblib`, `reports/model_metrics.csv`)[cite: 70, 72, 84, 85, 88].
- **Практичні роботи 4–7 — Мініпроєкт «Порівняння класичних моделей класифікації»** (`classification-mini-project/`): бінарна класифікація на наборі *Heart Disease Dataset* (позитивний клас — `1`) із єдиним `train/test` розбиттям, `ColumnTransformer` + `Pipeline`, 5-кратною стратифікованою крос-валідацією, вибором фінальної моделі до відкриття тестової вибірки та одноразовим тестуванням[cite: 94, 95, 96, 108].

## Мініпроєкт: порівняння класичних моделей класифікації

- **Дані:** *Heart Disease Dataset* (UCI Cleveland) — 303 об'єкти, 13 вхідних ознак (5 числових і 8 категоріальних), бінарна цільова змінна `target`[cite: 95, 100].
- **Методика:** фіксований поділ `80/20` (`stratify=y`, `random_state=42`), усі перетворення всередині `Pipeline`, підбір параметрів і вибір моделі лише на `train` через `StratifiedKFold(n_splits=5)`[cite: 97, 98, 108].
- **Досліджені моделі:** `LogisticRegression` (L1/L2, `C`), `KNeighborsClassifier` (масштабування, `k`, метрика відстані), `GaussianNB`, `LDA`, `QDA` (аналіз стабільності та `reg_param`), `SVC` (`linear` та `rbf` через `GridSearchCV`)[cite: 102, 103, 106, 108].

### Результати крос-валідації (навчальна вибірка, основна метрика — `CV F1`)

| Модель | Найкращі параметри | CV F1 | CV F1 std | CV ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| Logistic Regression | `penalty='l2', C=10, solver='liblinear'` | 0.8363 | 0.0127 | 0.9002 |
| SVM | `kernel='rbf', C=10, gamma=0.01` | 0.8270 | 0.0330 | 0.8968 |
| k-NN | `n_neighbors=3, p=1 (manhattan)` | 0.8175 | 0.0304 | 0.8768 |
| LDA | `default` | 0.8128 | 0.0257 | 0.8993 |
| QDA | `reg_param=0.1` | 0.7962 | 0.0271 | 0.8680 |
| GaussianNB | `default` | 0.7495 | 0.0215 | 0.8773 |

### Фінальна модель та одноразова оцінка на `test`

До відкриття тестової вибірки фінальною зафіксовано **`LogisticRegression(penalty='l2', C=10, solver='liblinear')`** у складі повного `Pipeline` (найвищий `CV F1 = 0.8363` та найменше стандартне відхилення `0.0127` між фолдами)[cite: 109]:

- **Test Accuracy:** `0.8689`
- **Test Precision:** `0.8333`
- **Test Recall:** `0.8929`
- **Test F1:** `0.8621`
- **Test ROC AUC:** `0.9621`

Збережені артефакти мініпроєкту знаходяться в `classification-mini-project/` (`notebooks/PR04_Trapeznykov_M.ipynb`, `models/best_classifier.joblib`, `reports/cv_model_comparison.csv`, `reports/final_test_metrics.csv`)[cite: 96, 108, 111].

*Примітка: результати мають навчальний характер і не є медичним діагнозом[cite: 95].*
