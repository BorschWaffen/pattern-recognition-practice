# Порівняння класичних моделей класифікації

## Постановка задачі
Бінарна класифікація наявності ознак серцевого захворювання у пацієнтів на основі 13 клінічних ознак (позитивний клас — `1`). Основна метрика вибору моделі — `CV F1`.

## Набір даних
Heart Disease Dataset (UCI Cleveland): 303 об'єкти, 5 числових та 8 категоріальних ознак, бінарна цільова змінна `target` (`0` — 164, `1` — 139).

## Методика експерименту
- фіксований train/test split (`80/20`, `stratify=y`, `random_state=42`);
- підбір гіперпараметрів і вибір моделі лише на `train`;
- 5-кратна `StratifiedKFold`;
- одноразове фінальне оцінювання на `test`.

## Підготовка даних
- `SimpleImputer` (медіана для числових, мода для категоріальних);
- `StandardScaler` (для числових ознак);
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`;
- `ColumnTransformer` та `Pipeline`.

## Досліджені моделі
- Logistic Regression;
- k-NN;
- Gaussian Naive Bayes;
- LDA;
- QDA;
- SVM.

## Порівняння за крос-валідацією
| Модель | CV F1 | CV ROC AUC |
| --- | ---: | ---: |
| Logistic Regression (`L2, C=10`) | 0.8363 | 0.9002 |
| SVM (`RBF, C=10, gamma=0.01`) | 0.8270 | 0.8968 |
| k-NN (`k=3, manhattan`) | 0.8175 | 0.8768 |
| LDA | 0.8128 | 0.8993 |
| QDA (`reg_param=0.1`) | 0.7962 | 0.8680 |
| Gaussian Naive Bayes | 0.7495 | 0.8773 |

## Фінальна модель і test-результат
Фінальною обрано `LogisticRegression(penalty='l2', C=10, solver='liblinear')` за найвищим `CV F1` (`0.8363`) та найменшим розкидом між фолдами (`std = 0.0127`).  
Результати на відкладеному `test`: **`Test Accuracy = 0.8689`**, **`Test F1 = 0.8621`**, **`Test ROC AUC = 0.9621`**.

## Запуск
Відкрити `notebooks/PR04_Trapeznykov_M.ipynb` у Google Colab і послідовно виконати всі комірки.
