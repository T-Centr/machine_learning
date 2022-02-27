"""
Постановка задачи¶
Загрузите данные и разделите выборку на обучающую/проверочную в
соотношении 80/20.

Примените метод ближайших соседей (kNN) для классификации скоринга, используйте
k=100. Используйте биометрические данные, все столбцы Insurance_History,
Family_Hist, Medical_History и InsurеdInfo. Заполните отсутствующие значения -1.

Проведите предсказание и проверьте качество через каппа-метрику.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов

pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов

pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных"""

data = pd.read_csv(
    "https://video.ittensive.com/machine-learning/prudential/train.csv.gz"
)
print(data.info())


"""Заполнение отсутствующих значений"""

data.fillna(value=-1, inplace=True)


"""Разделение данных"""

# Преобразуем выборки в отдельные наборы данных

data_train, data_test = train_test_split(data, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
print(data_train.head())


"""Расчет модели kNN (100 ближайших соседей)"""

# Соберем все нужные столбцы данных

columns_groups = [
    "Insurance_History",
    "Family_Hist",
    "Medical_History",
    "InsurеdInfo"
]
columns = ["Wt", "Ht", "Ins_Age", "BMI"]
for cg in columns_groups:
    columns.extend(data_train.columns[data_train.columns.str.startswith(cg)])
print(columns)

knn = KNeighborsClassifier(n_neighbors=100)
y = data_train["Response"]
x = pd.DataFrame(data_train, columns=columns)
knn.fit(x, y)


"""Предсказание данных"""

x_test = pd.DataFrame(data_test, columns=columns)
data_test["target"] = knn.predict(x_test)
print(data_test.head(20))


"""Оценка модели"""

print(
    "kNN, 100:",
    round(cohen_kappa_score(
        data_test["target"],
        data_test["Response"],
        weights="quadratic"
        ), 3)
)


"""Матрица неточностей"""

print(confusion_matrix(data_test["target"], data_test["Response"]))
