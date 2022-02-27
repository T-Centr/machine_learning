"""
Постановка задачи
Загрузим данные и разделим выборку на обучающую/проверочную в соотношении 80/20.

Применим метод ближайших соседей (kNN) для классификации скоринга.
Будем использовать только биометрические данные.

Проверим качество предсказания через каппа-метрику и матрицу неточностей.

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


"""Разделение данных"""

data_train, data_test = train_test_split(data, test_size=0.2)
print(data_train.head())


"""Расчет модели kNN (k ближайших соседей)"""

# Вычисляем не центры (кластеры) исходных групп, а расстояние до всех значений.
# Выбираем то значение, которое превалирует у k ближайших соседей.
#
# Для оценки качества модели возьмем k равным 10, 100, 1000, 10000.

columns = ["Wt", "Ht", "Ins_Age", "BMI"]
max_nn = data_train.groupby("Response").count()["Id"].min()
knn10 = KNeighborsClassifier(n_neighbors=10)
knn100 = KNeighborsClassifier(n_neighbors=100)
knn1000 = KNeighborsClassifier(n_neighbors=1000)
knn10000 = KNeighborsClassifier(n_neighbors=10000)
knnmax = KNeighborsClassifier(n_neighbors=max_nn)

y = data_train["Response"]
x = pd.DataFrame(data_train, columns=columns)
knn10.fit(x, y)
knn100.fit(x, y)
knn1000.fit(x, y)
knn10000.fit(x, y)
knnmax.fit(x, y)


"""Предсказание данных"""

# Внимание: 10000 соседей потребует порядка 4 Гб оперативной памяти

data_test = pd.DataFrame(data_test)
x_test = pd.DataFrame(data_test, columns=columns)
data_test["target_10"] = knn10.predict(x_test)
data_test["target_100"] = knn100.predict(x_test)
data_test["target_1000"] = knn1000.predict(x_test)
data_test["target_10000"] = knn10000.predict(x_test)
data_test["target_max"] = knnmax.predict(x_test)
print(data_test.head(20))


"""Оценка модели"""

print(
    "kNN, 10:",
    cohen_kappa_score(
        data_test["target_10"],
        data_test["Response"],
        weights="quadratic"
    )
)
print(
    "kNN, 100:",
    cohen_kappa_score(
        data_test["target_100"],
        data_test["Response"],
        weights="quadratic"
    )
)
print(
    "kNN, 1000:",
    cohen_kappa_score(
        data_test["target_1000"],
        data_test["Response"],
        weights="quadratic"
    )
)
print(
    "kNN, 10000:",
    cohen_kappa_score(
        data_test["target_10000"],
        data_test["Response"],
        weights="quadratic"
    )
)
print(
    "kNN, max:",
    cohen_kappa_score(
        data_test["target_max"],
        data_test["Response"],
        weights="quadratic"
    )
)


"""Матрица неточностей"""

print(confusion_matrix(data_test["target_10"], data_test["Response"]))
print(confusion_matrix(data_test["target_10000"], data_test["Response"]))
