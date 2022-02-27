"""
Постановка задачи
Загрузим данные и разделим выборку на обучающую/проверочную в соотношении 80/20.

Для обучающей выборки вычислим средние значения для веса, роста, индекса массы
тела и возраста - для каждого из принятых решений. Предскажем оценку скоринга по
близости данных средним значениям.

Проверим качество предсказания через F1-метрику и матрицу неточностей.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score


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


"""Разделение данных на обучающие и проверочные, 80/20"""

data_train, data_test = train_test_split(data, test_size=0.2)
print(data_train.head())


"""Вычисление средних значений для каждой оценки"""

# Проведем кластеризацию данных: разобьем их на группы по известной оценку
# скоринга (Response), вычислим центры этих групп как средние значения
# биометрических параметров.
columns = ["Wt", "Ht", "Ins_Age", "BMI"]
responses = np.arange(1, data["Response"].max() + 1)
clusters = [{}] * (len(responses) + 1)
for r in responses:
    clusters[r] = {}
    for c in columns:
        clusters[r][c] = data[data["Response"] == r][c].median()
print(clusters)



"""Выполним предсказание оценки скоринга на основе средних значений"""

# Будем использовать евклидово расстояние:
# 𝐷=sqrt(∑(𝑎𝑖−С𝑖) ** 2),
# где 𝑎𝑖 − значение параметров в проверочной
# выборке 𝐶𝑖 − значение центров кластеров по данным обучающей выборки
# Выберем принадлежность к кластеру, расстояние до которого минимально


def calc_model(x):
    D_min = 10000000
    target = 0
    for _, cluster in enumerate(clusters):
        if len(cluster) > 0:
            D = 0
            for c in columns:
                D += (cluster[c] - x[c]) ** 2
            D = np.sqrt(D)
            if D < D_min:
                target = _
                D_min = D
    x["target"] = target
    x["random"] = int(np.random.uniform(1, 8.01, 1)[0])
    x["sample"] = data.sample(1)["Response"].values[0]
    x["all8"] = 8
    return x


data_test = data_test.apply(calc_model, axis=1, result_type="expand")
print(data_test.head(20))


"""Оценка качества модели: F1"""

# скоринг \ исходные данные  8  1
#                         8  TP FP
#                         1  FN TN
# Точность=𝑇𝑃/(𝑇𝑃+𝐹𝑃)
# Полнота=𝑇𝑃/(𝑇𝑃+𝐹𝑁)
# 𝐹1=2∗(Точность∗Полнота)/(Точность+Полнота)

print(
    "Случайный выбор:",
    f1_score(
        data_test["random"],
        data_test["Response"],
        average="weighted"
    )
)
print(
    "Выбор по частоте:",
    f1_score(
        data_test["sample"],
        data_test["Response"],
        average="weighted"
    )
)
print(
    "Кластеризация:",
    f1_score(
        data_test["target"],
        data_test["Response"],
        average="weighted"
    )
)
print(
    "Самый популярный:",
    f1_score(
        data_test["all8"],
        data_test["Response"],
        average="weighted"
    )
)


"""Матрица неточностей"""

# скоринг \ исходные данные  1   2   3   4   5   6   7   8
#                         1  TP FP1 FP2 FP3 FP4 FP5 FP6 FP7
#                         2  FN1 TP FP1 FP2 FP3 FP4 FP5 FP6
#                         3  FN2 FN1 TP FP1 FP2 FP3 FP4 FP5
#                         4  FN3 FN2 FN1 TP FP1 FP2 FP3 FP4
#                         5  FN4 FN3 FN2 FN1 TP FP1 FP2 FP3
#                         6  FN5 FN4 FN3 FN2 FN1 TP FP1 FP2
#                         7  FN6 FN5 FN4 FN3 FN2 FN1 TP FP1
#                         8  FN7 FN6 FN5 FN4 FN3 FN2 FN1 TP

print(confusion_matrix(data_test["target"], data_test["Response"]))


"""Квадратичный коэффициент каппа Коэна"""

# Является логичным продолжением оценке по матрице неточностей, но более точно
# указывает на соответствие вычисленных значений реальным, поскольку используем
# матрицу весов: большая ошибка получает больший вес.
#
# Для расчета требуется вычислить матрицу весов (W), 8x8 выглядит так
# (каждый элемент - это квадрат разницы между номером строки и номером столба,
# разделенный на 64):

# матрица весов         1         2         3         4         5         6         7         8
#             1         0  0.015625    0.0625  0.140625      0.25  0.390625    0.5625  0.765625
#             2  0.015625         0  0.015625    0.0625  0.140625      0.25  0.390625    0.5625
#             3    0.0625  0.015625         0  0.015625    0.0625      0.25  0.390625    0.5625
#             4  0.140625    0.0625  0.015625         0  0.015625    0.0625      0.25  0.390625
#             5      0.25  0.140625    0.0625  0.015625         0  0.015625    0.0625      0.25
#             6  0.390625      0.25  0.140625    0.0625  0.015625         0  0.015625    0.0625
#             7    0.5625  0.390625      0.25  0.140625    0.0625  0.015625         0  0.015625
#             8  0.765625    0.5625  0.390625      0.25  0.140625    0.0625  0.015625         0


# После вычисления матрицы неточностей (O) вычисляют матрицу гистограмм
# расчетных и идеальных значений (E) - сколько всего оценок 1, оценок 2, и т.д.
# В случае оценок от 1 до 8 гистограммы будут выглядеть следующим образом:
#
# Расчет: [3372, 661, 1244, 1040, 1380, 276, 900, 3004]
#
# Идеал: [1193, 1302, 207, 261, 1120, 2257, 1633, 3904]
#
# Каждый элемент матрицы ij - это произведение i-расчетного значения
# на j-идеальное. Например, для ячейки 1-1 это будет 3372 * 1193 = 4022796.
# И т.д.
#
# Матрицу неточностей и матрицу гистограмм нормируют (делят каждый элемент
# матрицы на сумму всех элементов) и вычисляют взвешенную сумму, используя
# матрицу весов (каждый элемент матрицы весов умножают на соответствующий
# элемент другой матрицы, все произведения суммируют): e = W * E, o = W * O.
#
# Значение Kappa (каппа) вычисляется как 1 - o/e.

print(
    cohen_kappa_score(
        data_test["target"],
        data_test["Response"],
        weights="quadratic"
    )
)
print(
    cohen_kappa_score(
        data_test["all8"],
        data_test["Response"],
        weights="quadratic"
    )
)
