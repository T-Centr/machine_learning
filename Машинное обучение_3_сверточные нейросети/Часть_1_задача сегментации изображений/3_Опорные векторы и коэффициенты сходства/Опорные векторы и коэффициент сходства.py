"""
Постановка задачи

Загрузим подготовленные данные из HDF5. Разделим данные на обучающие и
проверочные и построим модель опорных векторов для типа облака Fish.

Проведем оценку качества предсказания по F1 и коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/clouds.data.h5 (959 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


"""Загрузка данных"""

clouds = pd.read_hdf("clouds.data.h5")
print(clouds.head())

# Оставим только данные по Fish, на них обучим модель

clouds.drop(
    labels=["Image", "Flower", "Gravel", "Sugar"],
    axis=1,
    inplace=True
)


"""Разделение данных"""

# Разделим всю выборку на 2 части случайным образом: 80% - для обучения модели,
# 20% - для проверки точности модели.

clouds_train, clouds_test = train_test_split(clouds, test_size=0.2)
clouds_train = pd.DataFrame(clouds_train)
clouds_test = pd.DataFrame(clouds_test)
del clouds
print(clouds_train.head())


"""Метод опорных векторов"""

# Последовательно рассчитываем коэффициенты для пакетов по 100 изображений,
# иначе может не поместиться в оперативную память.
# Используем warm_start=True, чтобы переиспользовать предыдущие параметры.

y = clouds_train["Fish"].notnull().astype("int8")
x = pd.DataFrame(clouds_train).drop(labels=["Fish"], axis=1)
model = SGDClassifier(loss="log", warm_start=True)

for i in range(len(clouds_train) // 100):
    model.partial_fit(
        x[i * 100:i * 100 + 100],
        y[i * 100:i * 100 + 100],
        classes=[0, 1]
    )

del x
del y


"""Средняя область облаков"""

# В качестве локализации облаков на изображении возьмем среднюю область
# по обучающей выборке

image_x = 2100
image_x_4 = image_x // 4
image_y = 1400
image_y_4 = image_y // 4


def locate_rectangle(a):
    vals = [int(i) for i in a[0].split(" ")]
    x = vals[0] // image_y
    y = vals[0] % image_y
    width = (image_x + (vals[-2] + vals[-1]) // image_y - x) % image_x
    height = (vals[-2] + vals[-1]) % image_y - y
    return [x, y, width, height]


areas = pd.DataFrame(clouds_train["Fish"].copy().dropna(axis=0))
areas = areas.apply(locate_rectangle, axis=1, result_type="expand")
coords = np.array(areas.mean() // 4)
print(coords)

sgd_mask = np.zeros(
    image_x_4 * image_y_4,
    dtype="uint8"
).reshape(image_x_4, image_y_4)
for x in range(image_x_4):
    for y in range(image_y_4):
        if (x >= coords[0] and
            x <= (coords[0] + coords[3]) and
            y >= coords[1] and
            y <= (coords[1] + coords[3])):
            sgd_mask[x][y] = 1
sgd_mask = sgd_mask.reshape(image_x_4 * image_y_4)
print(sgd_mask.sum())


"""Предсказание значений"""

result = pd.DataFrame(
    {
        "EncodedPixels": clouds_test["Fish"],
        "Is_Fish": clouds_test["Fish"].notnull().astype("int8")
    }
)
result["target"] = model.predict(
    pd.DataFrame(clouds_test).drop(
        labels=["Fish"],
        axis=1
    )
)
print(result.head(10))


"""Оценка точности предсказания: F1"""

# Точность = TruePositive / (TruePositive + FalsePositive)
# Полнота = TruePositive / (TruePositive + FalseNegative)
# F1 = 2 * Полнота * Точность / (Полнота + Точность)

print(
    "Опорные векторы:", round(f1_score(result["Is_Fish"], result["target"]), 3)
)
print(
    "Все Fish:", round(f1_score(result["Is_Fish"], np.ones(len(result))), 3)
)


"""Оценка по Дайсу"""

# Для каждого изображения и каждой фигуры считается пересечение площади
# обнаруженной фигуры (X) с ее реальной площадью (Y) по формуле:
# 𝐷𝑖𝑐𝑒=2 ∗ |𝑋∩𝑌| / |𝑋|+|𝑌|
# Если и X, и Y равны 0, то оценка принимается равной 1. Оценка берется как
# среднее по всем фигурам.
# Пока будем считать, что при определении типа облака на изображении, оно
# целикомразмещено на фотографии: т.е. область облака - это все изображение.
# Дополнительно посчитаем точность предсказания, что на фотографиях вообще
# нет облаков нужного типа.

image_x = 525
image_y = 350


def mask_rate(a, x, y):
    b = a // 1400 + 0.0
    return np.round(
        x * (b * x // 2100) + y * (a % 1400) // 1400
    ).astype("uint32")


def calc_mask(px, x=image_x, y=image_y):
    p = np.array([int(n) for n in px.split(" ")]).reshape(-1, 2)
    mask = np.zeros(y * x, dtype="uint8")
    for i, l in p:
        mask[mask_rate(i, x, y) - 1:mask_rate(l + i, x, y)] = 1
    return mask.reshape(y, x).transpose()


def calc_dice(x):
    dice = 0
    px = x["EncodedPixels"]
    if px != px and x["target"] == 0:
        dice = 1
    elif px == px and x["target"] == 1:
        mask = calc_mask(px).flatten()
        # target = sgd_mask
        target = np.ones(image_x * image_y, dtype="uint8")
        dice = 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
    return dice


dice = result.apply(calc_dice, axis=1, result_type="expand")
print("Опорные векторы, Fish:", round(dice.mean(), 3))
print(
    "Нет облаков, Fish:",
    round(len(result[result["Is_Fish"] == 0]) / len(result), 3)
)
