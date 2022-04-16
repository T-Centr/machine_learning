"""
Постановка задачи

Загрузим подготовленные данные из HDF5. Разделим данные на обучающие и
проверочные и построим двухслойный перцептрон для типа облака Fish.

Проведем оценку качества предсказания по коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/clouds.data.h5 (959 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


"""Используемые функции"""

image_x = 525
image_y = 350


def mask_rate(a, x, y):
    b = a // 1400 + 0.0
    return np.round(
        x * (b * x // 2100) + y * (a % 1400) // 1400
    ).astype("uint32")


def calc_mask(px, x=image_x, y=image_y):
    p = np.array([int(n) for n in px.split(' ')]).reshape(-1, 2)
    mask = np.zeros(x * y, dtype='uint8')
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
        target = np.ones(image_x*image_y, dtype='uint8')
        dice = 2 * np.sum(target[mask == 1]) / (np.sum(target)+np.sum(mask))
    return dice


"""Загрузка данных"""

clouds = pd.read_hdf('clouds.data.h5')
print(clouds.head())

# Оставим только данные по Fish, на них обучим модель

clouds.drop(labels=["Image", "Flower", "Gravel", "Sugar"],
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


"""Двухслойный перцептрон"""

# Последовательно рассчитываем коэффициенты для пакетов по 100 изображений,
# иначе может не поместиться в оперативную память.
# Используем warm_start=True, чтобы переиспользовать предыдущие параметры.

y = clouds_train["Fish"].notnull().astype("int8")
x = pd.DataFrame(clouds_train).drop(labels=["Fish"], axis=1)
model = MLPClassifier(
    hidden_layer_sizes=(31, ),
    max_iter=20,
    activation="logistic",
    verbose=10,
    random_state=1,
    learning_rate_init=0.02,
    warm_start=True
)

for i in range(len(clouds_train) // 100):
    model.partial_fit(x[i:i + 100], y[i:i + 100], classes=[0, 1])

del x
del y


"""Предсказание значений"""

result = pd.DataFrame({"EncodedPixels": clouds_test["Fish"]})
result["target"] = model.predict(
    clouds_test.drop(
        labels=["Fish"],
        axis=1
    )
)
print(result.head(10))


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

dice = result.apply(calc_dice, axis=1, result_type="expand")
print("MLP, Fish:", round(dice.mean(), 3))
