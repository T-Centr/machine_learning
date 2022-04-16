"""
Постановка задачи

Загрузите подготовленные данные из HDF5.

Разделите данные на обучающие и проверочные в соотношении 80/20.

Постройте многослойный перцептрон для типа облака Fish. Найдите наилучшее число
нейронов для 2 скрытых слоев: 1 входной, 2 скрытых, 1 выходной - всего 4 слоя.
Не более 30 нейронов на первом скрытом слое.

Проведите оценку качества предсказания по коэффициенту сходства.

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
    b = a//1400 + 0.0
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
        dice += 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
    return dice


"""Загрузка данных"""

clouds = pd.read_hdf('clouds.data.h5')
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


"""Многослойный перцептрон"""

# Зададим исходное число нейронов в каждом слое и последовательно проверим
# конфигурации слоев до достижения лучшего результата

result = pd.DataFrame({"Is_Fish": clouds_test["Fish"].notnull().astype("int8")})
y = clouds_train["Fish"].notnull().astype("int8")
x = pd.DataFrame(clouds_train).drop(labels=["Fish"], axis=1)
max_acc = 0
for i in range(5, 30):
    for k in range(2, i // 2 + 1):
        model = MLPClassifier(
            hidden_layer_sizes=(i, k, ),
            max_iter=20,
            activation="logistic",
            random_state=1,
            learning_rate_init=0.02
        )
        for j in range(0, len(clouds_train), 100):
            model.partial_fit(x[j:j + 100], y[j:j + 100], classes=[0, 1])
        result["target"] = model.predict(
            clouds_test.drop(
                labels=["Fish"],
                axis=1
            )
        )
        acc = result[
                  result["Is_Fish"] == result["target"]
              ].count()["target"] / len(result)
        if acc > max_acc:
            layers = (i, k, )
            max_acc = acc
        print(layers, acc)
print(layers)


"""Построение оптимальной модели"""

model = MLPClassifier(
    hidden_layer_sizes=layers,
    max_iter=20,
    activation="logistic",
    random_state=1,
    learning_rate_init=0.02
)
for j in range(0, len(clouds_train), 100):
    model.partial_fit(x[j:j + 100], y[j:j + 100], classes=[0, 1])


"""Предсказание значений"""

result = pd.DataFrame({"EncodedPixels": clouds_test["Fish"]})
result["target"] = model.predict(
    pd.DataFrame(clouds_test).drop(labels=["Fish"], axis=1)
)
print(result.head(10))


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

# Нет облаков - 0.5, опорные векторы - 0.3, MLP(31) - 0.5

dice = result.apply(calc_dice, axis=1, result_type="expand")
print("MLP, Fish:", round(dice.mean(), 3))
