"""
Постановка задачи

Загрузим подготовленные данные из HDF5.

Разделим данные на обучающие и проверочные в соотношении 80/20.

Используем Keras для построения нейросети с линейным, сверточными слоями и
слоями подвыборки. Проверим, какая конфигурация работает лучше линейных слоев.

Проведем оценку качества предсказания по коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)

https://video.ittensive.com/machine-learning/
clouds/train_images_small.tar.gz (212 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from skimage import io
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["PATH"] += (os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/')


"""Используемые функции"""


labels = ["Fish", "Flower", "Gravel", "Sugar"]
image_x = 525
image_y = 350
image_ch = 3


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
        target = np.ones(image_x * image_y, dtype='uint8')
        dice += 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
    return dice


"""Загрузка данных"""

data = pd.read_csv(
    'https://video.ittensive.com/machine-learning/clouds/train.csv.gz'
)
data["Image"] = data["Image_Label"].str.split("_").str[0]
data["Label"] = data["Image_Label"].str.split("_").str[1]
data.drop(labels=["Image_Label"], axis=1, inplace=True)
data_fish = data[data["Label"] == "Fish"]
print(data_fish.head())


"""Разделение данных"""

# Разделим всю выборку на 2 части случайным образом: 80% - для обучения модели,
# 20% - для проверки точности модели.

train, test = train_test_split(data_fish, test_size=0.2)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
del data
print(train.head())


"""Сверточная нейросеть"""

# Подготовим данные для обучения нейросети: будем загружать данные пакетами.
# Создадим функции для загрузки двух типов данных: графических из изображения и
# типа облака на изображении.


def load_y(df):
    return np.array(df["EncodedPixels"].notnull().astype("int8")).reshape(-1, 1)


def load_x(df):
    x = [[]] * len(df)
    for j, file in enumerate(df["Image"]):
        x[j] = io.imread(os.path.join(filesDir, file))
    return np.array(x).reshape(len(df), image_y, image_x, image_ch)


def load_data(df, batch_size):
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < len(df):
            limit = min(batch_end, len(df))
            yield (load_x(df[batch_start:limit]),
                  load_y(df[batch_start:limit]))
            batch_start += batch_size
            batch_end += batch_size

# Сверточный слой (Conv2D) применяет ядро преобразования (набор математических
# операций) к области исходного изображения (набора входов с предыдущего слоя)
# для выделения особенностей (например, определенных фигур - линий или уголков).
# Принимает в качестве входной формы только двумерные изображения и число
# цветовых каналов (трехмерный массив данных на 1 изображение).Сверточный слой
# "размноживает" исходное изображение: используется заданное (большое) число
# ядер свертки, которые оптимизируются на этапе обучения нейросети. Поэтому к
# полученным при свертке данным обычно применяют слой подвыборки (MaxPooling):
# выделяют самый значимый из квадрата 2x2 или 3x3 элемент, обнаруженный на
# сверточной слое, чтобы снизить число выходов и ускорить обучение нейросети.
# Свертку осуществляем с шагом (strides) 2.
#
# На выходе слоя подвыборки находится двумерный массив нейронов, полученный
# выборкой из множества преобразований исходного изображения, поэтому его нужно
# переформировать, перевести в одномерный. Для этого используется плоский
# слой (Flatten).

model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            input_shape=(image_y, image_x, image_ch),
            strides=(2, 2)
        ),
        Activation("relu"),
        Conv2D(32, (3, 3), strides=(2, 2)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Activation("softmax"),
        Dense(1)
    ]
)


"""Топология модели"""

# Потребуется graphviz

# keras.utils.plot_model(
tf.keras.utils.plot_model(
    model,
    to_file="ml0041.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB"
)
filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
batch_size = 20

# Построим и обучим модель, используя адаптивный градиентный спуск и
# абсолютную ошибку.

model.compile(optimizer="adam", loss="mean_absolute_error")
model.fit_generator(
    load_data(train, batch_size),
    epochs=100,
    steps_per_epoch=len(train) // batch_size,
    verbose=True
)


"""Предсказание значений"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)


def draw_prediction(prediction):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(prediction[0])
    ax.set_title("Fish")
    plt.show()


prediction = np.transpose(prediction)
draw_prediction(prediction)

test["target"] = (prediction[0] >= 1).astype("int8")
print(test[test["target"] > 0][["EncodedPixels", "target"]].head(20))


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

# Нет облаков - 0.5, MLP - 0.5

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Kers, (CONV3-32x2,POOL2):", round(dice.mean(), 3))
