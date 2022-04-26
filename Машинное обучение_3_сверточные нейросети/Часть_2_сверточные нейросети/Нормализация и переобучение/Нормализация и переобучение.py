"""
Постановка задачи

Загрузим данные и разделим их на обучающие и проверочные в соотношении 80/20.

Используем Keras для построения нейросети с линейным, сверточными слоями и
слоями подвыборки. Проверим, как повышают эффективность обучения нормализация
данных и отсев.

Обучим модель, используя последовательную загрузку данных. Проведем оценку
качества предсказания по коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)
https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz
(212 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from skimage import io
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

image_x = 525
image_y = 350
image_ch = 3


def mask_rate(a, x, y):
    b = a // 1400 + 0.0
    return np.round(
        x * (b * x // 2100) + y * (a % 1400) // 1400
    ).astype("uint32")


def calc_mask(px, x=image_x, y=image_y):
    p = np.array([int(n) for n in px.split(' ')]).reshape(-1, 2)
    mask = np.zeros(x * y, dtype='uint8')
    for i, l in p:
        mask[mask_rate(i, x, y) - 1:mask_rate(l+i, x, y)] = 1
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


def load_y(df):
    return np.array(
        df["EncodedPixels"].notnull().astype("int8")
    ).reshape(len(df), 1)


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


def draw_prediction(prediction):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(prediction[0])
    ax.set_title("Fish")
    plt.show()


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


"""Генератор данных для обучения"""

# Данные всех изображений могут не поместиться в оперативную память, поэтому
# будем обучать нейросеть последовательно, пакетами.

filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
batch_size = 50


"""Сверточная нейросеть"""

# Создадим модель. Архитектура в общем виде:
# INPUT -> [[CONV -> RELU] * N -> POOL?] * M -> [FC -> RELU] * K -> FC * L

model = Sequential([
    Conv2D(
        32,
        (3, 3),
        input_shape=(image_y, image_x, image_ch),
        kernel_initializer='glorot_uniform',
        strides=(2, 2)
    ),
    Activation("relu"),
    BatchNormalization(),
    Dropout(0.5),
    Conv2D(
        32,
        (3, 3),
        kernel_initializer='glorot_uniform',
        strides=(2, 2)
    ),
    Activation("relu"),
    BatchNormalization(),
    Dropout(0.5),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Activation("softmax"),
    Dense(1)
])

model.compile(
    optimizer=optimizers.nadam_v2.Nadam(lr=0.02),
    loss="mean_absolute_error"
)


"""Предсказание значений"""

model.fit_generator(
    load_data(train, batch_size),
    epochs=50,
    steps_per_epoch=len(train) // batch_size
)

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)

prediction = np.transpose(prediction)
draw_prediction(prediction)

test["target"] = (prediction[0] >= 0.95).astype("int8")
print(test[test["target"] > 0][["EncodedPixels", "target"]].head(100))


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

# Нет облаков - 0.5, MLP - 0.3, CONV3-32x2,POOL2 - 0.46

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, (CONV3-32x2,POOL2):", round(dice.mean(), 3))
