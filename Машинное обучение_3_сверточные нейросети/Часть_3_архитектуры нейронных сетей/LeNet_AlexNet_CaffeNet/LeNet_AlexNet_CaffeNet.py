"""
Постановка задачи

Разберем архитектуру LeNet и AlexNet для решения задач распознавания
изображений. Применим их для анализа исходных изображений.

Обучим модели, используя последовательную загрузку данных. Проведем оценку
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
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, ZeroPadding2D
from keras import optimizers
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
batch_size = 20
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
        mask[mask_rate(i, x, y) - 1: mask_rate(l + i, x, y)] = 1
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


"""LeNet5"""

# Первая успешная архитектура сверточной нейросети, 1998

lenet = Sequential([
    Conv2D(
        6,
        (5, 5),
        input_shape=(image_y, image_x, image_ch),
        kernel_initializer="glorot_uniform",
        strides=(1, 1)
    ),
    Activation("relu"),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(
        16,(5, 5),
        kernel_initializer="glorot_uniform",
        strides=(1, 1)
    ),
    Activation("relu"),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Activation("tanh"),
    Dense(120),
    Activation("tanh"),
    Dense(84),
    Activation("softmax"),
    Dense(1)
])


"""Обучение модели"""

# Обучим построенную модель и вычислим ее точность, используя самый лучший
# разделитель значений


def train_evaluate_model(model):
    model.compile(
        optimizer=optimizers.Nadam(lr=0.05),
        loss="mean_absolute_error"
    )
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
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(prediction[0])
    ax.set_title("Fish")
    plt.show()
    acc = prediction[0].mean()
    acc_max = prediction[0].max()
    if acc == acc_max:
        test["target"] = np.round(prediction[0])
        return test.apply(calc_dice, axis=1, result_type="expand").mean()
    else:
        dice_best = 0
        for i in range(0, 20):
            acc += (acc_max - acc) * i / 20
            test["target"] = (prediction[0] >= acc).astype("int8")
            dice = test.apply(calc_dice, axis=1, result_type="expand")
            if dice_best < dice.mean():
                dice_best = dice.mean()
            else:
                break
        return dice_best


print("Keras, LeNet:", round(train_evaluate_model(lenet), 3))

del lenet


"""AlexNet и CaffeNet"""

# Первая сверточная нейросеть, победившая в ImageNetCaffeNet - однопроцессорная
# версия AlexNet

# Для реализации потребуется задать шаг свертки (strides) и дополнительные слои
# для заполнения границ изображения нулями, чтобы не уменьшать область после
# свертки.

alexnet = Sequential([
    Conv2D(
        96,
        (11, 11),
        input_shape=(image_y, image_x, image_ch),
        kernel_initializer='glorot_uniform',
        strides=(4, 4)
    ),
    Activation("relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    ZeroPadding2D(padding=(2, 2)),
    Conv2D(256, (5, 5), kernel_initializer='glorot_uniform'),
    Activation("relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    ZeroPadding2D(padding=(1, 1)),
    Conv2D(384, (3, 3), kernel_initializer='glorot_uniform'),
    Activation("relu"),
    BatchNormalization(),
    ZeroPadding2D(padding=(1, 1)),
    Conv2D(384, (3, 3), kernel_initializer='glorot_uniform'),
    Activation("relu"),
    BatchNormalization(),
    ZeroPadding2D(padding=(1, 1)),
    Conv2D(256, (3, 3), kernel_initializer='glorot_uniform'),
    Activation("relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Activation("relu"),
    Dense(1024),
    Activation("relu"),
    Dense(1024),
    Activation("softmax"),
    Dense(1)
])

print("Keras, AlexNet:", round(train_evaluate_model(alexnet), 3))
