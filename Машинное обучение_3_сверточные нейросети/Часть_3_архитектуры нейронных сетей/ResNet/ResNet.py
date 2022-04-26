"""
Постановка задачи

Разберем архитектуру ResNet для решения задач распознавания изображений.
Построим эту нейросеть для анализа исходных изображений.

Используя обученную модель, построим предсказания и классификацию, используя
LightGBM. Проведем оценку качества предсказания по коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)
https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz
212 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from skimage import io
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Activation, GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
import lightgbm as lgb
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
batch_size = 20
image_x = 224
image_y = 224
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


def calc_dice (x):
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
        img = image.load_img(
            os.path.join(filesDir, file), target_size=(image_y, image_x))
        img = image.img_to_array(img)
        x[j] = np.expand_dims(img, axis=0)
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


"""ResNet-50"""

# Подключим обученную нейросеть (90 Мб) и построим поверх классификатора
# LightGBM.

model = ResNet50(weights="imagenet")

train_prediction = model.predict_generator(
    load_data(train, 1),
    steps=len(train),
    verbose=1
)


"""Обучение LightGBM по результатам ResNet"""

model_lgb = lgb.LGBMRegressor(random_state=17)
model_lgb.fit(
    pd.DataFrame(train_prediction),
    train["EncodedPixels"].notnull().astype("i1")
)
prediction = model_lgb.predict(train_prediction)
prediction = np.array(prediction).reshape(1, -1)
draw_prediction(prediction)

train["target"] = (prediction[0] > 0.75).astype("i1")
print(train[train["target"] > 0]["EncodedPixels"])

dice = train.apply(calc_dice, axis=1, result_type="expand")
print("Keras, Resnet-50+LightGBM, обучение:", round(dice.mean(), 3))


"""Построение предсказания"""

test_prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
prediction = model_lgb.predict(test_prediction)
prediction = np.array(prediction).reshape(1, -1)
draw_prediction(prediction)

test["target"] = (prediction[0] > 0.75).astype("i1")


"""Расчет точности предсказания"""

# Нет облаков - 0.5, MLP - 0.3, CONV/VGG - 0.48, AlexNet - 0.21, Inception - 0.5

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, ResNet-50+LightGBM:", round(dice.mean(), 3))
