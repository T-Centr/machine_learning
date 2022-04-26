"""
Постановка задачи

Разберем архитектуру Inception для решения задач распознавания изображений.
Построим эту нейросеть для анализа исходных изображений.

Используя обученную модель, построим предсказания. Проведем оценку качества
предсказания по коэффициенту сходства.

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
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import BatchNormalization, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
batch_size = 50
image_x = 299
image_y = 299
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


def load_y(df):
    return np.array(
        df["EncodedPixels"].notnull().astype("int8")
    ).reshape(len(df), 1)


def load_x(df):
    x = [[]] * len(df)
    for j, file in enumerate(df["Image"]):
        img = image.load_img(
            os.path.join(filesDir, file), target_size=(image_y, image_x)
        )
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


"""Inception v3"""

# Подключим обученную нейросеть (89 Мб) и построим поверх классификатора новые
# слои. Используем результат работы обученной нейросети как входной слой для
# обучения последнего слоя, нашего классификатора.

inc_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(image_y, image_x, image_ch)
)
inc_model.compile(optimizer="sgd", loss="mean_absolute_error")
inc_model_output = inc_model.predict_generator(
    load_data(train, 1),
    steps=len(train),
    verbose=1
)
top_model = Sequential([
    Flatten(input_shape=inc_model_output.shape[1:]),
    BatchNormalization(),
    Dropout(0.5),
    Activation("softmax"),
    Dense(1)
])
top_model.compile(
    optimizer=optimizers.nadam_v2.Nadam(lr=0.02),
    loss="mean_absolute_error"
)


"""Обучение модели"""

top_model.fit(
    inc_model_output,
    load_y(train),
    epochs=100
)
model = Model(
    inputs=inc_model.input,
    outputs=top_model(inc_model.output)
)
model.compile(
    optimizer="adam",
    loss="mean_absolute_error"
)
model.summary()


"""Построение предсказания"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
prediction = np.transpose(prediction)
draw_prediction(prediction)

test["target"] = (prediction[0] >= 1.1).astype("int8")
print(test[test["target"] > 0][["EncodedPixels", "target"]])


"""Расчет точности предсказания"""

# Нет облаков - 0.5, MLP - 0.3, CONV/VGG16 - 0.48, AlexNex - 0.21

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, Inception v3:", round(dice.mean(), 3))
