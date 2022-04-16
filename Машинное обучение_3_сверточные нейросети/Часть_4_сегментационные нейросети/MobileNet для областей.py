"""
Постановка задачи
Разберем архитектуру MobileNet и проведем частичное обучение, экспорт и импорт
модели, затем дообучение.

Перейдем от задачи классификации изображения к задаче локализации объекта на
изображении при помощи якорей.

Построим предсказание по обученной нейросети и проведем оценку качества
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
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

# Разобьем изображение на 5x5 квадратов и будем предсказывать наличие облаков
# заданной формы в каждом из квадратов.

# При вычислении коэффициента Дайса "растянем" предсказанную маску до размеров
# изображения.

batch_size = 10
filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
image_x = 525
image_y = 350
image_ch = 3
mask_x = 5
mask_y = 5


def mask_rate(a, x, y):
    b = a // 1400 + 0.0
    return round(
        x * (b * x // 2100) + y * (a % 1400) // 1400
    ).astype("uint32")


def calc_mask(px, x=image_x, y=image_y):
    p = np.array([int(n) for n in px.split(' ')]).reshape(-1, 2)
    mask = np.zeros(x * y, dtype='uint8')
    for i, l in p:
        mask[mask_rate(i, x, y):mask_rate(l + i, x, y)+1] = 1
    return mask.reshape(y, x).transpose()


def calc_dice(x):
    dice = 0
    px = x["EncodedPixels"]
    if px != px and x["target"] == 0:
        dice = 1
    elif px == px and x["target"] == 1:
        mask = calc_mask(px).flatten()
        target = np.kron(
            np.array(
                x["MaskPixels"].split(" ")
            ).reshape(mask_x, mask_y).astype("i1"),
            np.ones((image_y // mask_y, image_x // mask_x),
            dtype="i1")
        ).transpose().flatten()
        dice = 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
    return dice


def load_y(df):
    y = [[0]] * len(df)
    for i, ep in enumerate(df["EncodedPixels"]):
        if ep == ep:
            y[i] = calc_mask(ep, mask_x, mask_y).transpose().flatten()
        else:
            y[i] = np.zeros(mask_x * mask_y, dtype="i1")
    return np.array(y).reshape(len(df), mask_y, mask_x, 1)


def load_x(df):
    x = [[]] * len(df)
    for j, file in enumerate(df["Image"]):
        img = image.load_img(
            os.path.join(filesDir, file),
            target_size=(image_y, image_x)
        )
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x[j] = preprocess_input(img)
    return np.array(x).reshape(len(df), image_y, image_x, image_ch)


def load_data(df, batch_size):
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < len(df):
            limit = min(batch_end, len(df))
            yield (
                load_x(df[batch_start:limit]), load_y(df[batch_start:limit])
            )
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


"""MobileNetV2"""

# Используем обученную нейросеть и bn/dropout/softmax-слой поверх. Обучим модель
# из последнего слоя 50 эпох, сохраним обученные данные, затем загрузим их и
# продолжим обучение.

# Используем раннюю остановку и сохранение модели после каждой эпохи обучения.

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(image_y, image_x, image_ch)
)
base_model_output = base_model.predict_generator(
    load_data(train, 1),
    steps=len(train),
    verbose=1
)
top_model = Sequential([
    Flatten(input_shape=base_model_output.shape[1:]),
    BatchNormalization(),
    Dropout(0.5),
    Activation("softmax"),
    Dense(mask_x * mask_y)
])
top_model.compile(
    optimizer=optimizers.Nadam(lr=0.05),
    loss="mean_absolute_error"
)
top_model.fit(
    base_model_output,
    load_y(train).reshape(len(train), -1),
    epochs=100,
    callbacks=[
        EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=5,
            verbose=1,
            mode="auto"
        ),
        ModelCheckpoint(
            "mobilenet.clouds.h5",
            mode="auto",
            monitor="val_loss",
            verbose=1
        )
    ]
)


"""Продолжение обучения"""

# Загрузим модель из файла (структура + веса) и продолжим обучение

del top_model

top_model = keras.models.load_model("mobilenet.clouds.h5")
top_model.fit(
    base_model_output,
    load_y(train).reshape(len(train), -1),
    epochs=100,
    callbacks=[
        EarlyStopping(
            monitor="loss",
            min_delta=0.0001,
            patience=5, verbose=1,
            mode="auto"
        ),
        ModelCheckpoint(
            "mobilenet.clouds.h5",
            mode="auto",
            monitor="val_loss",
            verbose=1
        )
    ]
)

# Соберем модель из обученного финального слоя и базовой модели

model = Model(
    inputs=base_model.input,
    outputs=top_model(base_model.output)
)
model.compile(optimizer="adam", loss="mean_absolute_error")
model.summary()


"""Построение предсказания"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
prediction = (prediction > 0.5).astype("i1")
print(prediction[0:10])
print(load_y(test.head(10)).reshape(10, -1))

target = []
masks = []
for i,vals in enumerate(prediction):
    if vals.sum() > 4:
        targ = 1
    else:
        targ = 0
    target.append(targ)
    masks.append(
        np.array2string(vals.flatten().astype("int8"), separator=" ")[1:-1]
    )
test["MaskPixels"] = masks
test["target"] = target
print(test[test["target"] > 0][["EncodedPixels", "MaskPixels"]].head(20))


"""Расчет точности предсказания"""

# Нет облаков - 0.5, MLP - 0.3, CONV/VGG - 0.48, AlexNet - 0.21,
# Inception - 0.4, ResNet - 0.52, VGG16+Inception+ResNet - 0.54

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, MobileNet:", round(dice.mean(), 3))
