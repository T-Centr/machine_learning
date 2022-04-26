"""
Постановка задачи

Построим сеть UNet для локализации объекта на изображении.

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
import matplotlib.image as mpimg
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras import optimizers
from keras import backend as K
import os
import sys


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

batch_size = 10
filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
image_x = 480
image_y = 320
image_ch = 3
mask_x = 480
mask_y = 320


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
        target = np.array(x["TargetPixels"].split(" ")).astype("int8")
        dice += 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
    return dice


def load_y(df):
    y = [[0]] * len(df)
    for i, ep in enumerate(df["EncodedPixels"]):
        if ep == ep:
            y[i] = calc_mask(ep, mask_x, mask_y).transpose().flatten()
        else:
            y[i] = np.zeros(mask_x*mask_y, dtype="i1")
    return np.array(y).reshape(len(df), mask_y, mask_x, 1)


def load_x(df):
    x = [[]] * len(df)
    for j, file in enumerate(df["Image"]):
        img = image.load_img(
            os.path.join(filesDir, file),
            target_size=(image_y, image_x)
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
            yield (
                load_x(df[batch_start:limit]),
                load_y(df[batch_start:limit])
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


"""UNet"""

# Построим нейросеть, реализация модели:
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/
# train.py


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Размеры исходных изображений должны делиться на 8 для успешной
# свертки/подвыборки и объединения слоев.

inputs = Input((image_y, image_x, image_ch))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

# Выбор скорости обучения крайне критичен для UNet: при большой скорости сеть
# попадает в локальный минимум и считает все изображение за характерную область.

model.compile(
    optimizer=optimizers.nadam_v2.Nadam(lr=1e-5),
    # optimizer=optimizers.Nadam(lr=1e-5),
    loss=dice_coef_loss,
    metrics=["mae"]
)
model.summary()

# loss > 0,7 - статистический шум, 0,7 - везде облака, 0,5 - нет облаков,
# < 0.5 - начинаем определять реальные облака


# В данном примере 'model.fit_generator()' выдает ошибку (пока непонятную)
# поэтому закомментируем

# model.fit_generator(
#     load_data(train, batch_size),
#     epochs=20,
#     steps_per_epoch=len(train) // batch_size
# )


"""Построение предсказания"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
pred = prediction[0].reshape(image_y, image_x).astype("int8")
print(test.head(10))
sample = test[test["EncodedPixels"].notnull()][1:2]

# Построим изображение с UNet маской и с исходной для сравнения

img = image.load_img(
    os.path.join(filesDir, sample["Image"].values[0]),
    target_size=(image_y, image_x)
)
img = image.img_to_array(img).astype("uint8")
mask = calc_mask(sample["EncodedPixels"].values[0])
fig = plt.figure(figsize=(32, 16))
for i in range(2):
    area = fig.add_subplot(1, 2, i + 1)
    area.axis("off")
    if i == 0:
        area.set_title("U-Net")
        segmap = SegmentationMapsOnImage(pred, pred.shape)
    else:
        area.set_title("Fish")
        segmap = SegmentationMapsOnImage(mask, mask.shape)
    area.imshow(np.array(segmap.draw_on_image(img)).reshape(img.shape))
plt.show()

np.set_printoptions(threshold=sys.maxsize)
target = []
masks = []
for i, vals in enumerate(prediction):
    if vals.sum() > mask_x * mask_y / 10:
        targ = 1
    else:
        targ = 0
    target.append(targ)
    masks.append(
        np.array2string(vals.flatten().astype("int8"), separator=" ")[1:-1]
    )
test["target"] = target
test["TargetPixels"] = masks
print(test.head(20))

"""Расчет точности предсказания"""

# Нет облаков - 0.5, MLP - 0.3, CONV/VGG - 0.48, AlexNet - 0.21,
# Inception - 0.4, ResNet - 0.52, VGG16+Inception+ResNet - 0.54, MobileNet - 0.5

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, U-Net:", round(dice.mean(), 3))
