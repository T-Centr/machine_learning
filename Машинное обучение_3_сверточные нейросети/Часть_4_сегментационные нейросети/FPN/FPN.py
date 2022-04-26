"""
Постановка задачи

Разберем работу пирамидальной сети (FPN) для сегментации объектов на
изображении.

Обучим нейросеть и построим предсказания области облака. Проведем оценку
качества предсказания по коэффициенту сходства.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)
https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz
(212 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import segmentation_models
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import pandas as pd
import keras
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import sys


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

batch_size = 5
filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
image_x = 384
image_y = 256
image_ch = 3
mask_x = 384
mask_y = 256


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
        dice = 2 * np.sum(target[mask == 1]) / (np.sum(target) + np.sum(mask))
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
                preprocess_input(load_x(df[batch_start:limit])),
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
del data_fish
print(train.head())


"""FPN"""

# Используем архитектуру ResNet50 и предобученную модель

BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.FPN(
    BACKBONE,
    encoder_weights='imagenet',
    encoder_freeze=True,
    classes=1,
    activation="sigmoid"
)
model.compile(
    optimizer=optimizers.nadam_v2.Nadam(lr=0.01),
    loss=sm.losses.dice_loss,
    metrics=[sm.metrics.iou_score]
)
model.summary()
model.fit_generator(
    load_data(train, batch_size),
    epochs=10,
    steps_per_epoch=len(train) // batch_size,
    callbacks=[ModelCheckpoint("clouds.h5", mode='auto', monitor='val_loss')]
)


"""Построение предсказания"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
pred = prediction[0].reshape(image_y, image_x).astype("uint8")
print(test.head(10))
sample = test[test["EncodedPixels"].notnull()][0:1]

# Построим изображение с FPN маской и с исходной для сравнения

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
        area.set_title("FPN")
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
    if vals.sum() > mask_x * mask_y/10:
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
print("Keras, FPN:", round(dice.mean(), 3))
