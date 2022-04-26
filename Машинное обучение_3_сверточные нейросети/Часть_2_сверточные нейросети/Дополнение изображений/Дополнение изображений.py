"""
Постановка задачи

Загрузим уменьшенные изображения, уменьшим их еще в 2.5 раза и применим к ним
дополнение (augmentation): смещение, контрастность, поворот, размытие.

Загрузим данные и разделим их на обучающие и проверочные в соотношении 80/20.

Используем Keras для построения нейросети с линейным, сверточными слоями и
слоями подвыборки.

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
from scipy import ndimage
from skimage import transform,util,exposure,io,color
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""


image_x = 210
image_y = 140
image_ch = 1


def mask_rate(a, x, y):
    b = a//1400 + 0.0
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
            yield (load_x(df[batch_start:limit]), load_y(df[batch_start:limit]))
            batch_start += batch_size
            batch_end += batch_size


def draw_prediction(prediction):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(prediction[0])
    ax.set_title("Fish")
    plt.show()


"""Обработка и дополнение изображений"""

# Преобразуем изображение: уменьшим в 2,5 раза, добавим случайный шум, повернем
# на 5 градусов по часовой и против часовой стрелки, увеличим контрастность,
# изменим гамму и добавим размытие по 3 и 7 пикселям


def img_save(name, data):
    io.imsave(os.path.join(filesDir, name), (data * 255).astype(np.uint8))


def img_aug(name):
    if not os.path.isdir(filesDir):
        os.mkdir(filesDir)
    img = io.imread(os.path.join(dir_, name))
    img = transform.rescale(img, 1 / 2.5)
    img_save(name, img)
    img_save("noised_" + name, util.random_noise(img))
    img_save("rotcw_" + name, transform.rotate(img, -5))
    img_save("rotccw_" + name, transform.rotate(img, 5))
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    img_save(
        "cont_" + name, exposure.rescale_intensity(img, in_range=(v_min, v_max))
    )
    img_save("gamma_" + name, exposure.adjust_gamma(img, gamma=0.4, gain=0.9))
    img_save("blurred3_" + name, ndimage.uniform_filter(img, size=(3, 3, 1)))
    img_save("blurred7_" + name, ndimage.uniform_filter(img, size=(7, 7, 1)))


# Обработаем все изображения обучающей выборки

dir_ = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
filesDir = dir_ + "_tiny"

for file in os.listdir(dir_):
    img_aug(file)


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


"""Дополнение обучающих данных"""

# Дополним все измененные изображения типами и областями облаков

train.set_index("Image", inplace=True)
fileList = os.listdir(filesDir)
for file in fileList:
    img = file.split("_")
    if (file not in train.index.values and len(img) > 1 and
       img[1] in train.index.values):
        train.loc[file] = [train.loc[img[1]]["EncodedPixels"], "Fish"]
train.reset_index(inplace=True)
print(train.head())


"""Сверточная нейросеть"""

# Создадим и построим модель

model = Sequential([
    Conv2D(
        32,
        (3, 3),
        input_shape=(image_y, image_x, image_ch),
        kernel_initializer="glorot_uniform",
        strides=(2, 2)
    ),
    Activation("relu"),
    Conv2D(
        32,
        (3, 3),
        kernel_initializer="glorot_uniform",
        strides=(2, 2)
    ),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Activation("softmax"),
    Dense(1)
])

model.compile(
    optimizer=optimizers.nadam_v2.Nadam(lr=0.02),
    loss="mean_absolute_error"
)


"""Обучение модели"""

# Используем для обучения все изображения, включая измененные

batch_size = 100
model.fit_generator(
    load_data(train, batch_size),
    epochs=100,
    steps_per_epoch=len(train) // batch_size
)


"""Предсказание значений"""

prediction = model.predict_generator(
    load_data(test, 1),
    steps=len(test),
    verbose=1
)
prediction = np.transpose(prediction)
draw_prediction(prediction)
test["target"] = (prediction[0] >= 1).astype("int8")
print(test[test["target"] > 0][["EncodedPixels", "target"]].head(100))


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

# Нет облаков - 0.5, MLP - 0.3, CONV3-32x2,POOL2 - 0.48

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, (CONV3-32x2,POOL2):", round(dice.mean(), 3))
