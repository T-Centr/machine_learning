"""
Постановка задачи

Используйте обученные сети VGG, Inception, ResNet и softmax-слой поверх или
LightGBM-классификатор для независимого предсказания наличия облаков формы Fish
на фотографии.

Постройте простой усредняющий ансамбль из предсказаний. Проведите оценку
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras import optimizers
import lightgbm as lgb
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


"""Используемые функции"""

batch_size = 20
filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small"
image_x = 224
image_y = 224
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
            os.path.join(filesDir, file),
            target_size=(image_y, image_x)
        )
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x[j] = preprocess_input(img)
    return np.array(x).reshape(len(df), image_x, image_y, image_ch)


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


"""VGG16"""

# Используем обученную нейросеть и bn/dropout/softmax-слой поверх.

# При ограниченном размере оперативной памяти, можно сразу после обучения
# провести предсказания - и освободить память под другие модели.

# Сначала получаем предсказания базовой модели по обучающей выборке, и на этих
# предсказаниях обучим только финальный слой: так быстрее. Для общего
# предсказания соберем модель из базовой и финальной.


def build_train_model (base_model):
    base_model.compile(optimizer="sgd", loss="mean_absolute_error")
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
        Dense(1)
    ])
    top_model.compile(
        optimizer=optimizers.nadam_v2.Nadam(lr=0.05),
        # optimizer=optimizers.Nadam(lr=0.05),
        loss="mean_absolute_error"
    )
    top_model.fit(base_model_output, load_y(train), epochs=200)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    return model


model_vgg = build_train_model(
    VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_x, image_y, image_ch)
    )
)
model_vgg.summary()


"""Inception"""

# Проведем аналогичные построения для InceptionV3 модели

model_inc = build_train_model(
    InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(image_x, image_y, image_ch)
    )
)
model_inc.summary()


"""ResNet"""

# Для ResNet используем LightGBM классификатор поверх модели

model_resnet = ResNet50(weights="imagenet")
train_prediction = model_resnet.predict_generator(
    load_data(train, 1),
    steps=len(train),
    verbose=1
)
model_lgb = lgb.LGBMRegressor(random_state=17)
model_lgb.fit(
    pd.DataFrame(train_prediction),
    train["EncodedPixels"].notnull().astype("i1")
)


"""Построение предсказания"""

# Нормализуем предсказания от всех моделей, чтобы корректно просуммировать


def prep_pred(p):
    return np.transpose(MinMaxScaler().fit_transform(p))


prediction_vgg = prep_pred(
    model_vgg.predict_generator(
        load_data(test, 1),
        steps=len(test),
        verbose=1
    )
)
prediction_inc = prep_pred(
    model_inc.predict_generator(
        load_data(test, 1),
        steps=len(test),
        verbose=1
    )
)
prediction_rn = pd.DataFrame(
    model_resnet.predict_generator(
        load_data(test, 1),
        steps=len(test),
        verbose=1
    )
)
prediction_resnet = prep_pred(model_lgb.predict(prediction_rn).reshape(-1, 1))
prediction = prediction_vgg + prediction_inc + prediction_resnet
draw_prediction(prediction)

test["target"] = (prediction[0] > 2).astype("i1")
print(test[test["target"] > 0]["EncodedPixels"].head(100))

# Нет облаков - 0.5, MLP - 0.3, CONV/VGG - 0.48, AlexNet - 0.21,
# Inception - 0.5, ResNet - 0.52

dice = test.apply(calc_dice, axis=1, result_type="expand")
print("Keras, VGG16+InceptionV3+ResNet50:", round(dice.mean(), 3))
