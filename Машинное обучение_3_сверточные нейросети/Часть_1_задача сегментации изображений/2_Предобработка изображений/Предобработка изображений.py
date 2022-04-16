"""
Постановка задачи

Загрузим данные изображений, построим на них размеченные области, сформируем
массив серых пикселей из изображений.

Выгрузим массив пикселей в HDF5 для дальнейшей работы.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz
https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import numpy as np
import pandas as pd
from PIL import Image


"""Загрузка данных"""

train = pd.read_csv(
    "https://video.ittensive.com/machine-learning/clouds/train.csv.gz"
)
print(train.head())


"""Очистка данных"""

# Отделим категории и область каждой формы облаков

train["Image"] = train["Image_Label"].str.split("_").str[0]
train["Label"] = train["Image_Label"].str.split("_").str[1]
train.drop(labels=["Image_Label"], axis=1, inplace=True)
print(train.head(20))


data = pd.DataFrame({"Image": train["Image"].unique()})
for l in train["Label"].unique():
    data[l] = pd.Series(train[train["Label"] == l]["EncodedPixels"].values)
print(data.head())


"""Обработка изображений"""

# Предварительно загрузим весь архив с изображениями и распакуем его в
# train_images_small
# Пример: 0011165.jpg, размеры 525 * 350 = 183750 пикселей
# Каждое изображение приведем к серой палитре, результат загрузим в фрейм данных

imgdata = np.array([np.zeros(183750, dtype="uint8")] * len(data))
for i, img in enumerate(data["Image"].unique()):
    imgdata[i] = np.array(
        Image.open("train_images_small/" + img).convert("L"),
        dtype="uint8"
    ).reshape(1, -1)[0]
imgdata = pd.DataFrame(imgdata)
print(imgdata.head())

for column in data.columns:
    imgdata[column] = data[column]
del data
print(imgdata.head())


"""Сохраняем данные в HDF5"""

# Потребуется до 3 Гб оперативной памяти

imgdata.to_hdf(
    "clouds.data.h5",
    "data",
    format="fixed",
    # compression="gzip",
    complevel=9,
    mode="w"
)
