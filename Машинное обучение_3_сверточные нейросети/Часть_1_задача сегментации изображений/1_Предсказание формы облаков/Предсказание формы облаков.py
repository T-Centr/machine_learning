"""
Постановка задачи

Даны изображения облаков, сделанные со спутника. Часть из этих изображений уже
размечена учеными на содержание облаков определенной формы - это "цветок",
"рыба", "сахар" и "гравий". Для оставшихся фотографий необходимо определить как
класс области облаков, так и найти границу этой определенной области.

Загрузим данные уменьшенных изображений и проведем исследовательский анализ
данных для них. Найдем все взаимосвязи, которые помогут в построении модели
классификации.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import requests
from PIL import Image
from io import BytesIO


"""Загрузка обучающих данных"""

# Формат данных: название изображения+формы облаков, кодированная маска облаков
# Кодирование маски выполняется в RLE: пары значений содержат
# начало - порядковый номер пиксела в изображении - и длину - число пикселей
# после начала, которые нужно включить в маску.
# Например, кодировка 1 3 10 5 - 1-начало, 3-длина, 10-начало, 5-длина, будет
# означать последовательность пикселей 1, 2, 3, 10, 11, 12, 13, 14, 15.
# Пиксели нумеруются из левого верхнего угла сверху вниз слева направо. Первый
# пиксел имеет координаты (1,1), второй пиксел имеет координаты (1,2) и т.д.

train = pd.read_csv(
    "https://video.ittensive.com/machine-learning/clouds/train.csv.gz"
)
print(train.info())
print(train.head())


"""Очистка данных"""

# Разделим файлы и названия облаков на изображениях

train["Image"] = train["Image_Label"].str.split("_").str[0]
train["Label"] = train["Image_Label"].str.split("_").str[1]
train.drop(labels=["Image_Label"], axis=1, inplace=True)
print(train.head())


"""Размеченные данные"""

sizes = [train.EncodedPixels.count(), len(train) - train.EncodedPixels.count()]
explode = (0, 0.1)
fig, ax = plt.subplots(figsize=(16, 8))
ax.pie(
    sizes,
    explode=explode,
    labels=["Размеченные", "Пустые"],
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
ax.set_title("Размеченные и пустые маски облаков")
plt.show()


"""Формы облаков"""

fish = train[train["Label"] == "Fish"].EncodedPixels.count()
flower = train[train["Label"] == "Flower"].EncodedPixels.count()
gravel = train[train["Label"] == "Gravel"].EncodedPixels.count()
sugar = train[train["Label"] == "Sugar"].EncodedPixels.count()
sizes = [fish, flower, gravel, sugar]

fig, ax = plt.subplots(figsize=(16, 8))
ax.pie(
    sizes,
    labels=["Fish", "Flower", "Gravel", "Sugar"],
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
ax.set_title("Типы облаков")
plt.show()


"""Количество разных облаков на изображениях"""

_, area = plt.subplots(figsize=(6, 6))
train.groupby("Image")['EncodedPixels'].count().hist(ax=area)
ax.set_title("Число разных облаков")
plt.show()


"""Корреляция между формами облаков"""

labels = train["Label"].unique()
for label in labels:
    train["Label_" + label] = (
            (train["EncodedPixels"].notnull()) & (train["Label"] == label)
    ).astype("int8")
train_corr = train.groupby("Image")[
    "Label_Fish",
    "Label_Flower",
    "Label_Gravel",
    "Label_Sugar"
].sum()
corrs = np.corrcoef(train_corr.values.T)
sns.set(rc={'font.size':20, 'figure.figsize': (12, 12)})
sns.heatmap(
    corrs,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    yticklabels=labels,
    xticklabels=labels
).set_title("Корреляция", fontsize=30)
plt.show()


"""Области облаков на изображении"""

# Отметим области, в которых присутствуют облака выделенного типа

image_x = 2100
image_y = 1400


def mask_rate(a, x, y):
    b = a // 1400 + 0.0
    return np.round(
        x * (b * x // 2100) + y * (a % 1400) // 1400
    ).astype("uint32")


def calc_mask(px, x=image_x, y=image_y):
    p = np.array([int(n) for n in px.split(" ")]).reshape(-1, 2)
    mask = np.zeros(x * y, dtype="uint8")
    for i, l in p:
        mask[mask_rate(i, x, y) - 1: mask_rate(l + i, x, y)] = 1
    return mask.reshape(y, x).transpose()

# Загрузим изображение и преобразуем его в PNG для вывода через matplotlib

img = requests.get(
    "https://video.ittensive.com/machine-learning/clouds/"
    "train_images/" + train["Image"].values[0]
)
img = Image.open(BytesIO(img.content))
image_png = BytesIO()
img.save(image_png, format="PNG")
image_png.seek(0)
plt.figure(figsize=(21, 14))
plt.axis("off")
plt.imshow(mpimg.imread(image_png))
plt.show()

img = np.array(Image.open(image_png))
mask = calc_mask(train["EncodedPixels"].values[0])
segmap = SegmentationMapsOnImage(mask, mask.shape)
fig, area = plt.subplots(figsize=(21, 14))
area.axis("off")
plt.title("Fish")
area.imshow(np.array(segmap.draw_on_image(img)).reshape(img.shape))
plt.show()
