"""
Постановка задачи

Используйте обычный или расширенный (с искажениями) набор уменьшенных
изображений облаков из предыдущего урока.

Загрузите данные и разделите их на обучающие и проверочные в соотношении 80/20.

Используйте Keras для построения 4 нейросетей с линейным, сверточными слоями и
слоями подвыборки - для каждого типа облаков в отдельности. Конфигурация
произвольная, не менее 2 сверточных слоев и не менее 1 слоя подвыборки.

Обучите модели, используя последовательную загрузку данных. Объедините
предсказания моделей для проверочной выборки и проведите оценку качества
предсказания по коэффициенту сходства.

Можно использовать уже обученные модели нейросетей на расширенной выборке
уменьшенных изображений, загрузив их с помощью keras.models.load_model.
В этом случае проверку можно провести на всех изображениях.

Данные:

https://video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)
https://video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz
(212 Мб)

Модели:

https://video.ittensive.com/machine-learning/clouds/train_tiny.fish.h5
https://video.ittensive.com/machine-learning/clouds/train_tiny.flower.h5
https://video.ittensive.com/machine-learning/clouds/train_tiny.gravel.h5
https://video.ittensive.com/machine-learning/clouds/train_tiny.sugar.h5

Соревнование: https://www.kaggle.com/c/understanding_cloud_organization/

© ITtensive, 2020
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from skimage import io
import os


"""Используемые функции"""

filesDir = "C:/PythonProject/ITtensive/machine_learning/train_images_small_tiny"
labels = ["Fish", "Flower", "Gravel", "Sugar"]
image_x = 210
image_y = 140
image_ch = 1


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
data["target"] = 0
data.drop(labels=["Image_Label"], axis=1, inplace=True)
print(data.head())


"""Разделение данных"""

# Заведем отдельные наборы данных для разных типов облаков, чтобы обработать их
# раздельно. Будем проверять точность моделей на всей выборке.

df = {}
for label in labels:
    df[label] = pd.DataFrame(data[data["Label"] == label])
del data


"""Загрузка моделей"""

# Загрузим рассчитанные модели для каждого типа облаков

models = {}
for label in labels:
    models[label] = keras.models.load_model(
        "train_tiny." + label.lower() + ".h5"
    )
    models[label].summary()


"""Предсказание значений"""

prediction = [[]] * len(labels)
for i, label in enumerate(labels):
    print(label)
    prediction[i] = models[label].predict_generator(
        load_data(df[label], 1),
        steps=len(df[label]),
        verbose=1
    )

fig = plt.figure(figsize=(16, 16))
for i in range(len(labels)):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.hist(prediction[i])
    ax.legend([labels[i]])
plt.show()

for i, label in enumerate(labels):
    print(label)
    df[label]["target"] = np.round(prediction[i] >= 1).astype("int8")
    print(
        df[label][df[label]["target"] > 0][["EncodedPixels", "target"]].head()
    )


"""Оценка по Дайсу"""

# Пока будем считать, что при определении типа облака на изображении, оно
# целиком размещено на фотографии: т.е. область облака - это все изображение.

# Нет облаков - 0.5, MLP - 0.3, CONV3-32x2,POOL2 - 0.48

dice = 0
for i, label in enumerate(labels):
    dice += df[label].apply(calc_dice, axis=1, result_type="expand").mean()
    print(label, dice / (i + 1))
print("Keras, (CONV3-32x2,POOL2,CONV3-32x2,POOL2):", round(dice/len(labels), 3))
