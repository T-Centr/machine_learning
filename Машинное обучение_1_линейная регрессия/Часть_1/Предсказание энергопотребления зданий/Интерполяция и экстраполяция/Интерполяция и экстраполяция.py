"""
Постановка задачи.

Построить модель энергопотребления здания по часам. Погоду и характеристики
здания пока не рассматривать.

Данные: http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16, 8


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов

pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов

pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных: Энергопотребление здания"""

energy_0 = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz"
)
# print(energy_0.head())


"""Обогащение данных"""

# Добавим серию с часом суток для построения суточной модели потребления
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour
# print(energy_0.head())


"""Среднее потребление по часам"""

# Выведем среднее и медиану потребления энергии по часам
energy_0_hours = energy_0.groupby("hour")
energy_0_averages = pd.DataFrame(
    {"Среднее": energy_0_hours.mean()["meter_reading"],
     "Медиана": energy_0_hours.median()["meter_reading"]}
)
# energy_0_averages.plot()
# plt.show()


"""Фильтруем метрику"""

# Удаляем нулевые значения из статистики
energy_0_hours_filtered = energy_0[energy_0["meter_reading"] > 0].groupby("hour")
energy_0_averages_filtered = pd.DataFrame(
    {"Среднее": energy_0_hours_filtered.mean()["meter_reading"],
     "Медиана": energy_0_hours_filtered.median()["meter_reading"]}
)
# energy_0_averages_filtered.plot()
# plt.show()


"""Интерполируем данные по часам"""
# Построим модель внутрисуточного потребление энергии по зданию
x = np.arange(0, 24)
y = interp1d(x, energy_0_hours_filtered.median()["meter_reading"], kind="cubic")
xn = np.arange(0, 23.1, 0.1)
yn = y(xn)
plt.plot(x, energy_0_hours_filtered.median()["meter_reading"], 'o', xn, yn, '-')
plt.show()
