"""
Постановка задачи

Построить модель линейной регрессии энергопотребления здания, используя
температуру воздуха (air_temperature) и влажность (dew_temperature).

Рассчитать качество построенной модели по проверочным данным.

Данные:

http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz
http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов

pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов

pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных"""

buildings = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz"
)
weather = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz"
)
energy_0 = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz"
)
print(energy_0.info())


"""Объединение данных и фильтрация"""

energy_0 = pd.merge(
    left=energy_0, right=buildings, how="left",
    left_on="building_id", right_on="building_id"
)
energy_0.set_index(["timestamp", "site_id"], inplace=True)
weather.set_index(["timestamp", "site_id"], inplace=True)
energy_0 = pd.merge(
    left=energy_0, right=weather, how="left", left_index=True, right_index=True
)
energy_0.reset_index(inplace=True)
energy_0 = energy_0[energy_0["meter_reading"] > 0]
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour
print(energy_0.head())


"""Добавление часа в данные"""

energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour


"""Разделение данных на обучение и проверку"""

energy_0_train, energy_0_test = train_test_split(energy_0, test_size=0.2)
print(energy_0_train.head())


"""Модель линейной регрессии и среднее"""

# meter_reading = A * air_temperature + B * dew_temperature + C
# Дополнительно вычислим среднее по часам, чтобы сравнить линейную регрессию с
# более простой моделью
energy_0_train_averages = energy_0_train.groupby("hour").mean()["meter_reading"]

energy_0_train_lr = pd.DataFrame(
    energy_0_train,
    columns=["meter_reading", "air_temperature", "dew_temperature"]
)
y = energy_0_train_lr["meter_reading"]
x = energy_0_train_lr.drop(labels=["meter_reading"], axis=1)
model = LinearRegression().fit(x, y)
print(model.coef_, model.intercept_)


"""Оценка модели"""


def calculate_model(x):
    meter_reading_log = np.log(x.meter_reading + 1)
    meter_reading_mean = np.log(energy_0_train_averages[x.hour] + 1)
    meter_reading_lr = np.log(
        1 + x.air_temperature * model.coef_[0] +
        x.dew_temperature * model.coef_[1] + model.intercept_
    )
    x["meter_reading_lr_q"] = (meter_reading_log - meter_reading_lr) ** 2
    x["meter_reading_mean_q"] = (meter_reading_log - meter_reading_mean) ** 2
    return x


energy_0_test = energy_0_test.apply(
    calculate_model, axis=1, result_type="expand"
)
energy_0_test_lr_rmsle = np.sqrt(
    energy_0_test["meter_reading_lr_q"].sum() / len(energy_0_test)
)
energy_0_test_mean_rmsle = np.sqrt(
    energy_0_test["meter_reading_mean_q"].sum() / len(energy_0_test)
)
print("Качество среднего:", energy_0_test_mean_rmsle)
print("Качество линейной регрессии:", energy_0_test_lr_rmsle)
