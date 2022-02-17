"""
Постановка задачи

Заполним отсутствующие значения по погоде интерполяционными данными.

Посчитаем модель линейной регрессии по первому зданию и найдем ее точность.

Данные:

http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz
http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


"""Загрузка данных"""

buildings = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz"
)
weather = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz"
)
energy = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz"
)


"""Отсечение здания 0 и отсутствующих значений"""

energy = energy[(energy["building_id"] == 0)]


"""Объединение данных"""

energy = pd.merge(
    left=energy, right=buildings, how="left",
    left_on="building_id", right_on="building_id"
)
energy = energy.set_index(["timestamp", "site_id"])
weather = weather.set_index(["timestamp", "site_id"])
energy = pd.merge(
    left=energy, right=weather, how="left",
    left_index=True, right_index=True
)
energy.reset_index(inplace=True)
energy = energy.drop(columns=["meter", "site_id", "floor_count"], axis=1)
del buildings
del weather
# print(energy.info())


"""Оптимизация памяти"""


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[: 5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[: 3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        elif col == "timestamp":
            df[col] = pd.to_datetime(df[col])
        elif str(col_type)[: 8] != "datetime":
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        'Потребление памяти меньше на', round(start_mem - end_mem, 2),
        'Мб (минус', round(100 * (start_mem - end_mem) / start_mem, 1), '%)'
    )
    return df


energy = reduce_mem_usage(energy)


"""Интерполяция данных"""

energy["precip_depth_1_hr"] = energy["precip_depth_1_hr"].apply(
    lambda x: x if x > 0 else 0)
interpolate_columns = [
    "air_temperature", "dew_temperature", "cloud_coverage", "wind_speed",
    "precip_depth_1_hr", "sea_level_pressure"
]
for col in interpolate_columns:
    energy[col] = energy[col].interpolate(limit_direction='both', kind='cubic')


"""Проверка качества интерполяции"""

pd.set_option('use_inf_as_na', True)
for col in interpolate_columns:
    print(col, "Inf+NaN:", energy[col].isnull().sum())


"""Разделение данных"""

energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0], test_size=0.2
)
# print(energy_train.head())


"""Линейная регрессия"""

regression_columns = [
    "meter_reading", "air_temperature", "dew_temperature", "cloud_coverage",
    "wind_speed", "precip_depth_1_hr", "sea_level_pressure"
]

energy_train_lr = pd.DataFrame(energy_train, columns=regression_columns)
y = energy_train_lr["meter_reading"]
x = energy_train_lr.drop(labels=["meter_reading"], axis=1)
model = LinearRegression().fit(x, y)
# print(model.coef_, model.intercept_)


"""Предсказание и оценка модели"""


def calculate_model(x):
    lr = np.sum(
        [x[col] * model.coef_[i] for i, col in enumerate(regression_columns[1:])]
    )
    lr += model.intercept_
    x["meter_reading_lr_q"] = (np.log(1 + x.meter_reading) - np.log(1 + lr)) ** 2
    return x


energy_test = energy_test.apply(calculate_model, axis=1, result_type="expand")
energy_test_lr_rmsle = np.sqrt(
    energy_test["meter_reading_lr_q"].sum() / len(energy_test)
)
print("Качество линейной регрессии:", energy_test_lr_rmsle,
      round(energy_test_lr_rmsle, 1))
