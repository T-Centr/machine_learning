"""
Постановка задачи

Подготовим данные для построения модели: получим, объединим, оптимизируем и
обогатим данные.

Сохраним готовые данные в нескольких форматах: CSV, HDF5

Данные:

http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz
http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from sklearn.model_selection import train_test_split
import os


"""Загрузка данных, отсечение 20 зданий, объединение и оптимизация"""


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
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
        elif str(col_type)[:8] != "datetime":
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        'Потребление памяти меньше на', round(start_mem - end_mem, 2),
        'Мб (минус', round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df


buildings = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz"
)
weather = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz"
)
weather = weather[weather["site_id"] == 0]
energy = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz"
)
energy = energy[energy["building_id"] < 20]
energy = pd.merge(
    left=energy,
    right=buildings,
    how="left",
    left_on="building_id",
    right_on="building_id"
)
del buildings


"""Интерполяция значений"""

weather["precip_depth_1_hr"] = weather["precip_depth_1_hr"].apply(
    lambda x: x if x > 0 else 0)
interpolate_columns = [
    "air_temperature",
    "dew_temperature",
    "cloud_coverage",
    "wind_speed",
    "wind_direction",
    "precip_depth_1_hr",
    "sea_level_pressure"
]
for col in interpolate_columns:
    weather[col] = weather[col].interpolate(limit_direction='both', kind='cubic')


"""Обогащение данных: погода"""

weather["air_temperature_diff1"] = weather["air_temperature"].diff()
weather.at[0, "air_temperature_diff1"] = weather.at[1, "air_temperature_diff1"]
weather["air_temperature_diff2"] = weather["air_temperature_diff1"].diff()
weather.at[0, "air_temperature_diff2"] = weather.at[1, "air_temperature_diff2"]


"""Объединение погодных данных"""

energy = energy.set_index(["timestamp", "site_id"])
weather = weather.set_index(["timestamp", "site_id"])
energy = pd.merge(
    left=energy,
    right=weather,
    how="left",
    left_index=True,
    right_index=True
)
energy.reset_index(inplace=True)
energy = energy.drop(
    columns=[
        "meter",
        "site_id",
        "year_built",
        "square_feet",
        "floor_count"
    ],
    axis=1
)
energy = reduce_mem_usage(energy)
del weather
print(energy.info())


"""Обогащение данных: дата"""

energy["hour"] = energy["timestamp"].dt.hour.astype("int8")
energy["weekday"] = energy["timestamp"].dt.weekday.astype("int8")
energy["week"] = energy["timestamp"].dt.week.astype("int8")
energy["month"] = energy["timestamp"].dt.month.astype("int8")
energy["date"] = pd.to_datetime(energy["timestamp"].dt.date)
dates_range = pd.date_range(start='2015-12-31', end='2017-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
energy['is_holiday'] = energy['date'].isin(us_holidays).astype("int8")
for weekday in range(0, 7):
    energy['is_wday' + str(weekday)] = energy['weekday'].isin(
        [weekday]).astype("int8")
for week in range(1, 54):
    energy['is_w' + str(week)] = energy['week'].isin([week]).astype("int8")
for month in range(1, 13):
    energy['is_m' + str(month)] = energy['month'].isin([month]).astype("int8")


"""Логарифмирование данных"""

# z = A * x + B * y -> log z = A * x + B * y => z = e^Ax * e^By => z = a^x * b^y
energy["meter_reading_log"] = np.log(energy["meter_reading"] + 1)


"""Экспорт данных в CSV и HDF5"""

print(energy.info())
energy.to_csv("energy.0-20.ready.csv.gz", index=False)

energy = pd.read_csv("energy.0-20.ready.csv.gz")
print(energy.info())


"""Экспорт данных в HDF5"""

# Экспорт данных в HDF5
# HDF5: / ->
#
# Группа (+ метаданные)
# Набор данных

energy = reduce_mem_usage(energy)
energy.to_hdf(
    'energy.0-20.ready.h5',
    "energy",
    format='table',
    # compression="gzip",
    complevel=9,
    mode="w"
)
print("CSV:", os.path.getsize(os.getcwd() + '\energy.0-20.ready.csv.gz'))
print("HDF5:", os.path.getsize(os.getcwd() + '\energy.0-20.ready.h5'))

energy = pd.read_hdf('energy.0-20.ready.h5', "energy")
print(energy.info())


"""Разделение данных и экспорт в HDF5"""

energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0], test_size=0.2)
print(energy_train.head())

pd.set_option('io.hdf.default_format', 'table')
store = pd.HDFStore(
    'energy.0-20.ready.split.h5',
    complevel=9,
    complib='zlib',
    mode="w"
)
store["energy_train"] = energy_train
store["energy_test"] = energy_test
store.put(
    "metadata", pd.Series(["Набор обогащенных тестовых данных по 20 зданиям"]))
store.close()
print("HDF5:", os.path.getsize(os.getcwd() + '\energy.0-20.ready.split.h5'))

# Для хранения атрибутов наборов данных также можно использовать
# store.get_storer('energy_train').attrs.my_attr


"""Чтение из HDF5"""

store = pd.HDFStore('energy.0-20.ready.split.h5')
energy_test = store.get("energy_test")[:]
energy_train = store.get("energy_train")[:]
metadata = store.get("metadata")[:]
store.close()
print(metadata[0])
print(energy_train.head())
