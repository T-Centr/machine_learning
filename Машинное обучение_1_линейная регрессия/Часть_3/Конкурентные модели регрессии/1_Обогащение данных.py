"""
Постановка задачи

Заполним отсутствующие значения по погоде интерполяционными данными.

Для точки росы вычтем температуру воздуха, направление ветра разложим на синус
и косинус, для температуры воздуха вычислим первую и вторую производные. Также
введем параметры по праздничным дням, дням недели, месяцам и неделям года.

Посчитаем модель линейной регрессии по первым 20 зданиям и найдем ее точность.
Проверим, какой набор параметров позволяет улучшить точность.

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
from sklearn.linear_model import LinearRegression


"""Загрузка данных, отсечение 20 зданий, объединение и оптимизация"""


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
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2),
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
print(energy.info())


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

weather["wind_direction_rad"] = weather["wind_direction"] * np.pi / 180
weather["wind_direction_sin"] = np.sin(weather["wind_direction_rad"])
weather["wind_direction_cos"] = np.cos(weather["wind_direction_rad"])
weather["air_temperature_diff1"] = weather["air_temperature"].diff()
weather.at[0, "air_temperature_diff1"] = weather.at[1, "air_temperature_diff1"]
weather["air_temperature_diff2"] = weather["air_temperature_diff1"].diff()
weather.at[0, "air_temperature_diff2"] = weather.at[1, "air_temperature_diff2"]


"""Объдинение погодных данных"""

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
del weather
energy = reduce_mem_usage(energy)
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
    energy['is_wday' + str(weekday)] = energy[
        'weekday'].isin([weekday]).astype("int8")
for week in range(1, 54):
    energy['is_w' + str(week)] = energy['week'].isin([week]).astype("int8")
for month in range(1, 13):
    energy['is_m' + str(month)] = energy['month'].isin([month]).astype("int8")


"""Логарифмирование данных"""

# z = A * x + B * y -> log z = A * x + B * y => z = e^Ax * e^By => z = a^x * b^y
energy["meter_reading_log"] = np.log(energy["meter_reading"] + 1)


"""Разделение данных"""

energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0],
    test_size=0.2
)
print(energy_train.info())


"""Линейная регрессия"""

hours = range(0, 24)
buildings = range(0, energy_train["building_id"].max() + 1)
lr_columns = [
    "meter_reading_log",
    "hour",
    "building_id",
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
    "cloud_coverage",
    "air_temperature_diff1",
    "air_temperature_diff2",
    "is_holiday"
]
for wday in range(0, 7):
    lr_columns.append("is_wday" + str(wday))
for week in range(1, 54):
    lr_columns.append("is_w" + str(week))
for month in range(1, 13):
    lr_columns.append("is_m" + str(month))
energy_train_lr = pd.DataFrame(energy_train, columns=lr_columns)
energy_lr = [[]] * len(buildings)
for building in buildings:
    energy_lr[building] = [[]] * len(hours)
    energy_train_b = energy_train_lr[energy_train_lr["building_id"] == building]
    for hour in hours:
        energy_train_bh = energy_train_b[energy_train_b["hour"] == hour]
        y = energy_train_bh["meter_reading_log"]
        x = energy_train_bh.drop(
            labels=[
                "meter_reading_log",
                "hour",
                "building_id"
            ],
            axis=1
        )
        model = LinearRegression(fit_intercept=False).fit(x, y)
        energy_lr[building][hour] = model.coef_
        energy_lr[building][hour] = np.append(
            energy_lr[building][hour],
            model.intercept_
        )
print(energy_lr[0])


"""Расчет качества"""


def calculate_model(x):
    lr = -1
    model = energy_lr[x.building_id][x.hour]
    if len(model) > 0:
        lr = np.sum([x[col] * model[i] for i, col in enumerate(lr_columns[3:])])
        lr += model[len(lr_columns) - 3]
        lr = np.exp(lr)
    if lr < 0:
        lr = 0
    x["meter_reading_lr_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    return x


energy_test = energy_test.apply(
    calculate_model,
    axis=1,
    result_type="expand"
)
energy_test_lr_rmsle = np.sqrt(
    energy_test["meter_reading_lr_q"].sum() / len(energy_test))
print(
    "Качество линейной регрессии, 20 зданий:",
    energy_test_lr_rmsle,
    round(energy_test_lr_rmsle, 1)
)
