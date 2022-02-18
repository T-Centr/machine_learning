"""
Постановка задачи

Получите данные по энергопотреблению первых 20 зданий (building_id от 0 до 19).

Заполните отсутствующие значения по погоде интерполяционными данными.

Разделите данные на обучающие/проверочные в пропорции 80/20.

Постройте (1) первый набор моделей линейной регрессии по часам для каждого из
первых 20 зданий по следующим параметрам: air_temperature, dew_temperature,
cloud_coverage, wind_speed, sea_level_pressure.

Постройте для этих же 20 зданий (2) второй набор моделей линейной регрессии по
часам по параметрам: дни недели и праздники (is_holiday). Требуется построить
еще 480 моделей.

Используйте логарифм целевого показателя (meter_reading_log).

Найдите, какой набор моделей работает точнее для первого здания
(building_id = 0): у какого набора будет меньше метрика качества?

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
        elif str(col_type)[:8] != "datetime":
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
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


"""Проведем интерполяцию данных"""

interpolate_columns = [
    "air_temperature",
    "dew_temperature",
    "cloud_coverage",
    "wind_speed",
    "sea_level_pressure"
]
for col in interpolate_columns:
    weather[col] = weather[col].interpolate(limit_direction='both', kind='cubic')


"""Объединим погодные данные с основными"""

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
        "floor_count",
        "primary_use",
        "year_built"
    ],
    axis=1
)
del weather
energy = reduce_mem_usage(energy)
print(energy.info())


"""Добавим час, день недели и праздники в данные"""

energy["hour"] = energy["timestamp"].dt.hour.astype("int8")
energy["weekday"] = energy["timestamp"].dt.weekday.astype("int8")
for weekday in range(0, 7):
    energy['is_wday' + str(weekday)] = energy['weekday'].isin(
        [weekday]).astype("int8")
dates_range = pd.date_range(start='2015-12-31', end='2017-01-01')
us_holidays = calendar().holidays(
    start=dates_range.min(),
    end=dates_range.max()
)
energy["date"] = pd.to_datetime(energy["timestamp"].dt.date)
energy['is_holiday'] = (energy['date'].isin(us_holidays)).astype("int8")
energy["meter_reading_log"] = np.log(energy["meter_reading"] + 1)


"""Разделим данные"""

energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0],
    test_size=0.2
)
print(energy_train.head())


"""Построим массив моделей линейной регрессии - по зданию и часу,
всего 480 моделей"""

hours = range(0, 24)
buildings = range(0, energy_train["building_id"].max() + 1)
lr_columns = [
    "meter_reading_log",
    "hour", "building_id",
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
    "cloud_coverage"
]
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
            energy_lr[building][hour], model.intercept_
        )
print(energy_lr)


"""Рассчитаем качество построенных моделей"""


def calculate_model(x):
    model = energy_lr[x.building_id][x.hour]
    lr = np.sum([x[col] * model[i] for i, col in enumerate(lr_columns[3:])])
    lr += model[len(lr_columns) - 3]
    lr = np.exp(lr)
    x["meter_reading_lr_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    return x


energy_test = energy_test.apply(calculate_model, axis=1, result_type="expand")


"""Построим модели по дате"""

lr_columns = ["meter_reading_log", "hour", "building_id", "is_holiday"]
for wday in range(0, 7):
    lr_columns.append("is_wday" + str(wday))
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
            energy_lr[building][hour], model.intercept_
        )


"""Рассчитаем качество моделей по дате"""


def calculate_model_is(x):
    model = energy_lr[x.building_id][x.hour]
    if model[0] != 0:
        lr = np.sum([x[col] * model[i] for i, col in enumerate(lr_columns[3:])])
        lr += model[len(lr_columns) - 3]
        lr = np.exp(lr)
    if lr < 0:
        lr = 0
    x["meter_reading_is_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    return x


energy_test = energy_test.apply(calculate_model_is, axis=1, result_type="expand")

"""Найдем, какой набор моделей оказался лучше для 1 здания
(метрика качества меньше)"""

energy_test_0_sum = energy_test[energy_test["building_id"] == 0].sum()
if energy_test_0_sum["meter_reading_is_q"] > energy_test_0_sum["meter_reading_lr_q"]:
    print("Лучше линейная регрессия по погоде")
else:
    print("Лучше линейная регрессия по дате")
