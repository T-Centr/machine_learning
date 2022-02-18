"""
Постановка задачи

Посчитаем модель линейной регрессии по первым 100 зданиям и найдем ее точность,
используя в качестве параметров только дни недели и праздники, применяя
fit_intercept=False и логарифмируя целевой показатель.

Для вычисления отсутствующих или некорректных данных построим модели по всем
зданиям одного типа в одном городе и во всех городах.

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


"""Загрузка данных, отсечение 100 зданий, объединение и оптимизация"""


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
energy = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz"
)
energy = energy[(energy["building_id"] < 100)]
energy = pd.merge(
    left=energy,
    right=buildings,
    how="left",
    left_on="building_id",
    right_on="building_id"
)
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
        "year_built",
        "square_feet",
        "floor_count"
    ],
    axis=1
)
del buildings
del weather
energy = reduce_mem_usage(energy)
print(energy.info())


"""Обогащение данных: час, дни недели, праздники, логарифм"""

energy["hour"] = energy["timestamp"].dt.hour.astype("int8")
energy["weekday"] = energy["timestamp"].dt.weekday.astype("int8")
for weekday in range(0, 7):
    energy['is_wday' + str(weekday)] = energy[
        'weekday'].isin([weekday]).astype("int8")
energy["date"] = pd.to_datetime(energy["timestamp"].dt.date)
dates_range = pd.date_range(start='2015-12-31', end='2017-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
energy['is_holiday'] = energy['date'].isin(us_holidays).astype("int8")
energy["meter_reading_log"] = np.log(energy["meter_reading"] + 1)


"""Разделение данных"""

energy_train, energy_test = train_test_split(
    energy[(energy["meter_reading"] > 0)],
    test_size=0.2
)
print(energy_train.info())


"""Линейная регрессия: по часам"""

hours = range(0, 24)
buildings = range(0, energy_train["building_id"].max() + 1)
lr_columns = ["meter_reading_log", "hour", "building_id", "is_holiday"]
for wday in range(0, 7):
    lr_columns.append("is_wday" + str(wday))
energy_train_lr = pd.DataFrame(energy_train, columns=lr_columns)
energy_lr = [[]] * len(buildings)
for building in buildings:
    energy_lr[building] = [[]] * len(hours)
    energy_train_b = energy_train_lr[energy_train_lr["building_id"] == building]
    for hour in hours:
        energy_lr[building].append([0] * (len(lr_columns) - 3))
        energy_train_bh = pd.DataFrame(
            energy_train_b[energy_train_b["hour"] == hour])
        y = energy_train_bh["meter_reading_log"]
        if len(y) > 0:
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
                energy_lr[building][hour], model.intercept_)
print(energy_lr[0])


"""Линейная регрессия: по типам зданий"""

sites = range(0, energy["site_id"].max() + 1)
primary_uses = energy["primary_use"].unique()
lr_columns_use = [
    "meter_reading_log",
    "hour",
    "building_id",
    "is_holiday",
    "primary_use",
    "site_id"
]
for wday in range(0, 7):
    lr_columns_use.append("is_wday" + str(wday))
energy_lr_use = {}
energy_lr_use_site = {}
energy_train_lr = pd.DataFrame(energy_train, columns=lr_columns_use)
for primary_use in primary_uses:
    energy_train_u = energy_train_lr[
        energy_train_lr["primary_use"] == primary_use]
    if len(energy_train_u) > 0:
        energy_lr_use_site[primary_use] = [[]] * len(sites)
        for site in sites:
            energy_lr_use_site[primary_use][site] = [[]] * len(hours)
            energy_train_us = energy_train_u[energy_train_u["site_id"] == site]
            if len(energy_train_us) > 0:
                for hour in hours:
                    energy_train_uth = energy_train_us[
                        energy_train_us["hour"] == hour]
                    y = energy_train_uth["meter_reading_log"]
                    if len(y) > 0:
                        x = energy_train_uth.drop(
                            labels=[
                                "meter_reading_log",
                                "hour",
                                "building_id",
                                "site_id",
                                "primary_use"
                            ],
                            axis=1
                        )
                        model = LinearRegression(fit_intercept=False).fit(x, y)
                        energy_lr_use_site[primary_use][site][hour] = model.coef_
                        energy_lr_use_site[primary_use][site][hour] = np.append(
                            energy_lr_use_site[primary_use][site][hour],
                            model.intercept_
                        )
        energy_lr_use[primary_use] = [[]] * len(hours)
        for hour in hours:
            energy_train_th = energy_train_u[energy_train_u["hour"] == hour]
            y = energy_train_th["meter_reading_log"]
            if len(y) > 0:
                x = energy_train_th.drop(
                    labels=[
                        "meter_reading_log",
                        "hour",
                        "building_id",
                        "site_id",
                        "primary_use"
                    ],
                    axis=1
                )
                model = LinearRegression(fit_intercept=False).fit(x, y)
                energy_lr_use[primary_use][hour] = model.coef_
                energy_lr_use[primary_use][hour] = np.append(
                    energy_lr_use[primary_use][hour], model.intercept_)
print(energy_lr_use_site)


"""Расчет качества"""

# Используем индивидуальные модели здания, иначе общую модель по всем зданиям
# данного типа в городе, иначе общую модель по всем зданиям такого типа
# (по всем городам)


def calculate_model(x):
    lr = - 1
    model = energy_lr[x.building_id][x.hour]
    if len(model) == 0:
        model = energy_lr_use_site[x.primary_use][x.site_id][x.hour]
    if len(model) == 0:
        model = energy_lr_use[x.primary_use][x.hour]
    if len(model) > 0:
        lr = np.sum([x[col] * model[i] for i,col in enumerate(lr_columns[3:])])
        lr += model[len(lr_columns) - 3]
        lr = np.exp(lr)
    if lr < 0:
        lr = 0
    x["meter_reading_lr_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    return x


energy_test = energy_test.apply(calculate_model, axis=1, result_type="expand")
energy_test_lr_rmsle = np.sqrt(
    energy_test["meter_reading_lr_q"].sum() / len(energy_test)
)
print("Качество линейной регрессии, 100 зданий:",
      energy_test_lr_rmsle, round(energy_test_lr_rmsle, 1))
