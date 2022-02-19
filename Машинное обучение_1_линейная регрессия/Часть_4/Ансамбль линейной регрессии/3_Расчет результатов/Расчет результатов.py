"""
Постановка задачи
Посчитаем модели линейной регрессии для 20 зданий по оптимальному набору
параметров: метеорологические данные, дни недели, недели года, месяцы и
праздники по всему набору данных.

Загрузим данные решения, посчитаем значение энергопотребления для требуемых дат
для тех зданий, которые посчитаны в модели, и выгрузим результат в виде файла.

Данные:

http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz
http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz
http://video.ittensive.com/machine-learning/ashrae/test.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_test.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


"""Загрузка данных 20 зданий из HDF5"""

energy = pd.read_hdf('energy.0-20.ready.h5', "energy")
print(energy.info())


"""Загрузка данных для расчета, оптимизация памяти"""


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
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2),
          'Мб (минус', round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df

# Все результаты в оперативной памяти занимают порядка 8 Гб. Для оптимизации
# потребления памяти сначала рассчитаем результаты только для первыx 20 зданий,
# а затем присоединим к ним остальные, заполненные нулями.

buildings = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz",
    usecols=["site_id", "building_id"]
)
weather = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/weather_test.csv.gz"
)
weather = weather[weather["site_id"] == 0]
weather = weather.drop(columns=["wind_direction"], axis=1)
results = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/test.csv.gz"
)
results = results[(results["building_id"] < 20) & (results["meter"] == 0)]
results = pd.merge(
    left=results,
    right=buildings,
    how="left",
    left_on="building_id",
    right_on="building_id"
)
del buildings
results = results.drop(columns=["meter"], axis=1)
print(results.info())


"""Интерполяция значений и обогащение погодных данных: только для 1 города"""

interpolate_columns = [
    "air_temperature",
    "dew_temperature",
    "cloud_coverage",
    "wind_speed",
    "sea_level_pressure"
]
for col in interpolate_columns:
    weather[col] = weather[col].interpolate(limit_direction='both', kind='cubic')
weather["air_temperature_diff1"] = weather["air_temperature"].diff()
weather.at[0, "air_temperature_diff1"] = weather.at[1, "air_temperature_diff1"]
weather["air_temperature_diff2"] = weather["air_temperature_diff1"].diff()
weather.at[0, "air_temperature_diff2"] = weather.at[1, "air_temperature_diff2"]


"""Объединение данных по погоде"""

results = results.set_index(["timestamp", "site_id"])
weather = weather.set_index(["timestamp", "site_id"])
results = pd.merge(
    left=results,
    right=weather,
    how="left",
    left_index=True,
    right_index=True
)
results.reset_index(inplace=True)
results = results.drop(columns=["site_id"], axis=1)
del weather
results = reduce_mem_usage(results)
print(results.info())


"""Обогащение данных по дате"""

results["hour"] = results["timestamp"].dt.hour.astype("int8")
results["weekday"] = results["timestamp"].dt.weekday.astype("int8")
results["week"] = results["timestamp"].dt.week.astype("int8")
results["month"] = results["timestamp"].dt.month.astype("int8")
results["date"] = pd.to_datetime(energy["timestamp"].dt.date)
dates_range = pd.date_range(start='2016-12-31', end='2018-06-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
results['is_holiday'] = results['date'].isin(us_holidays).astype("int8")
for weekday in range(0, 7):
    results['is_wday' + str(weekday)] = results['weekday'].isin(
        [weekday]).astype("int8")
for week in range(1, 54):
    results['is_w' + str(week)] = results['week'].isin([week]).astype("int8")
for month in range(1, 13):
    results['is_m' + str(month)] = results['month'].isin([month]).astype("int8")


"""Линейная регрессия"""

hours = range(0, 24)
buildings = range(0, energy["building_id"].max() + 1)
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
energy_train_lr = pd.DataFrame(energy, columns=lr_columns)
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


"""Расчет финальных показателей, только энергопотребление,
только 20 первых зданий"""


def calculate_model(x):
    lr = -1
    model = energy_lr[x.building_id][x.hour]
    if len(model) > 0:
        lr = np.sum([x[col] * model[i] for i, col in enumerate(lr_columns[3:])])
        lr += model[len(lr_columns) - 3]
        lr = np.exp(lr)
    if lr < 0 or lr != lr or lr * lr == lr:
        lr = 0
    x["meter_reading"] = lr
    if x["row_id"] % 1000000 == 0:
        print("Готово", x["row_id"])
    return x


results = results.apply(calculate_model, axis=1, result_type="expand")


"""Усечение данных до требуемого формата: row_id, meter_reading"""

results_ready = pd.DataFrame(results, columns=["row_id", "meter_reading"])


"""Загрузка всех данных для заполнения их нулями"""

results = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/test.csv.gz",
    usecols=["row_id"]
)
results = pd.merge(
    left=results,
    right=results_ready,
    how="left",
    left_on="row_id",
    right_on="row_id"
)
results.fillna(value=0, inplace=True)
print(results.info())


"""Выгрузка результатов в CSV файл"""

# Итоговый файл занимает около 1 Гб
results.to_csv("submission.csv", index=False)


"""Освобождение памяти"""

del energy
del results
del results_ready
