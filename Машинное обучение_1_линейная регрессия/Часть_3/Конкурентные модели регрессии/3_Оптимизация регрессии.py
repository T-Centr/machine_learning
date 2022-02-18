"""
Постановка задачи

Рассмотрим несколько моделей линейной регрессии, чтобы выяснить более
оптимальную для первых 20 зданий.

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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import ElasticNet,BayesianRidge


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
energy = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz"
)
energy = energy[(energy["building_id"] < 20)]
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
        "site_id",
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
    energy['is_wday' + str(weekday)] = energy['weekday'].isin(
        [weekday]).astype("int8")
energy["date"] = pd.to_datetime(energy["timestamp"].dt.date)
dates_range = pd.date_range(start='2015-12-31', end='2017-01-01')
us_holidays = calendar().holidays(
    start=dates_range.min(),
    end=dates_range.max()
)
energy['is_holiday'] = energy['date'].isin(us_holidays).astype("int8")
energy["meter_reading_log"] = np.log(energy["meter_reading"] + 1)


"""Разделение данных"""

energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0],
    test_size=0.2
)
print(energy_train.head())


"""Линейная регрессия: по часам"""

from sklearn.metrics import r2_score

hours = range(0, 24)
buildings = range(0, energy_train["building_id"].max() + 1)
lr_columns = ["meter_reading_log", "hour", "building_id", "is_holiday"]
for wday in range(0, 7):
    lr_columns.append("is_wday" + str(wday))
lr_models = {
    "LinearRegression": LinearRegression,
    "Lasso-0.01": Lasso,
    "Lasso-0.1": Lasso,
    "Lasso-1.0": Lasso,
    "Ridge-0.01": Ridge,
    "Ridge-0.1": Ridge,
    "Ridge-1.0": Ridge,
    "ElasticNet-1-1": ElasticNet,
    "ElasticNet-0.1-1": ElasticNet,
    "ElasticNet-1-0.1": ElasticNet,
    "ElasticNet-0.1-0.1": ElasticNet,
    "BayesianRidge": BayesianRidge
}
energy_train_lr = pd.DataFrame(energy_train, columns=lr_columns)

# Линейная регрессия
# 𝑧=𝐴𝑥+𝐵𝑦+𝐶,|𝑧−𝑧0|**2→𝑚𝑖𝑛
#
# Лассо + LARS Лассо
# 1/2𝑛|𝑧−𝑧0|**2+𝑎(|𝐴|+|𝐵|)→𝑚𝑖𝑛
#
# Гребневая регрессия
# |𝑧−𝑧0|**2+𝑎(𝐴**2+𝐵**2)→𝑚𝑖𝑛
#
# ElasticNet: Лассо + Гребневая регрессия
# 1/2𝑛|𝑧−𝑧0|**2+𝛼𝑝|𝐴**2+𝐵**2|+(𝛼−𝑝)(|𝐴|+|𝐵|)/2→𝑚𝑖𝑛

lr_models_scores = {}
for _ in lr_models:
    lr_model = lr_models[_]
    energy_lr_scores = [[]] * len(buildings)
    for building in buildings:
        energy_lr_scores[building] = [0] * len(hours)
        energy_train_b = energy_train_lr[
            energy_train_lr["building_id"] == building
        ]
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
            if _ in ["Ridge-0.1", "Lasso-0.1"]:
                model = lr_model(alpha=.1, fit_intercept=False).fit(x, y)
            elif _ in ["Ridge-0.01", "Lasso-0.01"]:
                model = lr_model(alpha=.01, fit_intercept=False).fit(x, y)
            elif _ == "ElasticNet-1-1":
                model = lr_model(
                    alpha=1,
                    l1_ratio=1,
                    fit_intercept=False
                ).fit(x, y)
            elif _ == "ElasticNet-1-0.1":
                model = lr_model(
                    alpha=1,
                    l1_ratio=.1,
                    fit_intercept=False
                ).fit(x, y)
            elif _ == "ElasticNet-0.1-1":
                model = lr_model(
                    alpha=.1,
                    l1_ratio=1,
                    fit_intercept=False
                ).fit(x, y)
            elif _ == "ElasticNet-0.1-0.1":
                model = lr_model(
                    alpha=.1,
                    l1_ratio=.05,
                    fit_intercept=False
                ).fit(x, y)
            else:
                model = lr_model(fit_intercept=False).fit(x, y)
            energy_lr_scores[building][hour] = r2_score(y, model.predict(x))
    lr_models_scores[_] = np.mean(energy_lr_scores)

print(lr_models_scores)


"""Проверим модели: LinearRegression, Lasso, BayesianRidge"""

energy_lr = [[]] * len(buildings)
energy_lasso = [[]] * len(buildings)
energy_br = [[]] * len(buildings)
for building in buildings:
    energy_train_b = energy_train_lr[energy_train_lr["building_id"] == building]
    energy_lr[building] = [[]] * len(hours)
    energy_lasso[building] = [[]] * len(hours)
    energy_br[building] = [[]] * len(hours)
    for hour in hours:
        energy_train_bh = pd.DataFrame(
            energy_train_b[energy_train_b["hour"] == hour]
        )
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
            energy_lr[building][hour] = np.append(
                [model.coef_], model.intercept_)
            model = Lasso(fit_intercept=False, alpha=.01).fit(x, y)
            energy_lasso[building][hour] = np.append(
                [model.coef_], model.intercept_)
            model = BayesianRidge(fit_intercept=False).fit(x, y)
            energy_br[building][hour] = np.append(
                [model.coef_], model.intercept_)
print(energy_lr[0][0])
print(energy_lasso[0][0])
print(energy_br[0][0])
