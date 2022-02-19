"""
Постановка задачи

Загрузим подготовленные данные по энергопотреблению первых 20 зданий
(building_id от 0 до 19).

Соберем два набора моделей: по дате (праздники, дни недели и т.д.) и по погоде.

Проведем 10 разбиений данных на обучающие/проверочные и выявим оптимальные веса
моделей для каждого часа для каждого здания.

Вычислим оптимизированную метрику качества для ансамбля моделей.

http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz
http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz
http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


"""Загрузка данных 20 зданий из HDF5"""

energy = pd.read_hdf('energy.0-20.ready.h5', "energy")
print(energy.info())


"""Обозначим набор параметров для каждой модели"""

lr_weather_columns = [
    "meter_reading_log",
    "hour",
    "building_id",
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
    "air_temperature_diff1",
    "air_temperature_diff2",
    "cloud_coverage"
]
lr_days_columns = ["meter_reading_log", "hour", "building_id", "is_holiday"]
for wday in range(0, 7):
    lr_days_columns.append("is_wday" + str(wday))
for week in range(1, 54):
    lr_days_columns.append("is_w" + str(week))
for month in range(1, 13):
    lr_days_columns.append("is_m" + str(month))
hours = range(0, 24)
buildings = range(0, energy["building_id"].max() + 1)


"""Введем функцию для вычисления качества моделей"""


def calculate_model(x, df_lr, lr_columns):
    lr = -1
    model = df_lr[x.building_id][x.hour]
    if len(model) > 0:
        lr = np.sum([x[col] * model[i] for i, col in enumerate(lr_columns[3:])])
        lr += model[len(lr_columns)-3]
        lr = np.exp(lr)
    if lr < 0 or lr * lr == lr:
        lr = 0
    x["meter_reading_lr_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    return x


"""Введем функции для разделения данных, построение моделей и вычисления их
качества (для обновления весов ансамбля)"""

# Ансамбль моделей линейной регрессии: Z = A * погода + B * дни_недели, A+B=1


def train_model(df, columns):
    df_train_lr = pd.DataFrame(df, columns=columns)
    df_lr = [[]] * len(buildings)
    for building in buildings:
        df_lr[building] = [[]] * len(hours)
        df_train_b = df_train_lr[df_train_lr["building_id"]==building]
        for hour in hours:
            df_train_bh = df_train_b[df_train_b["hour"] == hour]
            y = df_train_bh["meter_reading_log"]
            x = df_train_bh.drop(
                labels=["meter_reading_log",
                        "hour",
                        "building_id"
                        ],
                axis=1
            )
            model = LinearRegression(fit_intercept=False).fit(x, y)
            df_lr[building][hour] = model.coef_
            df_lr[building][hour] = np.append(
                df_lr[building][hour], model.intercept_)
    return df_lr


def calculate_weights_model(df_test, df_train, lr_columns):
    df_test = df_test.apply(
        calculate_model,
        axis=1,
        result_type="expand",
        df_lr=train_model(df_train, lr_columns),
        lr_columns=lr_columns
    )
    return pd.Series(
        df_test.groupby(["hour", "building_id"]).sum()["meter_reading_lr_q"]
    )


def calculate_weights():
    df_train, df_test = train_test_split(
        energy[energy["meter_reading"] > 0],
        test_size=0.2
    )
    return (calculate_weights_model(df_test, df_train, lr_weather_columns),
            calculate_weights_model(df_test, df_train, lr_days_columns))


"""Рассчитаем оптимальные веса для каждого часа и здания"""

# 10 раз разобьем исходный набор данных на обучающую/тестовую выборку,
# рассчитаем в каждом случае значения ошибки для каждого здания и часа
# Сформируем список весов: 1 - учитываем регрессию по дням недели,
# 0 - учитываем регрессию по погоде
weights_weather = []
weights_days = []
for i in range(0, 10):
    print ("Расчет весов ансамбля, итерация", i)
    weights_weather_model, weights_days_model = calculate_weights()
    if len(weights_weather) > 0:
        weights_weather = weights_weather + weights_weather_model
    else:
        weights_weather = weights_weather_model
    if len(weights_days) > 0:
        weights_days = weights_days + weights_days_model
    else:
        weights_days = weights_days_model
weights = [0] * len(buildings)
for b in buildings:
    weights[b] = [0] * len(hours)
    for h in hours:
        if weights_weather.loc[h].at[b] > weights_days.loc[h].at[b]:
            weights[b][h] = 1
print(weights)


"""Посчитаем ансамбль линейной регрессии"""

# Разделим данные на обучающие/тестовые
energy_train, energy_test = train_test_split(
    energy[energy["meter_reading"] > 0],
    test_size=0.2
)


"""Обучим модели линейной регрессии по дате/погоде"""

energy_lr_days = train_model(energy_train, lr_days_columns)
energy_lr_weather = train_model(energy_train, lr_weather_columns)


"""Рассчитаем финальное качество ансамбля"""

# Если вес 1, то считаем регрессию по дням недели, если 0 - то по погоде


def calculate_model_ensemble(x, model, columns):
    lr = -1
    if len(model) > 0:
        lr = np.sum([x[col] * model[i] for i, col in enumerate(columns[3:])])
        lr += model[len(columns) - 3]
        lr = np.exp(lr)
    if lr < 0 or lr * lr == lr:
        lr = 0
    return lr


def calculate_models_ensemble(x):
    lr_d = calculate_model_ensemble(
        x,
        energy_lr_days[x.building_id][x.hour],
        lr_days_columns
    )
    lr_w = calculate_model_ensemble(
        x,
        energy_lr_weather[x.building_id][x.hour],
        lr_weather_columns
    )
    if weights[x.building_id][x.hour] == 1:
        lr = lr_d
    else:
        lr = lr_w
    lr_sum = (lr_w + lr_d) / 2
    x["meter_reading_lr_q"] = (np.log(x.meter_reading + 1) - np.log(1 + lr)) ** 2
    x["meter_reading_sum_q"] = (np.log(x.meter_reading + 1) - np.log(
        1 + lr_sum)) ** 2
    return x


# В теории, в идеальном случае, ансамбль линейной регрессии не должен давать
# никакого преимущества, потому что если
# 𝑧1=𝐴𝑥+𝐵𝑦+𝐶,𝑧2=𝐷𝑠+𝐸𝑡+𝐹,то
#
# 𝑧=𝛼𝑧1+𝛽𝑧2=𝛼𝐴𝑥+𝛼𝐵𝑦+𝛼𝐶+𝛽𝐷𝑠+𝛽𝐸𝑡+𝛽𝐹=𝐴1𝑥+𝐵1𝑦+𝐷1𝑠+𝐸1𝑡+𝐹1
#
# И по сути ансамбль линейной регрессии - это просто линейная регрессия по всем
# переменным. Но при использовании небольших наборов
# (чтобы исключить переобучение) связанных переменных для разных моделей
# регрессии можно получить небольшой выигрыш.
#
# Ансамбль регрессии в нашем случае не дает никакого улучшения относительно
# регрессии по совокупному набору параметров.
#
# Однако, использование усредненной суммы показателей каждой конкретной модели
# дало выигрыш порядка 6% относительно модели по всем показателям. В этом случае
# сумму моделей линейной регрессии "компенсирует" ошибки каждой конкретной
# модели и работает точнее.

energy_test = energy_test.apply(
    calculate_models_ensemble,
    axis=1,
    result_type="expand"
)
energy_test_lr_rmsle = np.sqrt(
    energy_test["meter_reading_lr_q"].sum() / len(energy_test)
)
energy_test_sum_rmsle = np.sqrt(
    energy_test["meter_reading_sum_q"].sum() / len(energy_test)
)
print("Качество ансамбля, 20 зданий:", energy_test_lr_rmsle,
      round(energy_test_lr_rmsle, 1))
print("Качество ансамбля суммы, 20 зданий:", energy_test_sum_rmsle,
      round(energy_test_sum_rmsle, 1))
