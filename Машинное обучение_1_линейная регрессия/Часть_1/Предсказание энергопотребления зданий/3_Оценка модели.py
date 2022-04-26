"""
Постановка задачи

Построить простую модель энергопотребления здания по среднему значению, оценить
эффективность модели через метрику

RMSLE =

n - число наблюдений
log - натуральный логарифм
p_i - вычисленное значение метрики
a_i - заданное значение метрики

Данные: http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz

Соревнование: https://www.kaggle.com/c/ashrae-energy-prediction/

© ITtensive, 2020
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов

pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов

pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных"""

# Дополнительно сразу отсечем пустые дни и выделим час из значения времени
energy_0 = pd.read_csv(
    "http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz"
)
energy_0 = energy_0[energy_0["meter_reading"] > 0]
energy_0["timestamp"] = pd.to_datetime(energy_0["timestamp"])
energy_0["hour"] = energy_0["timestamp"].dt.hour
print(energy_0.head())


"""Разделение данных на обучение и проверку"""

# Выделим 20% всех данных на проверку, остальные оставим на обучение
energy_0_train, energy_0_test = train_test_split(energy_0, test_size=0.2)
print(energy_0_train.head())


"""Создадим модели"""

# Среднее и медианное значение потребление энергии по часам
energy_0_train_hours = energy_0_train.groupby("hour")
energy_0_train_averages = pd.DataFrame({
    "Среднее": energy_0_train_hours.mean()["meter_reading"],
    "Медиана": energy_0_train_hours.median()["meter_reading"]
})
print(energy_0_train_averages)


"""Функция проверки модели"""

# RMSLE =

# Для вычисления метрики создадим шесть новых столбцов в тестовом наборе данных:
# с логарифмом значения метрики, предсказанием по среднему и по медиане, а также
# с квадратом разницы предсказаний и логарифма значения. Последний столбец
# добавим, чтобы сравнить предсказание с его отсутствием - нулями в значениях.


def calculate_model(x):
    meter_reading_log = np.log(x.meter_reading + 1)
    meter_reading_mean = np.log(energy_0_train_averages["Среднее"][x.hour] + 1)
    meter_reading_median = np.log(
        energy_0_train_averages["Медиана"][x.hour] + 1
    )
    x["meter_reading_mean_q"] = (meter_reading_log - meter_reading_mean) ** 2
    x["meter_reading_median_q"] = (meter_reading_log - meter_reading_median) ** 2
    x["meter_reading_zero_q"] = (meter_reading_log) ** 2
    return x


energy_0_test = energy_0_test.apply(
    calculate_model,
    axis=1,
    result_type="expand"
)
print(energy_0_test.head())


# Теперь остается просуммировать квадраты расхождений, разделить на количество
# значений и извлечь квадратный корень
energy_0_test_median_rmsle = np.sqrt(
    energy_0_test["meter_reading_median_q"].sum() / len(energy_0_test)
)
energy_0_test_mean_rmsle = np.sqrt(
    energy_0_test["meter_reading_mean_q"].sum() / len(energy_0_test)
)
energy_0_test_zero_rmsle = np.sqrt(
    energy_0_test["meter_reading_zero_q"].sum() / len(energy_0_test)
)
print("Качество медианы:", energy_0_test_median_rmsle)
print("Качество среднего:", energy_0_test_mean_rmsle)
print("Качество нуля:", energy_0_test_zero_rmsle)
