"""
Постановка задачи

Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Построим LightGBM модель с оптимальными параметрами. Выгрузим результаты
расчетов в требуемом формате.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz
https://video.ittensive.com/machine-learning/prudential/test.csv.gz
https://video.ittensive.com/machine-learning/prudential/sample_submission.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import lightgbm as lgb
from sklearn import preprocessing


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов
pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов
pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных"""

data = pd.read_csv(
    "https://video.ittensive.com/machine-learning/prudential/train.csv.gz"
)
print(data.info())


"""Предобработка данных"""


def data_preprocess(df):
    df["Product_Info_2_1"] = df["Product_Info_2"].str.slice(0, 1)
    df["Product_Info_2_2"] = pd.to_numeric(df["Product_Info_2"].str.slice(1, 2))
    df.drop(labels=["Product_Info_2"], axis=1, inplace=True)
    for l in df["Product_Info_2_1"].unique():
        df["Product_Info_2_1" + l] = df["Product_Info_2_1"].isin([l]).astype("int8")
    df.drop(labels=["Product_Info_2_1"], axis=1, inplace=True)
    df.fillna(value=-1, inplace=True)
    data["Response"] = data["Response"] - 1
    return df


data = data_preprocess(data)


"""Набор столбцов для расчета"""

columns_groups = [
    "Insurance_History",
    "InsurеdInfo",
    "Medical_Keyword",
    "Family_Hist",
    "Medical_History",
    "Product_Info"
]
columns = ["Wt", "Ht", "Ins_Age", "BMI"]
for cg in columns_groups:
    columns.extend(data.columns[data.columns.str.startswith(cg)])
print(columns)


"""Нормализация данных"""

scaler = preprocessing.StandardScaler()
data_transformed = pd.DataFrame(
    scaler.fit_transform(
        pd.DataFrame(
            data,
            columns=columns
        )
    )
)
columns_transformed = data_transformed.columns
data_transformed["Response"] = data["Response"]


"""Оптимизация памяти"""


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
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2),
          'Мб (минус', round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df


data_transformed = reduce_mem_usage(data_transformed)
print(data_transformed.info())


"""LightGBM"""

# Рассчитаем модель по оптимальным показателям. Возможно уточнение/дообучение
# уже на всей выборке без разбиения на обучающую/тестовую.

x = pd.DataFrame(data_transformed, columns=columns_transformed)
model = lgb.LGBMRegressor(
    random_state=17,
    max_depth=17,
    min_child_samples=18,
    num_leaves=35,
    n_estimators=1000
)

# Построим модель

model.fit(x, data["Response"])


"""Загрузка данных для расчетов"""

# Применим построенную модель для расчета актуальных данных.
# Будем использовать ранее рассчитанные значения нормализация данных.

data_test = pd.read_csv(
    "https://video.ittensive.com/machine-learning/prudential/test.csv.gz"
)
data_test = data_preprocess(data_test)
data_test = reduce_mem_usage(data_test)
data_test_transformed = pd.DataFrame(
    scaler.transform(
        pd.DataFrame(
            data_test,
            columns=columns
        )
    )
)
print(data_test_transformed.info())


"""Предсказание данных и оценка модели"""

# LightGBM возвращает дробное значение класса, его нужно округлить.
# Дополнительно приведем значение класса к диапазону 1...8

data_test_transformed["Response"] = np.round(
    model.predict(data_test_transformed)) + 1
data_test_transformed["Response"] = (
    data_test_transformed["Response"].apply(
        lambda x: 1 if x < 1 else 8 if x > 8 else x
    )
)
print(data_test_transformed.head())


"""Формирование результатов"""

# Загрузим пример данных для отправки и заменим в нем столбец Response на
# рассчитанный ранее.

submission = pd.read_csv(
    "https://video.ittensive.com/machine-learning/"
    "prudential/sample_submission.csv.gz"
)
print(submission.head())

submission["Response"] = data_test_transformed["Response"].astype("int8")
print(submission.head())


"""Выгрузка результатов"""

submission.to_csv("submission.csv", index=False)
