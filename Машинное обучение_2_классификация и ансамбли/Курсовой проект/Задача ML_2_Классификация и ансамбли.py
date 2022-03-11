"""
Постановка задачи

Загрузите данные, приведите их к числовым, заполните пропуски, нормализуйте
данные и оптимизируйте память.

Сформируйте параллельный ансамбль (стекинг) из CatBoost, градиентного бустинга,
XGBoost и LightGBM. Используйте лучшие гиперпараметры, подобранные ранее, или
найдите их через перекрестную проверку. Итоговое решение рассчитайте на
основании самого точного предсказания класса у определенной модели ансамбля:
выберите для каждого класса модель, которая предсказывает его лучше всего.

Проведите расчеты и выгрузите результат в файл submission.csv
Итоговый файл с кодом (.py или .ipynb) выложите в github с портфолио.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz
https://video.ittensive.com/machine-learning/prudential/test.csv.gz
https://video.ittensive.com/machine-learning/prudential/sample_submission.csv.gz
"""


import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from catboost import Pool, CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
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


"""Построение базовых моделей"""

x = pd.DataFrame(data_transformed, columns=columns_transformed)

# XGBoost

model_xgb = XGBClassifier(
    max_depth=17,
    max_features=27,
    n_estimators=76,
    min_samples_leaf=20
)
model_xgb.fit(x, data['Response'])

# CatBoost

model_cb = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.57,
    random_seed=17,
    depth=6,
    l2_leaf_reg=2,
    loss_function='MultiClass',
    bootstrap_type="MVS"
)
model_cb.fit(Pool(data=x, label=data["Response"]))

# Градиентный бустинг

print(x.info())

model_gbc = GradientBoostingClassifier(
    random_state=17,
    max_depth=13,
    max_features=26,
    min_samples_leaf=21,
    n_estimators=75
)
model_gbc.fit(x, data['Response'])

# LightGBM

model_lgb = lgb.LGBMRegressor(
    random_state=17,
    max_depth=18,
    min_child_samples=17,
    num_leaves=35,
    n_estimators=10000
)
model_lgb.fit(x, data['Response'])


"""Загрузка данных для расчета"""

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


"""Расчет предказаний"""

data_test["target_xgb"] = model_xgb.predict(data_test_transformed)

data_test["target_cb"] = model_cb.predict(Pool(data=data_test_transformed))

data_test["target_gbc"] = model_gbc.predict(data_test_transformed)

data_test["target_lgb"] = np.round(
    model_lgb.predict(data_test_transformed)).astype("int8")

# Классы смещены на 1: начинаются от 0 и заканчиваются 7. Судя по рассчитанным
# матрицам ошибок, для 0, 1, 3, 4 и 6 классов точнее работает градиентный
# бустинг, для 2 - XGBoost, для 5 - LightGBM, для 7 - логистическая регрессия.

# Точные параметры классов можно перерассчитать, например, через перекрестную
# проверку всех данных.


def vote_class(x):
    if x.target_xgb == 2:
        class_ = x.target_xgb
    elif x.target_lgb == 7:
        class_ = x.target_lgb
    elif x.target_cb == 0:
        class_ = x.target_cb
    else:
        class_ = x.target_gbc
    x["Response"] = class_ + 1
    return x


data_test = data_test.apply(vote_class, axis=1)
print(data_test.head())


"""Формирование и выгрузка результатов"""

# Загрузим примерный файл, заменим в нем результаты и сохраним.

# Число строк в файле будет равно размену набора данных + 1 заголовочная строка.

submission = pd.read_csv(
    "https://video.ittensive.com/machine-learning/prudential/"
    "sample_submission.csv.gz"
)
submission["Response"] = data_test["Response"].astype("int8")
submission.to_csv("submission.csv", index=False)
print(len(submission["Response"]) + 1)


"""Само-проверка модели"""

# Рассчитаем точность классификации на обучающих данных.

data_copy = data_transformed.copy()
x_copy = pd.DataFrame(data_copy, columns=columns_transformed)
copy_dataset = Pool(data=x_copy, label=data_copy["Response"])
data_copy["target_xgb"] = model_xgb.predict(x_copy)
data_copy["target_cb"] = model_cb.predict(copy_dataset)
data_copy["target_gbc"] = model_gbc.predict(x_copy)
data_copy["target_lgb"] = np.round(model_lgb.predict(x_copy)).astype("int8")

class_target = ["target_gbc"] * 8


def vote_class_enumerate(x):
    for _, target in enumerate(class_target):
        if x[target] == _:
            x["Response"] = x[target]
            break
    return x


kappa_min = 0
for target_model in ["xgb", "cb", "gbc", "lgb"]:
    print("Проверяем модель:", target_model)
    target_model = "target_" + target_model
    for class_ in range(0, 8):
        target_model_prev = class_target[class_]
        class_target[class_] = target_model
        data_copy = data_copy.apply(vote_class_enumerate, axis=1)
        kappa = cohen_kappa_score(
            data_copy["Response"],
            data["Response"],
            weights='quadratic'
        )
        if kappa > kappa_min:
            kappa_min = kappa
        else:
            class_target[class_] = target_model_prev
    print("Максимальная оценка:", kappa_min)
print(class_target)

data_copy = data_copy.apply(vote_class_enumerate, axis=1)


"""Матрица неточностей"""

print(
    "Результат:",
    round(
        cohen_kappa_score(
            data_copy["Response"],
            data["Response"],
            weights='quadratic'
        ),
        3
    )
)
print(confusion_matrix(data_copy["Response"], data["Response"]))
