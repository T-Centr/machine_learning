"""
Постановка задачи

Загрузите данные, приведите их к числовым, заполните пропуски, нормализуйте
данные и оптимизируйте память.

Разделите выборку на обучающую/проверочную в соотношении 80/20.

Постройте 2 модели - kNN по 100 соседей и множественную логистическую
регрессию - по наиболее оптимальным наборам параметров (для каждой модели),
используйте для этого перекрестную проверку GridSearchCV.

Проведите предсказание и проверьте качество через каппа-метрику.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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

data["Product_Info_2_1"] = data["Product_Info_2"].str.slice(0, 1)
data["Product_Info_2_2"] = pd.to_numeric(data["Product_Info_2"].str.slice(1, 2))
data.drop(labels=["Product_Info_2"], axis=1, inplace=True)
for l in data["Product_Info_2_1"].unique():
    data["Product_Info_2_1" + l] = data["Product_Info_2_1"].isin([l]).astype("int8")
data.drop(labels=["Product_Info_2_1"], axis=1, inplace=True)
data.fillna(value=-1, inplace=True)


"""Общий набор столбцов для расчета"""

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


"""Разделение данных"""

# Преобразуем выборки в отдельные наборы данных

data_train, data_test = train_test_split(data_transformed, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
print(data_train.head())


"""Логистическая регрессия"""

# Найдем оптимальный набор столбцов и рассчитаем по нему модель логистической
# регрессии


def regression_model(columns, df, fit=True):
    x = pd.DataFrame(df, columns=columns)
    model = LogisticRegression(max_iter=1000)
    if fit:
        model.fit(x, y=df["Response"])
    return model


def model_score(columns, df_train, model_func):
    x = pd.DataFrame(df_train, columns=columns)
    model = model_func(columns, df_train, False)
    cv_grid = GridSearchCV(
        model,
        {},
        cv=5,
        n_jobs=2,
        scoring=make_scorer(cohen_kappa_score)
    )
    cv_grid.fit(x, df_train["Response"])
    return cv_grid.best_score_


def find_opt_columns(data_train, model_func):
    kappa_score_opt = 0
    columns_opt = []
    for col in columns_transformed:
        kappa_score = model_score([col], data_train, model_func)
        if kappa_score > kappa_score_opt:
            columns_opt = [col]
            kappa_score_opt = kappa_score
    for col in columns_transformed:
        if col not in columns_opt:
            columns_opt.append(col)
            kappa_score = model_score(columns_opt, data_train, model_func)
            if kappa_score < kappa_score_opt:
                columns_opt.pop()
            else:
                kappa_score_opt = kappa_score
    return columns_opt, kappa_score_opt


columns_opt_logr, kappa_score_opt = find_opt_columns(data_train, regression_model)
model_logr = regression_model(columns_opt_logr, data_train)
print(kappa_score_opt, columns_opt_logr)


"""kNN"""

# Посчитаем оптимальную модель для kNN


def knn_model(columns, df_train, fit=True):
    y = data_train["Response"]
    x = pd.DataFrame(df_train, columns=columns)
    model = KNeighborsClassifier(n_neighbors=100)
    if fit:
        model.fit(x, y)
    return model


columns_opt_knn, kappa_score_opt = find_opt_columns(data_train, knn_model)
model_knn = knn_model(columns_opt_knn, data_train)
print(kappa_score_opt, columns_opt_knn)


"""Предсказание данных и оценка модели"""

x_test = pd.DataFrame(data_test, columns=columns_opt_logr)
data_test["target_logr"] = model_logr.predict(x_test)
x_test = pd.DataFrame(data_test, columns=columns_opt_knn)
data_test["target_knn"] = model_knn.predict(x_test)

print(
    "Логистическая регрессия:",
    round(
        cohen_kappa_score(
            data_test["target_logr"],
            data_test["Response"],
            weights="quadratic"
        ),
        3
    )
)
print(
    "kNN, 100:",
    round(
        cohen_kappa_score(
            data_test["target_knn"],
            data_test["Response"],
            weights="quadratic"
        ),
        3
    )
)


"""Матрица неточностей"""

print(
    "Логистическая регрессия:\n",
    confusion_matrix(
        data_test["target_logr"],
        data_test["Response"]
    )
)
print(
    "kNN:\n",
    confusion_matrix(
        data_test["target_knn"],
        data_test["Response"]
    )
)
