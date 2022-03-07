"""
Постановка задачи

Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Сформируем параллельный ансамбль из логистической регрессии, SVM, случайного
леса и LightGBM. Используем лучшие гиперпараметры, подобранные ранее. Итоговое
решение рассчитаем на основании весов (вероятностей).

Проведем предсказание и проверим качество через каппа-метрику.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

data["Product_Info_2_1"] = data["Product_Info_2"].str.slice(0, 1)
data["Product_Info_2_2"] = pd.to_numeric(data["Product_Info_2"].str.slice(1, 2))
data.drop(labels=["Product_Info_2"], axis=1, inplace=True)
for l in data["Product_Info_2_1"].unique():
    data["Product_Info_2_1" + l] = data["Product_Info_2_1"].isin([l]).astype("int8")
data.drop(labels=["Product_Info_2_1"], axis=1, inplace=True)
data.fillna(value=-1, inplace=True)
data["Response"] = data["Response"] - 1


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


"""Разделение данных"""

# Преобразуем выборки в отдельные наборы данных

data_train, data_test = train_test_split(data_transformed, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
print(data_train.head())


"""Построение базовых моделей"""

x = pd.DataFrame(data_train, columns=columns_transformed)

# SVM

model_svm = SVC(kernel='linear', probability=True, max_iter=10000)
model_svm.fit(x, data_train["Response"])

# Логистическая регрессия

model_logr = LogisticRegression(max_iter=1000)
model_logr.fit(x, data_train["Response"])

# Случайный лес

model_rf = RandomForestClassifier(
    random_state=17,
    n_estimators=76,
    max_depth=17,
    max_features=27,
    min_samples_leaf=20
)
model_rf.fit(x, data_train["Response"])

# LightGBM, используем multiclass

model_lgb = lgb.LGBMRegressor(
    random_state=17,
    max_depth=18,
    min_child_samples=18,
    num_leaves=75,
    n_estimators=1000,
    objective="multiclass",
    num_class=8
)
model_lgb.fit(x, data_train["Response"])


"""Расчет предказаний"""

# Кроме непосредственно значений дополнительно посчитаем вероятности совпадения
# с тем или иным классом

x_test = pd.DataFrame(data_test, columns=columns_transformed)

data_test_svm_proba = pd.DataFrame(model_svm.predict_proba(x_test))

data_test_logr_proba = pd.DataFrame(model_logr.predict_proba(x_test))

data_test_rf_proba = pd.DataFrame(model_rf.predict_proba(x_test))

data_test_lgb = pd.DataFrame(model_lgb.predict(x_test))

# Несколько вариантов ансамблей с голосованием (выбор класса выполняется для
# каждого кортежа по отдельности):
#
# "Мягкое" голосование (в том числе, с определенными весами): суммируются
# вероятности каждого класса среди всех оценок, выбирается наибольшее.
# "Жесткое" (мажоритарное) голосование: выбирается самый популярный класс среди
# моделей (число моделей должно быть нечетным).
# Отсечение: из вероятностей моделей выбирается только наиболее значимые,
# например, больше 0.3.
# Экспертное голосование: вес оценки эксперта зависит от кортежа данных и самого
# класса (например, если определенная модель предсказывает определенный класс
# точнее других).
# Здесь используем "мягкое" голосование, для этого необходимо рассчитать
# вероятности всех класса для каждого кортежа данных.


def vote_class(x):
    a = np.argmax(x.values)
    return a


data_test_proba = data_test_svm_proba.copy()
for i in range(0, 8):
    data_test_proba[i] = 5 * data_test_proba[i]
    data_test_proba[i] = data_test_proba[i] + 5 * data_test_logr_proba[i]
    data_test_proba[i] = data_test_proba[i] + data_test_rf_proba[i]
    data_test_proba[i] = data_test_proba[i] + 12 * data_test_lgb[i]
data_test_proba["target"] = data_test_proba.apply(vote_class, axis=1)


"""Оценка ансамбля"""

# Рассчитаем оценку взвешенного предсказания 4 моделей

# Кластеризация дает 0.192, kNN(100) - 0.382, лог. регрессия - 0.512/0.496,
# SVM - 0.95, реш. дерево - 0.3, случайный лес - 0.487, XGBoost - 0.536,
# градиентный бустинг - 0.56, LightGBM - 0.569, CatBoost - 0.542

print(
    "Ансамбль классификации:",
    round(
        cohen_kappa_score(
            data_test_proba["target"],
            data_test["Response"],
            weights="quadratic"
        ),
        3
    )
)


"""Матрица неточностей"""

print(
    "Ансамбль классификации\n",
    confusion_matrix(
        data_test_proba["target"],
        data_test["Response"]
    )
)


"""Само-проверка"""

# Проверим, насколько ансамбль хорошо предсказывает обучающие данные

data_copy = data_train.copy()
x_copy = pd.DataFrame(data_copy, columns=columns_transformed)

data_copy_svm = pd.DataFrame(model_svm.predict_proba(x_copy))

data_copy_logr = pd.DataFrame(model_logr.predict_proba(x_copy))

data_copy_rf = pd.DataFrame(model_rf.predict_proba(x_copy))

data_copy_lgb = pd.DataFrame(model_lgb.predict(x_copy))

for i in range(0, 8):
    data_copy_svm[i] = 5 * data_copy_svm[i]
    data_copy_svm[i] = data_copy_svm[i] + 5 * data_copy_logr[i]
    data_copy_svm[i] = data_copy_svm[i] + data_copy_rf[i]
    data_copy_svm[i] = data_copy_svm[i] + 12 * data_copy_lgb[i]
target = data_copy_svm.apply(vote_class, axis=1)

print(
    "Результат:",
    round(
        cohen_kappa_score(
            target,
            data_copy["Response"],
            weights="quadratic"
        ),
        3
    )
)


print(confusion_matrix(target, data_copy["Response"]))
