"""
Постановка задачи
Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Построим модель опорных векторов (SVM) для наиболее оптимального разделения
параметров на классы, используем несколько реализаций: линейную (LinearSVC) и
через градиентный бустинг (SGDClassifier).

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
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
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
    scaler.fit_transform(pd.DataFrame(data, columns=columns))
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


"""SVM"""

# Выбираем направления преобразований исходных данных, чтобы различные классы
# можно было разделить гиперплоскостью по значениям параметров.

x = pd.DataFrame(data_train, columns=columns_transformed)
model_lin = LinearSVC(max_iter=10000)
model_lin.fit(x, data_train['Response'])

model_sgd = SGDClassifier()
model_sgd.fit(x, data_train['Response'])


"""Предсказание данных и оценка модели"""

x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test["target_lin"] = model_lin.predict(x_test)
data_test["target_sgd"] = model_sgd.predict(x_test)

# Кластеризация дает 0.192, kNN(100) - 0.3, лог. регрессия - 0.512/0.496

print(
    "SVM (линейный):",
    round(cohen_kappa_score(
        data_test["target_lin"],
        data_test["Response"],
        weights='quadratic'
    ),
        3
    )
)
print(
    "SVM (градиент):",
    round(
        cohen_kappa_score(
            data_test["target_sgd"],
            data_test["Response"],
            weights='quadratic'
        ),
        3
    )
)


"""Матрица неточностей"""

print(
    "SVM (линейный)\n",
    confusion_matrix(
        data_test["target_lin"],
        data_test["Response"]
    )
)
print(
    "SVM (градиент)\n",
    confusion_matrix(
        data_test["target_sgd"],
        data_test["Response"]
    )
)
