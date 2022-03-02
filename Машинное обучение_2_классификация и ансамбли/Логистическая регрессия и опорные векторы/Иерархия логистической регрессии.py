"""
Постановка задачи
Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Построим 4 модели логистической регрессии: для 8, 6 и остальных классов, для 2,
5 и остальных, для 1, 7 и остальных, и для 4 и 3 - по убыванию частоты значения.
Будем использовать перекрестную проверку при принятии решения об оптимальном
наборе столбцов.

Проведем предсказание и проверим качество через каппа-метрику.

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


"""Оптимизация памяти"""


def reduce_mem_usage (df):
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


data = reduce_mem_usage(data)
print(data.info())


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


"""Предобработка данных"""

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


"""Разделение данных"""

# Преобразуем выборки в отдельные наборы данных

data_train, data_test = train_test_split(data_transformed, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
print(data_train.head())


"""Логистическая регрессия"""

# В обучающих данных пометим все классы, кроме 6 и 8, как 0 - и проведем
# обучение по такому набору данных.
#
# Затем в оставшихся данных (в которых класс не равен 6 или 8) заменим все
# классы, кроме 7 и 1, на 0 - и снова проведем обучение. И т.д. Получим иерархию
# классификаторов: 8/6/нет -> 7/1/нет -> 2/5/нет -> 4/3


def regression_model(columns, df):
    x = pd.DataFrame(df, columns=columns)
    model = LogisticRegression(max_iter=1000)
    model.fit(x, df["Response"])
    return model


def logistic_regression(columns, df_train):
    model = regression_model(columns, df_train)
    logr_grid = GridSearchCV(
        model,
        {},
        cv=5,
        n_jobs=2,
        scoring=make_scorer(cohen_kappa_score)
    )
    x = pd.DataFrame(df_train, columns=columns)
    logr_grid.fit(x, df_train["Response"])
    return logr_grid.best_score_


"""Оптимальный набор столбцов"""

# Для каждого уровня иерархии это будет свой набор столбцов в исходных данных.

"""Перекрестная проверка"""

# Разбиваем обучающую выборку еще на k (часто 5) частей, на каждой части данных
# обучаем модель. Затем проверяем 1-ю, 2-ю, 3-ю, 4-ю части на 5; 1-ю, 2-ю, 3-ю,
# 5-ю части на 4 и т.д.
#
# В итоге обучение пройдет весь набор данных, и каждая часть набора будет
# проверена на всех оставшихся (перекрестным образом).


def find_opt_columns(data_train):
    kappa_score_opt = 0
    columns_opt = []
    for col in columns_transformed:
        kappa_score = logistic_regression([col], data_train)
        if kappa_score > kappa_score_opt:
            columns_opt = [col]
            kappa_score_opt = kappa_score
    for col in columns_transformed:
        if col not in columns_opt:
            columns_opt.append(col)
            kappa_score = logistic_regression(columns_opt, data_train)
            if kappa_score < kappa_score_opt:
                columns_opt.pop()
            else:
                kappa_score_opt = kappa_score
    return columns_opt, kappa_score_opt


# Будем последовательно "урезать" набор данных при расчете более глубоких
# моделей: после получения разделения на 8 и остальные отсечем все данные со
# значением 8, и т.д.
#
# После каждого расчета модели будем вычислять значения в проверочной выборке.
# Проверочную выборку нулями заполнять не будем, иначе оценка будет считаться
# некорректно.
#
# Набор разделений 6/8, 2/5, 1/7, 3/4 дает наибольшую точность

responses = [[6, 8], [2, 5], [1, 7], [3, 4]]
logr_models = [{}] * len(responses)
data_train_current = data_train.copy()
i = 0
for response in responses:
    m_train = data_train_current.copy()
    if response != [3, 4]:
        m_train["Response"] = m_train["Response"].apply(
            lambda x: 0 if x not in response else x
        )
    columns_opt, kappa_score_opt = find_opt_columns(m_train)
    print(i, kappa_score_opt, columns_opt)
    logr_models[i] = {
        "model": regression_model(columns_opt, m_train),
        "columns": columns_opt
    }
    if response != [3, 4]:
        data_train_current = data_train_current[
            ~data_train_current["Response"].isin(response)
        ]
    i += 1


"""Предсказание данных и оценка модели"""

# Последовательно считаем предсказания для каждой классификации. После этого
# объединяем предсказание по иерархии.


def logr_hierarchy(x):
    for response in range(0, len(responses)):
        if x["target" + str(response)] > 0:
            x["target"] = x["target" + str(response)]
            break
    return x


for response in range(0, len(responses)):
    model = logr_models[response]["model"]
    columns_opt = logr_models[response]["columns"]
    x = pd.DataFrame(data_test, columns=columns_opt)
    data_test["target" + str(response)] = model.predict(x)


data_test = data_test.apply(
    logr_hierarchy,
    axis=1,
    result_type="expand"
)
print(data_test.head())

# Кластеризация дает 0.192, kNN(100) - 0.3, простая лог. регрессия - 0.512

print(
    "Логистическая регрессия, 4 уровня:",
    round(
        cohen_kappa_score(
            data_test["target"],
            data_test["Response"],
            weights="quadratic"
        ),
        3
    )
)


"""Матрица неточностей"""

print(confusion_matrix(data_test["target"], data_test["Response"]))
