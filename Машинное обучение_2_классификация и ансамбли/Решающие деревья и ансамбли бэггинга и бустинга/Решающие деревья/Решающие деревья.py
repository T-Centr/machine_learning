"""
Постановка задачи
Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Построим несколько моделей дерева решений, найдем оптимальную через перекрестную
валидацию (CV).

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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
from IPython.display import SVG, display
from graphviz import Source
import os
os.environ["PATH"] += (os.pathsep + 'C:/Program Files/Graphviz/bin/')


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

# "Облегченный" вариант для визуализации дерева

columns = ["Wt", "Ht", "Ins_Age", "BMI"]


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


"""Дерево решений"""

# Минимальное число "одинаковых" значений для ветвления - 10

x = pd.DataFrame(data_train, columns=columns_transformed)
model = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
model.fit(x, data_train["Response"])


"""Визуализация дерева"""

# Доступно несколько форматов вывода, большой SVG выводится в Jupyter Notebook
# сложнее, поэтому используем PNG.
#
# В качестве названий параметров передаем исходный список.

graph = Source(
    export_graphviz(
        model,
        out_file=None,
        feature_names=columns,
        filled=True,
        class_names=data_train["Response"].unique().astype("str")
    )
)
with open("tree.png", "wb") as f:
    f.write(graph.pipe(format="png"))


"""Влияние признаков"""

# Выведем долю влияния признаков на конечный ответ в дерево

print(model.feature_importances_)


"""Перекрестная проверка (CV)"""

# Разбиваем обучающую выборку еще на k (часто 5) частей, на каждой части данных
# обучаем модель. Затем проверяем 1-ю, 2-ю, 3-ю, 4-ю части на 5; 1-ю, 2-ю, 3-ю,
# 5-ю части на 4 и т.д.
#
# В итоге обучение пройдет весь набор данных, и каждая часть набора будет
# проверена на всех оставшихся (перекрестным образом).
#
# Перекрестная проверка используется для оптимизации параметров исходной
# модели - решающего дерева в данном случае. Зададим несколько параметров для
# перебора и поиска самой точной модели.
#
# Для проверки будем использовать каппа-метрику.

tree_params = {
    'max_depth': range(10, 20),
    'max_features': range(1, round(len(columns_transformed))),
    'min_samples_leaf': range(20, 100)
}

tree_grid = GridSearchCV(
    model,
    tree_params,
    cv=5,
    n_jobs=2,
    verbose=True,
    scoring=make_scorer(cohen_kappa_score)
)
tree_grid.fit(x, data_train["Response"])


# Выведем самые оптимальные параметры и построим итоговую модель

print(tree_grid.best_params_)
model = DecisionTreeClassifier(
    random_state=17,
    min_samples_leaf=tree_grid.best_params_['min_samples_leaf'],
    max_features=tree_grid.best_params_['max_features'],
    max_depth=tree_grid.best_params_['max_depth']
)
model.fit(x, data_train["Response"])


"""Предсказание данных и оценка модели"""

x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test["target"] = model.predict(x_test)

# Кластеризация дает 0.192, kNN(100) - 0.3,
# лог. регрессия - 0.512/0.496, SVM - 0.95


print(
    "Решающее дерево:",
    round(
        cohen_kappa_score(
            data_test["target"],
            data_test["Response"],
            weights="quadratic"
        ),
        3
    )
)

# Решающее дерево: 0.314


"""Матрица неточностей"""

print(
    "Решающее дерево:\n",
    confusion_matrix(
        data_test["target"],
        data_test["Response"]
    )
)
