"""
Постановка задачи

Загрузите данные, приведите их к числовым, заполните пропуски, нормализуйте
данные и оптимизируйте память.

Разделите выборку на обучающую/проверочную в соотношении 80/20.

Постройте ансамбль решающих деревьев, используя градиентный бустинг
(GradientBoostingClassifier). Используйте перекрестную проверку, чтобы найти
наилучшие параметры ансамбля, или используйте параметры от случайного
леса: max_depth=17, max_features=27, min_samples_leaf=20, n_estimators=76.

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
from sklearn.ensemble import GradientBoostingClassifier
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


"""Набор столбцов для расчета"""

columns_groups = [
    "Insurance_History",
    "InsurеdInfo",
    "Medical_Keyword",
    "Family_Hist",
    "Medical_History",
    "Product_Info"]
columns = ["Wt", "Ht", "Ins_Age", "BMI"
           ]
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


"""Градиентный бустинг"""

# Деревья для градиентного бустинга строятся последовательно для минимизации
# ошибки предыдущего дерева (или деревьев). При этом в самом дереве разбиение
# выполняется по минимизации информационных потерь, без учета сортировки
# исходных данных по количеству информации в них.

x = pd.DataFrame(data_train, columns=columns_transformed)
model = GradientBoostingClassifier(
    random_state=17,
    max_depth=14,
    max_features=27,
    min_samples_leaf=20,
    n_estimators=76
)

# Диапазон тестирования параметров модели ограничен только вычислительной
# мощностью. Для проверки модели имеет смысл провести индивидуальные
# перекрестные проверки для каждого параметра в отдельности, затем в итоговой
# проверке перепроверить самые лучшие найденные параметры с отклонением +/-10%.

tree_params = {
    'max_depth': range(12, 15),
    'max_features': range(25, 28),
    'n_estimators': range(74, 77),
    'min_samples_leaf': range(20, 23)
}
tree_grid = GridSearchCV(
    model,
    tree_params,
    cv=5,
    n_jobs=2,
    verbose=True,
    scoring=make_scorer(cohen_kappa_score)
)
tree_grid.fit(x, data_train['Response'])
print(tree_grid.best_params_)

# Выведем самые оптимальные параметры и построим итоговую модель

print(tree_grid.best_params_)
model = GradientBoostingClassifier(
    random_state=17,
    min_samples_leaf=tree_grid.best_params_['min_samples_leaf'],
    max_depth=tree_grid.best_params_['max_depth'],
    max_features=tree_grid.best_params_['max_features']
)

model.fit(x, data_train['Response'])


"""Предсказание данных и оценка модели"""

x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test["target"] = model.predict(x_test)

# Кластеризация дает 0.192, kNN(100) - 0.3, лог. регрессия - 0.512/0.496,
# SVM - 0.95, реш. дерево - 0.3, случайный лес - 0.487, XGBoost - 0.522

print(
    "Градиентный бустинг:",
    round(
        cohen_kappa_score(
            data_test["target"],
            data_test["Response"],
            weights='quadratic'
        ),
        3
    )
)


"""Матрица неточностей"""

print(
    "Градиентный бустинг\n",
    confusion_matrix(
        data_test["target"],
        data_test["Response"]
    )
)
