"""
Постановка задачи

Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Построим ансамбль решающих деревьев, используя патентованный градиентный бустинг
Яндекса (CatBoost). Используем перекрестную проверку, чтобы найти наилучшие
параметры ансамбля.

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
from catboost import Pool, CatBoostClassifier
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
        pd.DataFrame(data,
                     columns=columns)
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
    end_mem = df.memory_usage().sum() / 1024**2
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


"""CatBoost"""

# Основные преимущества: умение работать с категориальными (номинативными)
# признаками и бОльшая точность, чем LighGBM

# Алгоритм запускается сразу на всех ядрах процессора, это существенно ускоряет
# работу.

# В качестве ансамблирования выберем метод опорных векторов (MVS), он ранее
# показал хорошую точность (и для CatBoost он тоже повышает точность на
# рассматриваемых данных).

x = pd.DataFrame(data_train, columns=columns_transformed)
train_dataset = Pool(data=x, label=data_train["Response"])
model = CatBoostClassifier(
    iterations=10,
    learning_rate=0.57,
    random_seed=17,
    depth=6,
    loss_function="MultiClass",
    bootstrap_type="MVS",
    custom_metric="WKappa"
)

# Диапазон тестирования параметров модели ограничен только вычислительной
# мощностью. Для проверки модели имеет смысл провести индивидуальные
# перекрестные проверки для каждого параметра в отдельности, затем в итоговой
# проверке перепроверить самые лучшие найденные параметры с отклонением +/-10%.

# Гиперпараметры оптимизации:

# depth - максимальная глубина деревьев,
# learning_rate - скорость обучения
# l2_leaf_reg - L2 параметр регуляризации для функции стоимости

cb_params = {
    "depth": range(5, 8),
    'learning_rate': np.arange(0.56, 0.59, 0.01),
    'l2_leaf_reg': range(1, 5),
}
cb_grid = model.grid_search(
    cb_params,
    cv=5,
    X=x,
    y=data_train["Response"],
    verbose=True
)
print(cb_grid["params"])

# Выведем самые оптимальные параметры и построим итоговую модель

print(cb_grid["params"])
model = CatBoostClassifier(
    iterations=100,
    learning_rate=cb_grid["params"]["learning_rate"],
    depth=cb_grid["params"]["depth"],
    l2_leaf_reg=cb_grid["params"]["l2_leaf_reg"],
    random_seed=17,
    loss_function="MultiClass",
    bootstrap_type="MVS",
    custom_metric="WKappa"
)

model.fit(train_dataset)


"""Предсказание данных и оценка модели"""

x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test["target"] = model.predict(Pool(data=x_test))

# Кластеризация дает 0.192, kNN(100) - 0.3, лог. регрессия - 0.512/0.496,
# SVM - 0.95, реш. дерево - 0.3, случайный лес - 0.487, XGBoost - 0.536,
# градиентный бустинг - 0.56, LightGBM - 0.569

print (
    "CatBoost:",
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

print (
    "CatBoost\n",
    confusion_matrix(
        data_test["target"],
        data_test["Response"]
    )
)
