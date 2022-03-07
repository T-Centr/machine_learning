"""
Постановка задачи

Загрузим данные, приведем их к числовым, заполним пропуски, нормализуем данные и
оптимизируем память.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Построим последовательный ансамбль решающих деревьев, используя облегченный
градиентный бустинг (LightGBM). Используем перекрестную проверку, чтобы найти
наилучшие параметры ансамбля.

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
import lightgbm as lgb
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

# Дополнительно преобразуем значение класса: начнем его с нуля.

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


"""LightGBM"""

# Основное отличие этого градиентного бустинга от предыдущих - использование
# сильно-разнородных (определяется разностью, гистограммой самих данных)
# экземпляров в выборке для формирования первоначального дерева: сначала
# рассматриваются все крайние, "плохие", случаи, а затем к ним "достраиваются"
# средние, "хорошие". Это позволяет еще быстрее минимизировать ошибку моделей.

# Из дополнительных плюсов: алгоритм запускается сразу на всех ядрах процессора,
# это существенно ускоряет работу.

x = pd.DataFrame(data_train, columns=columns_transformed)
model = lgb.LGBMRegressor(
    random_state=17,
    max_depth=18,
    min_child_samples=19,
    num_leaves=34
)

# Также возможно провести классификации множества классов через LightGBM.
# В этом случае модель вернет вероятности принадлежности к каждому классу,
# возвращенные значения нужно будет дополнительно обработать через argmax,
# чтобы получить единственное значение класса.

# model = lgb.LGBMRegressor(
#     random_state=17,
#     max_depth=17,
#     min_child_samples=18,
#     num_leaves=34,
#     objective="multiclass",
#     num_class=8
# )


# Диапазон тестирования параметров модели ограничен только вычислительной
# мощностью. Для проверки модели имеет смысл провести индивидуальные
# перекрестные проверки для каждого параметра в отдельности, затем в итоговой
# проверке перепроверить самые лучшие найденные параметры с отклонением +/-10%.
#
# Проверку качества по каппа метрике при оптимизации выполнить не удастся из-за
# нецелых значений Light GBM. Гиперпараметры оптимизации:
#
# max_depth - максимальная глубина деревьев,
# num_leaves - число листьев в каждом
# min_child_samples - минимальное число элементов выборке в листе

lgb_params = {
    'max_depth': range(16, 19),
    'num_leaves': range(34, 37),
    'min_child_samples': range(17, 20)
}
grid = GridSearchCV(model, lgb_params, cv=5, n_jobs=4, verbose=True)
grid.fit(x, data_train["Response"])


# Выведем самые оптимальные параметры и построим итоговую модель,
# используя 1000 последовательных деревьев.

print(grid.best_params_)
model = lgb.LGBMRegressor(
    random_state=17,
    max_depth=grid.best_params_['max_depth'],
    min_child_samples=grid.best_params_['min_child_samples'],
    num_leaves=grid.best_params_['num_leaves'],
    n_estimators=1000,
    objective="multiclass", num_class=8
)

model.fit(x, data_train["Response"])


"""Предсказание данных и оценка модели"""

# LightGBM возвращает дробное значение класса, его нужно округлить.
# Для multiclass используем argmax


def calculate_model(x):
    return np.argmax(model.predict([x]))


x_test = pd.DataFrame(data_test, columns=columns_transformed)
data_test["target"] = x_test.apply(calculate_model, axis=1, result_type="expand")


# Кластеризация дает 0.192, kNN(100) - 0.3, лог. регрессия - 0.512/0.496,
# SVM - 0.95, реш. дерево - 0.3, случайный лес - 0.487, XGBoost - 0.536,
# градиентный бустинг - 0.56


print(
    "LightGBM:",
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

print(
    "LightGBM\n",
    confusion_matrix(
        data_test["target"],
        data_test["Response"]
    )
)
