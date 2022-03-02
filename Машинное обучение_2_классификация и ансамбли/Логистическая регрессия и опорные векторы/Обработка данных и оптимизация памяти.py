"""
Постановка задачи
Загрузим данные и подготовим все данные для анализа: проведем нормализацию и
преобразование категорий. Оптимизируем потребление памяти.

Разделим выборку на обучающую/проверочную в соотношении 80/20.

Применим наивный Байес для классификации скоринга. Будем использовать все
возможные столбцы.

Проверим качество предсказания через каппа-метрику и матрицу неточностей.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

Соревнование: https://www.kaggle.com/c/prudential-life-insurance-assessment/

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
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


"""Категоризация данных"""

data["Product_Info_2_1"] = data["Product_Info_2"].str.slice(0, 1)
data["Product_Info_2_2"] = pd.to_numeric(data["Product_Info_2"].str.slice(1, 2))
data.drop(labels=["Product_Info_2"], axis=1, inplace=True)
print(data.info())


"""Оптимизация потребления памяти"""


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
    print("Потребление памяти меньше на", round(start_mem - end_mem, 2),
          "Мб (минус)", round(100 * (start_mem - end_mem) / start_mem, 1), "%)")
    return df


data = reduce_mem_usage(data)
print(data.info())


"""Предобработка: категоризация, единичные векторы"""

#
# Product
#    A
#    B
#    C
#    A
#
# Переходит в
#
# ProductA  ProductB  ProductC
#     1         0         0
#     0         1         0
#     0         0         1
#     1         0         0

# Можно использовать sklearn.preprocessing.OneHotEncoder, но для этого
# потребуется дополнительно преобразовать фрейм данных (набор единичных векторов
# для каждого кортежа данных).
#
# Также не будем использовать кодирование категорий
# (A->1, B->2, C->3, D->4, E->5), потому что это переводит номинативную
# случайную величину в ранговую/числовую, и является существенным допущением
# относительно исходных данных.

for l in data["Product_Info_2_1"].unique():
    data["Product_Info_2_1" + l] = data["Product_Info_2_1"].isin([l]).astype("int8")
data.drop(labels=["Product_Info_2_1"], axis=1, inplace=True)


"""Заполним отсутствующие значения"""

# -1 увеличивает "расстояние" при расчете ближайших соседей

data.fillna(value=-1, inplace=True)


"""Столбцы для модели"""

columns_groups = [
    "Insurance_History",
    "InsuredInfo",
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

# Дополнительно проведем z-нормализацию данных через предварительную обработку
# (preprocessing). Нормализуем весь исходный набор данных.

scaler = preprocessing.StandardScaler()
scaler.fit(pd.DataFrame(data, columns=columns))


"""Разделение данных"""

data_train, data_test = train_test_split(data, test_size=0.2)
print(data_train.head())


"""Расчет модели наивного Байеса"""

# 𝑃(𝐴∣𝐵)=(𝑃(𝐵∣𝐴)*𝑃(𝐴))/(𝑃(𝐵))
#
# Для каждого параметра вычисляется его вероятность принять определенное
# значение - P(B). Для каждого класса вычисляется его вероятность
# (по факту, доля) - P(A). Затем вычисляется вероятность для каждого параметра
# принять определенное значение при определенном классе - P(B\A).
#
# По всем вычисленным значениям находится вероятность при известных параметрах
# принять какое-либо значение класса.

y = data_train["Response"]
x = scaler.transform(pd.DataFrame(data_train, columns=columns))

bayes = GaussianNB()
bayes.fit(x, y)


"""Предсказание данных"""

data_test = pd.DataFrame(data_test)
x_test = scaler.transform(pd.DataFrame(data_test, columns=columns))
data_test["target"] = bayes.predict(x_test)
print(data_test.head())


"""Оценка модели"""

print("Байес:", cohen_kappa_score(
    data_test["target"],
    data_test["Response"],
    weights="quadratic"
))


"""Матрица неточностей"""

print(confusion_matrix(data_test["target"], data_test["Response"]))
