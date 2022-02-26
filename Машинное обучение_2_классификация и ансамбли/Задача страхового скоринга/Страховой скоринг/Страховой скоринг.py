"""
Постановка задачи
Задача страхового скоринга:
https://www.kaggle.com/c/prudential-life-insurance-assessment

Требуется провести классификацию клиентов по уровню благонадежности для
страхования жизни (всего 8 градаций) - Response. Для оценки доступно несколько
параметров: виды страховки (Product_Info), возраст (Ins_Age), рост (Ht),
вес (Wt), индекс массы тела (BMI), данные о работе (Employment_Info),
данные страховки (InsuredInfo), история страхования (Insurance_History),
семья (Family_Hist), медицинские данные (Medical_History) и
медицинские термины (Medical_Keyword) - всего 126 переменных.

Загрузим данные и исследуем их. Найдем возможные "утечки" и взаимосвязи
параметров для построения моделей.

Данные:

https://video.ittensive.com/machine-learning/prudential/train.csv.gz

© ITtensive, 2020
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import seaborn as sns
rcParams['figure.figsize'] = 16, 8


pd.set_option('display.max_rows', None)  # Сброс ограничений на количество
# выводимых рядов

pd.set_option('display.max_columns', None)  # Сброс ограничений на число
# выводимых столбцов

pd.set_option('display.max_colwidth', None)  # Сброс ограничений на количество
# символов в записи


"""Загрузка данных"""

train = pd.read_csv(
    "https://video.ittensive.com/machine-learning/prudential/train.csv.gz"
)
print(train.info())
print(train.head())


"""Распределение значений"""

train.hist(figsize=(16, 24), layout=(16, 8))
plt.show()


"""Зависимость скоринга от параметров: история страхования"""


def data_correlation_plot(df, columns):
    rows = np.ceil(len(columns) / 4)
    fig = plt.figure(figsize=(12, rows*3))
    i = 1
    for column in columns:
        type_ = str(df[column].dtypes)
        if type_[0:3] == "int" or type_[0:5] == "float":
            area = fig.add_subplot(rows, 4, i)
            pd.DataFrame(
                df,
                columns=["Response", column]
            ).plot.scatter(x=column,
                           y="Response",
                           ax=area
                           )
            i += 1
    plt.show()


data_correlation_plot(
    train, train.columns[train.columns.str.startswith("Insurance_History")]
)


"""Зависимость скоринга от параметров: параметры страхования"""

data_correlation_plot(
    train, train.columns[train.columns.str.startswith("InsuredInfo")]
)


"""Зависимость скоринга от параметров: физиология"""

data_correlation_plot(train, ["Wt", "Ht", "Ins_Age", "BMI"])


"""Зависимость скоринга от параметров: семья"""

data_correlation_plot(
    train, train.columns[train.columns.str.startswith("Family_Hist")]
)


"""Зависимость скоринга от параметров: здоровье"""

data_correlation_plot(
    train, train.columns[train.columns.str.startswith("Medical_History")]
)


"""Зависимость скоринга от параметров: страховка"""

train["Product_Info_2_1"] = train["Product_Info_2"].str.slice(0, 1)
train["Product_Info_2_2"] = train[
    "Product_Info_2"].str.slice(1, 2).astype("int8")
train.drop(labels=["Product_Info_2"], axis=1, inplace=True)

data_correlation_plot(
    train, train.columns[train.columns.str.startswith("Product_Info")]
)


"""Взаимная корреляция биометрии"""

data = pd.DataFrame(
    train[train["Response"] == 1], columns=["Wt", "Ht", "Ins_Age", "BMI"]
)
sns.pairplot(data, height=4)
plt.show()
del data


"""Кластеризация по биометрии"""

columns_groups = [
    ["Wt", "Ht"],
    ["Wt", "Ins_Age"],
    ["Wt", "BMI"],
    ["Ht", "Ins_Age"],
    ["Ht", "BMI"],
    ["Ins_Age", "BMI"]
]
colors = ["#A62639", "#DB324D", "#56494E", "#A29C9B", "#511C29", "#0000FF",
          "#FF00FF", "#FFFF00", "#00FFFF", "#00FF00"]
fig = plt.figure(figsize=(18, 12))
i = 1
for c in columns_groups:
    data = pd.DataFrame(train, columns=c.append("Response"))
    legend = []
    area = fig.add_subplot(2, 3, i)
    for response in range(1, train["Response"].max() + 1):
        group = data[data["Response"] == response].plot.scatter(
            x=c[0],
            y=c[1],
            ax=area,
            c=colors[response-1]
        )
        legend.append(response)
    area.legend(legend)
    i += 1
    del data
plt.show()
