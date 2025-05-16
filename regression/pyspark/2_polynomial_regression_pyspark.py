from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisPolynomialRegression") \
    .getOrCreate()

# 2. Загрузка данных из sklearn и создание Spark DataFrame
from sklearn import datasets
iris = datasets.load_iris()
data = [(float(x[0]), float(x[0]**2), float(x[0]**3), float(x[3])) for x in iris.data]  # Создаём полиномиальные признаки

df = spark.createDataFrame(data, ["sepal_length", "sepal_width", "petal_length", "petal_width"])

# 3. Векторизация признаков (полином 3-й степени)
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length"], outputCol="features")
df1 = assembler.transform(df).select("features", "petal_width")

# 4. Разделение данных на обучающую и тестовую выборки
train_df, test_df = df1.randomSplit([0.8, 0.2], seed=42)

# 5. Создание и обучение модели полиномиальной регрессии
lr = LinearRegression(featuresCol="features", labelCol="petal_width")
lr_model = lr.fit(train_df)

# 6. Предсказание
predictions = lr_model.transform(test_df)

# 7. Вывод коэффициентов модели
print("Коэффициенты модели:", lr_model.coefficients)
print("Свободный член:", lr_model.intercept)


# Завершение работы SparkSession
spark.stop()
