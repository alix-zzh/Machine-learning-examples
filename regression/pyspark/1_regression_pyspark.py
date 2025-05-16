from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisLinearRegression") \
    .getOrCreate()

# 2. Загрузка данных из sklearn и создание Spark DataFrame
from sklearn import datasets
iris = datasets.load_iris()
data = [(float(x[0]), float(x[1]), float(x[2]), float(x[3])) for x in iris.data]
df = spark.createDataFrame(data, ["sepal_length", "sepal_width", "petal_length", "petal_width"])

# 3. Векторизация признаков (используем три параметра для предсказания ширины лепестка)
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length"], outputCol="features")
df = assembler.transform(df).select("features", "petal_width")

# 4. Разделение данных на обучающую и тестовую выборки
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 5. Создание и обучение модели линейной регрессии
lr = LinearRegression(featuresCol="features", labelCol="petal_width")
lr_model = lr.fit(train_df)

# 6. Предсказание
predictions = lr_model.transform(test_df)

# 7. Вывод коэффициентов модели
print("Коэффициенты модели:", lr_model.coefficients)
print("Свободный член:", lr_model.intercept)

# 8. Преобразование данных для визуализации
actual_vs_predicted = predictions.select("petal_width", "prediction").collect()
actual_values = np.array([row["petal_width"] for row in actual_vs_predicted])
predicted_values = np.array([row["prediction"] for row in actual_vs_predicted])

# 9. Визуализация предсказанных значений
plt.figure(figsize=(8, 6))
plt.scatter(actual_values, predicted_values, color='blue', edgecolors='k')
plt.xlabel("Истинное значение ширины лепестка")
plt.ylabel("Предсказанное значение ширины лепестка")
plt.title("Линейная регрессия на ирисах Фишера (PySpark)")
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], linestyle="--", color="red")  # Линия идеального соответствия
plt.show()

# Завершение работы SparkSession
spark.stop()
