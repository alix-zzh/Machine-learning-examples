from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
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

# 8. Расчёт метрик
# Используем RegressionEvaluator из PySpark для основных метрик
evaluator_rmse = RegressionEvaluator(labelCol="petal_width", predictionCol="prediction", metricName="rmse")
evaluator_mse = RegressionEvaluator(labelCol="petal_width", predictionCol="prediction", metricName="mse")
evaluator_mae = RegressionEvaluator(labelCol="petal_width", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="petal_width", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mse = evaluator_mse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# Для вычисления скорректированного R^2, MAPE и Explained Variance Score нужно собрать данные в Python
pred_data = predictions.select("petal_width", "prediction").collect()
y_true = np.array([row["petal_width"] for row in pred_data])
y_pred = np.array([row["prediction"] for row in pred_data])

# Число наблюдений в тестовой выборке и число признаков (p = 3)
n_test = len(y_true)
p = 3
adjusted_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1) if n_test > p + 1 else None

# Средняя абсолютная процентная ошибка (MAPE)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Explained Variance Score: 1 - Var(y_true - y_pred) / Var(y_true)
explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)

# 9. Вывод полученных метрик
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Корень из MSE (RMSE): {rmse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")
if adjusted_r2 is not None:
    print(f"Скорректированный R²: {adjusted_r2:.4f}")
else:
    print("Скорректированный R²: не может быть вычислен (недостаточно наблюдений)")
print(f"MAPE (средняя абсолютная процентная ошибка): {mape:.2f}%")
print(f"Explained Variance Score: {explained_variance:.4f}")

# 10. Преобразование данных для визуализации
actual_vs_predicted = predictions.select("petal_width", "prediction").collect()
actual_values = np.array([row["petal_width"] for row in actual_vs_predicted])
predicted_values = np.array([row["prediction"] for row in actual_vs_predicted])

# 11. Визуализация предсказанных значений
plt.figure(figsize=(8, 6))
plt.scatter(actual_values, predicted_values, color='blue', edgecolors='k')
plt.xlabel("Истинное значение ширины лепестка")
plt.ylabel("Предсказанное значение ширины лепестка")
plt.title("Линейная регрессия на ирисах Фишера (PySpark)")
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], linestyle="--", color="red")  # Линия идеального соответствия
plt.show()

# Завершение работы SparkSession
spark.stop()
