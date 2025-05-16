from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
from sklearn.metrics import  matthews_corrcoef

# Инициализация Spark
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("LogisticRegressionIris") \
    .getOrCreate()

# Загрузка данных из sklearn и создание DataFrame
from sklearn import datasets
iris = datasets.load_iris()
data = spark.createDataFrame(
    [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y)) for x, y in zip(iris.data, iris.target)],
    ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
)

# Векторизация признаков
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)
data = assembler.transform(data)

# Нормализация данных
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
data = scaler.fit(data).transform(data)

# Разделение на обучающую и тестовую выборки
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Создание и обучение модели логистической регрессии
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=100)
model = lr.fit(train_data)

# Предсказания
predictions = model.transform(test_data)

# Вычисление Accuracy, Precision, Recall, F1-score
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

# Matthews correlation coefficient (MCC)
y_true = np.array(predictions.select("label").collect()).flatten()
y_pred = np.array(predictions.select("prediction").collect()).flatten()
mcc = matthews_corrcoef(y_true, y_pred)

# Вывод метрик
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"Matthews correlation coefficient (MCC): {mcc:.4f}")


# Завершение работы Spark
spark.stop()
