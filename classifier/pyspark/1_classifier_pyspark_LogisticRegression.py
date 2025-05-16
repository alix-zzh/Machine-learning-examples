from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Инициализация Spark

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("LogisticRegressionIris") \
    .getOrCreate()


# Загрузка данных
from sklearn import datasets
iris = datasets.load_iris()
data = spark.createDataFrame(
    [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y)) for x, y in zip(iris.data, iris.target)],
    ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
)

# Векторизация признаков
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
data = assembler.transform(data)

# Нормализация данных
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
data = scaler.fit(data).transform(data)

# Разделение на обучающую и тестовую выборки
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Создание и обучение модели логистической регрессии
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=100)
model = lr.fit(train_data)

# Предсказание
predictions = model.transform(test_data)

# Оценка точности
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f'Точность модели: {accuracy:.2f}')

# Вывод параметров модели
print("Коэффициенты:", model.coefficientMatrix)
print("Свободный член:", model.interceptVector)

# Завершение работы Spark
spark.stop()
