from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Инициализация Spark
spark = SparkSession.builder \
    .appName("IrisDecisionTree") \
    .getOrCreate()

# Загрузка данных из sklearn
from sklearn import datasets
iris = datasets.load_iris()
# Собираем данные в список кортежей: признаки и метка класса
data = [
    (float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y))
    for x, y in zip(iris.data, iris.target)
]
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]

# Создание DataFrame
df = spark.createDataFrame(data, schema=columns)

# Векторизация признаков
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)
df = assembler.transform(df)

# Нормализация данных
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)

# Разделение на обучающую и тестовую выборки
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Создание и обучение модели дерева решений с ограничением глубины (maxDepth=3)
dt_classifier = DecisionTreeClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxDepth=3,
    seed=42
)
model = dt_classifier.fit(train_data)

# Предсказание и оценка точности
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Точность модели: {accuracy:.2f}")

# Вывод структуры дерева решений (текстовое представление)
print("Структура дерева решений:")
print(model.toDebugString)

# Завершение работы Spark
spark.stop()
