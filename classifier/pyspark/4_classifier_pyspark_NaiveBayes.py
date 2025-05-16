from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisNaiveBayes") \
    .getOrCreate()

# 2. Загрузка данных из sklearn и создание Spark DataFrame
from sklearn import datasets
iris = datasets.load_iris()
data = [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y)) for x, y in zip(iris.data, iris.target)]
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
df = spark.createDataFrame(data, schema=columns)

# 3. Векторизация признаков
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
df = assembler.transform(df)

# 4. Разделение данных на обучающую и тестовую выборки
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 5. Создание и обучение модели Наивного Байеса
nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
nb_model = nb.fit(train_df)

# 6. Предсказание на тестовой выборке и оценка точности
predictions = nb_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Точность модели: {accuracy:.2f}")

# 7. Вывод параметров модели (априорные вероятности классов)
print("Априорные вероятности классов:", nb_model.pi)
print("Логарифмы условных вероятностей признаков по классам:")
print(nb_model.theta)

# Завершение работы SparkSession
spark.stop()
