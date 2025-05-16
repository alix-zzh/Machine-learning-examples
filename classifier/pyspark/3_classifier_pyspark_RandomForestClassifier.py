from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisRandomForest") \
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

# 5. Создание и обучение модели случайного леса
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=3, seed=42)
rf_model = rf.fit(train_df)

# 6. Предсказание и оценка точности
predictions = rf_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Точность модели: {accuracy:.2f}")

# 7. Визуализация важности признаков
feature_importances = rf_model.featureImportances.toArray()
feature_names = iris.feature_names

plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Важность признака")
plt.title("Важность признаков в случайном лесу")
plt.show()

# 8. Визуализация структуры одного дерева
tree_text = rf_model.trees[0].toDebugString
print("Структура первого дерева случайного леса:")
print(tree_text)

# Завершение работы SparkSession
spark.stop()
