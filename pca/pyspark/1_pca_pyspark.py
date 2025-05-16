from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisPCA") \
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

# 4. Нормализация данных
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# 5. Применение PCA (снижаем размерность до 2 компонент)
pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca.fit(df)
df_pca = pca_model.transform(df)

# 6. Вывод доли объяснённой дисперсии
explained_variance = pca_model.explainedVariance.toArray()
print("Доля объясненной дисперсии:", explained_variance)

# 7. Преобразование данных для визуализации
pca_values = df_pca.select("pcaFeatures").rdd.map(lambda row: row[0]).collect()
X_pca = [list(vec) for vec in pca_values]
X_pca = np.array(X_pca)

y = df_pca.select("label").rdd.map(lambda row: row[0]).collect()

# 8. Визуализация PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("PCA на ирисах Фишера (PySpark)")
plt.colorbar(label="Классы")
plt.show()

# Завершение работы SparkSession
spark.stop()
