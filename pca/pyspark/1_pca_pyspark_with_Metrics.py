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
data = [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y))
        for x, y in zip(iris.data, iris.target)]
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
df = spark.createDataFrame(data, schema=columns)

# 3. Векторизация признаков
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features")
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
print("Доля объясненной дисперсии для каждой компоненты:", explained_variance)

# 7. Преобразование данных для визуализации
pca_values = df_pca.select("pcaFeatures").rdd.map(lambda row: row[0]).collect()
X_pca = np.array([list(vec) for vec in pca_values])
y = df_pca.select("label").rdd.map(lambda row: row[0]).collect()

# 8. Визуализация PCA (распределение данных в плоскости первых двух компонент)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("PCA на ирисах Фишера (PySpark)")
plt.colorbar(label="Классы")
plt.show()

# 9. Дополнительные метрики для оценки PCA

# 9.1 Накопленная объяснённая дисперсия
cumulative_variance = np.cumsum(explained_variance)
print("Накопленная объяснённая дисперсия:", cumulative_variance)

# 9.2 Ошибка восстановления (Reconstruction Error)
# Для вычисления ошибки восстановления необходимо сравнить исходные (нормализованные) признаки
# с теми, что получены после обратного преобразования из PCA-пространства.
# Сначала соберем оригинальные нормализованные признаки и PCA-признаки.
data_collected = df_pca.select("scaledFeatures", "pcaFeatures").collect()
X_scaled = np.array([np.array(row["scaledFeatures"]) for row in data_collected])
X_pca_arr = np.array([np.array(row["pcaFeatures"]) for row in data_collected])
# Получение матрицы главных компонент (каждый столбец соответствует компоненте)
W = pca_model.pc.toArray()  # размерность: (число признаков, k)
# Обратное преобразование: X_reconstructed = Z * W^T, где Z = X_pca_arr
X_reconstructed = np.dot(X_pca_arr, W.T)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print("Ошибка восстановления (Reconstruction Error):", reconstruction_error)

# 9.3 Scree Plot (график доли объясненной дисперсии)
components = np.arange(1, len(explained_variance) + 1)
plt.figure(figsize=(6, 4))
plt.plot(components, explained_variance, 'o-', label="Доля объяснённой дисперсии")
plt.plot(components, cumulative_variance, 's--', label="Накопленная объяснённая дисперсия")
plt.xlabel("Номер компоненты")
plt.ylabel("Доля дисперсии")
plt.title("Scree Plot")
plt.legend()
plt.show()

# 9.4 Критерий Кайзера (Kaiser Criterion)
# При стандартизации данных суммарная дисперсия равна числу признаков.
# Если умножить долю объяснённой дисперсии на число признаков, получится оценка собственного значения.
# Оставляем те компоненты, у которых собственное значение > 1.
num_features = len(assembler.getInputCols())
kaiser_components = [i+1 for i, ev in enumerate(explained_variance) if ev * num_features > 1]
print("Компоненты, удовлетворяющие критерию Кайзера:", kaiser_components)

# Завершение работы SparkSession
spark.stop()
