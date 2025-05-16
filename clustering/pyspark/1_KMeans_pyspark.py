from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from sklearn import datasets

# 1. Инициализация SparkSession
spark = SparkSession.builder \
    .appName("IrisKMeans") \
    .getOrCreate()

# 2. Загрузка данных из sklearn и создание Spark DataFrame
iris = datasets.load_iris()
# Преобразуем данные в список кортежей, содержащий 4 признака
data = [(float(x[0]), float(x[1]), float(x[2]), float(x[3])) for x in iris.data]
# Определяем имена столбцов
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# Создаем DataFrame
df = spark.createDataFrame(data, columns)

# 3. Векторизация признаков
# Объединяем все 4 признака в один вектор, который требуется для методов MLlib
assembler = VectorAssembler(inputCols=columns, outputCol="features")
df_feats = assembler.transform(df)

# 4. Применение K-Means
# Устанавливаем число кластеров k=3 и задаем seed для воспроизводимости
kmeans = KMeans(featuresCol="features", k=3, seed=42)
model = kmeans.fit(df_feats)

# 5. Получение предсказанных меток кластеров
predictions = model.transform(df_feats)

# Вывести центроиды кластеров
centers = model.clusterCenters()
print("Центроиды кластеров:")
for center in centers:
    print(center)

# Вывод первых 10 строк с признаками и присвоенными метками кластеров
predictions.select("features", "prediction").show(10, truncate=False)

# Завершение работы SparkSession
spark.stop()
