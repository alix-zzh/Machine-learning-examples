import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Загрузка датасета "Ирисы Фишера"
iris = datasets.load_iris()
X = iris.data       # 4-мерное пространство характеристик
y = iris.target     # метки классов (для справки)

# Стандартизация данных (важно для корректного измерения расстояний)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ----------------------------
# 1. Построение дендрограммы
# ----------------------------
# Для дендрограммы используем scipy.cluster.hierarchy.linkage с методом Ward,
# который минимизирует внутрикластерную сумму квадратов расстояний.
Z = sch.linkage(X_std, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Дендограмма для ирисов Фишера")
dendrogram = sch.dendrogram(Z, labels=iris.target)
plt.xlabel("Объекты")
plt.ylabel("Евклидово расстояние")
plt.show()

# -----------------------------------------------------------
# 2. Кластеризация с использованием AgglomerativeClustering
# -----------------------------------------------------------
# Здесь используется агломеративная иерархическая кластеризация с 3 кластерами
# (поскольку в датасете 3 вида ирисов), с евклидовым расстоянием и linkage='ward'
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters = agg.fit_predict(X_std)
print("Назначенные метки кластеров:", clusters)

# -----------------------------------------------------------------------------
# 3. Визуализация результатов кластеризации на двумерной проекции (с помощью PCA)
# -----------------------------------------------------------------------------
# Поскольку исходные данные имеют 4 измерения, используем PCA для проекции в 2D.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=50)
plt.title("Иерархическая кластеризация (Agglomerative) на ирисах Фишера (PCA проекция)")
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.show()
