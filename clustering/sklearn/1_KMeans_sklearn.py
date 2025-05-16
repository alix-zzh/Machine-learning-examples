import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Загрузка набора данных "Ирисы Фишера"
iris = datasets.load_iris()
X = iris.data       # Признаки: 4-мерное пространство
y = iris.target     # Исходные метки (для справки)

# Стандартизация данных для повышения устойчивости алгоритма
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Применение алгоритма K-Means для кластеризации
# Выбираем 3 кластера, так как в датасете 3 вида ирисов
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

# Для визуализации уменьшаем размерность до 2-х с помощью метода главных компонент (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Визуализация результатов кластеризации
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", edgecolors="k", s=50)
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("Кластеризация K-Means (Ирисы Фишера)")

# Отображение центроидов кластеров
centers = kmeans.cluster_centers_            # Центроиды в стандартизированном пространстве
centers_pca = pca.transform(centers)           # Преобразуем центроиды в PCA-пространство
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker="X", label="Центроиды")
plt.legend()
plt.show()
