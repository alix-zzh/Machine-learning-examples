import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Загрузка данных
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Применение метода главных компонент
pca = PCA(n_components=2)  # Снижаем размерность до 2 признаков
X_pca = pca.fit_transform(X_scaled)

# Визуализация данных
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("PCA на ирисах Фишера")
plt.colorbar(label="Классы")
plt.show()

# Вывод объясненной дисперсии
print("Доля объясненной дисперсии:", pca.explained_variance_ratio_)
