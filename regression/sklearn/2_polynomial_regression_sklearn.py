import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Загрузка данных
iris = datasets.load_iris()
X = iris.data[:, 0:1]  # Используем только длину чашелистика для простоты визуализации
y = iris.data[:, 3]  # Предсказываем ширину лепестка

# Преобразование данных в полиномиальные признаки
poly = PolynomialFeatures(degree=3)  # Полином третьей степени
X_poly = poly.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Визуализация полиномиальной регрессии
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Исходные данные")
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
plt.plot(X_range, model.predict(X_range_poly), color='red', linewidth=2, label="Полиномиальная регрессия")
plt.xlabel("Длина чашелистика")
plt.ylabel("Ширина лепестка")
plt.title("Полиномиальная регрессия на ирисах Фишера")
plt.legend()
plt.show()

# Вывод коэффициентов модели
print("Коэффициенты модели:", model.coef_)
print("Свободный член:", model.intercept_)
