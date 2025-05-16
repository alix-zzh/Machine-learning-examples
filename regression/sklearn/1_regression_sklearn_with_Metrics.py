import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

# Загрузка данных
iris = datasets.load_iris()
X = iris.data[:, 0:3]  # Используем три признака (длина и ширина чашелистика, длина лепестка)
y = iris.data[:, 3]  # Предсказываем ширину лепестка

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)


# Вычисление метрик

# 1. Среднеквадратичная ошибка (MSE)
mse = mean_squared_error(y_test, y_pred)

# 2. Корень среднеквадратичной ошибки (RMSE)
rmse = np.sqrt(mse)

# 3. Средняя абсолютная ошибка (MAE)
mae = mean_absolute_error(y_test, y_pred)

# 4. Коэффициент детерминации (R²)
r2 = r2_score(y_test, y_pred)

# 5. Скорректированный R²
n_test = len(y_test)          # число наблюдений в тестовой выборке
p = X_train.shape[1]          # число признаков
adjusted_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1)

# 6. Средняя абсолютная процентная ошибка (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 7. Explained Variance Score - показывает, какая доля дисперсии целевой переменной объясняется моделью
explained_variance = explained_variance_score(y_test, y_pred)

# Вывод полученных метрик
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")
print(f"Скорректированный R²: {adjusted_r2:.4f}")
print(f"MAPE (средняя абсолютная процентная ошибка): {mape:.2f}%")
print(f"Explained Variance Score: {explained_variance:.4f}")

# Вывод коэффициентов модели
print("Коэффициенты модели:", model.coef_)
print("Свободный член:", model.intercept_)

# Визуализация предсказанных значений
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.xlabel("Истинное значение ширины лепестка")
plt.ylabel("Предсказанное значение ширины лепестка")
plt.title("Линейная регрессия на ирисах Фишера")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")  # Линия идеального соответствия
plt.show()
