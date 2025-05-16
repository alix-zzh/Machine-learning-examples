import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
