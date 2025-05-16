from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Загрузка данных
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели наивного Байеса
nb = GaussianNB()
nb.fit(X_train, y_train)

# Предсказание и оценка точности
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Точность модели: {accuracy:.2f}')

# Вывод параметров модели
print("\nСредние значения признаков для каждого класса:")
print(nb.theta_)

