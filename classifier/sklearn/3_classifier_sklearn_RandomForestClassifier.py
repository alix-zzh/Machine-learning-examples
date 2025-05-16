import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# Загрузка данных
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели случайного леса
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Предсказание и оценка точности
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Точность модели: {accuracy:.2f}')

# Визуализация важности признаков
importances = rf.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Важность признака')
plt.title('Важность признаков в случайном лесу')
plt.show()

# Визуализация одного из деревьев
plt.figure(figsize=(12, 8))
plot_tree(rf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
