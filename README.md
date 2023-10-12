# SRO6.1
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Загрузка датасета
data = load_breast_cancer()
X = data.data
y = data.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Количество деревьев в ансамбле
M = 100

# Создание и обучение модели Random Forest
def train_random_forest(X, y, M):
    forest = []
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    for _ in range(M):
        # Bootstrap sampling
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        # Random feature selection
        num_selected_features = int(np.sqrt(n_features))  # Square root of total features
        selected_features = np.random.choice(n_features, num_selected_features, replace=False)
        X_subset = X_bootstrap[:, selected_features]
        
        # Train a decision tree on the selected subset
        tree = DecisionTreeClassifier()
        tree.fit(X_subset, y_bootstrap)
        forest.append((tree, selected_features))
    
    return forest

# Predict using the Random Forest
def predict_random_forest(forest, X):
    predictions = np.zeros((X.shape[0], len(forest)))
    for i, (tree, selected_features) in enumerate(forest):
        X_subset = X[:, selected_features]
        predictions[:, i] = tree.predict(X_subset)
    return np.mean(predictions, axis=1).round().astype(int)

# Обучение Random Forest
forest = train_random_forest(X_train, y_train, M)

# Прогнозирование на тестовом наборе
y_pred = predict_random_forest(forest, X_test)

# Оценка модели
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
