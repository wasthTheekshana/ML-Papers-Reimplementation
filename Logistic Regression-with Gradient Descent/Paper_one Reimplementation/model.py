from fontTools.misc.bezierTools import epsilon
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets


y_binary = (y['G3'] >= 10).astype(int)  # Create binary target variable


categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X.loc[:, col] = LabelEncoder().fit_transform(X[col])


scaler = StandardScaler()
X_scaled  = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled ,y_binary.values.reshape(-1,1),test_size=0.2, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y ,y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0

    for epochs in range(epochs):
        z = np.dot(X,weights) + bias
        y_pred = sigmoid(z)

        dw  = np.dot(X.T, (y_pred - y)) / n_samples
        db =  np.mean(y_pred - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias

weights,bias = train_logistic_regression(X_train, y_train)
y_pred_test = sigmoid(np.dot(X_test,weights) + bias)
y_pred_class = (y_pred_test >= 0.5).astype(int)

print("Accuracy: ", accuracy_score(y_test, y_pred_class))
print(y_test)
print(y_pred_class)
print("Confusion Matrix :",confusion_matrix(y_test, y_pred_class))

print("\nClassification Report:\n", classification_report(y_test, y_pred_class))
