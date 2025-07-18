# Use Iris Data set in sklearn

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource

from main import train_logistic_regration, predict

#Load the iris dataset

iris = load_iris()
X = iris.data
y = iris.target

# Convert to binary classification problem (0 vs 1)
y_binary = (y == 0).astype(int)

X = X[:, :2]  # Use only the first two features for simplicity

X_train, X_test, y_train, y_test = train_test_split(X,y_binary, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

weights, bias  = train_logistic_regration(X_train, y_train, lr=0.01, epochs=1000)
print("weights: {}".format(weights))
print("bias: {}".format(bias))

y_pred = predict(X_test, weights, bias)
accurancy = np.mean(y_pred == y_test)

print("Accuracy: {}".format(accurancy))


# Plotting decision boundary
def plot_decision_boundary(X,y,weights, bias,title):

    x_min, x_max = X[:,0].min()- 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))

    grid  = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid, weights, bias)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=[8,6])
    cmap_light = ListedColormap(["#FFAAAA","#AAFFAA"])
    cmap_bold  = ListedColormap(["#FF0000","#00AA00"])

    plt.contour(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolors="k")
    plt.title(title)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.grid(True)
    plt.show()

print("X_train",X_train)
plot_decision_boundary(X_train,y_train,weights,bias,"Logistic Regression Decision Boundary (Iris - Train Set)")
plot_decision_boundary(X_test,y_test,weights,bias,"Logistic Regression Decision Boundary (Iris - Test Set)")