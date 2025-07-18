import numpy as np
from fontTools.misc.bezierTools import epsilon
from numpy.matlib import randn

np.random.seed(42)

X= np.random.randn(100, 2)

true_weight = np.array([1,1])

y = (X @ true_weight + np.random.randn(100) > 0).astype(int)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_loss(Y,y_hat):
    m = Y.shape[0]
    epsilon = 1e-15
    loss = -1/m * np.sum(Y * np.log(y_hat + epsilon) + (1-Y) *(np.log(1-y_hat + epsilon)))
    return loss

def train_logistic_regration(X,y, lr=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for epochs in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)

        w -= lr *dw
        b -= lr * db

        if epochs % 100 == 0:
            loss =  compute_loss(y,y_hat)
            print("epochs: {}, loss: {}".format(epochs, loss))
    return w, b


weights, bias  = train_logistic_regration(X, y, lr=0.01, epochs=1000)
print("weights: {}".format(weights))
print("bias: {}".format(bias))


def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_hat = sigmoid(z)
    return (y_hat > 0.5).astype(int)

preds  = predict(X, weights, bias)
accurancy = np.mean(preds == y)
print("accuracy: {}".format(accurancy))

