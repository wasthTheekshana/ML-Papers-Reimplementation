import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fashion_data = tf.keras.datasets.fashion_mnist

X_train, X_test, y_train, y_test =fashion_data.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.rehsape(-1, 28*28)

def one_hot_encode(y,num_classes=10):
    one_hot_y = np.zeros((y.shape[0],num_classes))
    one_hot_y[y,  np.arange(y.shape[0])] = 1