import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fashion_data = tf.keras.datasets.fashion_mnist
np.random.seed(42)

input_size = 784
hidden_size = 128
output_size = 10


(X_train, y_train), (X_test, y_test) =fashion_data.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

def one_hot_encode(y,num_classes=10):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

def initialize_parameters(inout_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

print(f"W1 shpe: {W1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"b2 shape: {b2.shape}")

def Relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expz = np.exp(Z-np.max(Z, axis=1, keepdims=True))
    return expz /expz.sum(axis=1, keepdims=True)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  # (n_samples, hidden_size)
    A1 = Relu(Z1)
    Z2 = np.dot(A1, W2) + b2  # (n_samples, output_size)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Select 5 samples (shape: (5, 784))
Z1, A1, Z2, A2 = forward_propagation(X_train[:5], W1, b1, W2, b2)
print("A2 shape:", A2.shape)  # should be (5, 10)
print("Probabilities sum per sample:", np.sum(A2, axis=1))  # should be close to 1

def compute_loss(A2,y):
    m= y.shape[1]
    log_probs = np.log(A2 + 1e-8)
    loss = -np.sum(y * log_probs) / m
    return loss

def back_propagation(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = back_propagation(
    X_train[:5], y_train[:5], Z1, A1, Z2, A2, W2
)

print("dW1 shape:", dW1.shape)
print("dW2 shape:", dW2.shape)


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

learning_rate = 0.1
W1, b1, W2, b2 = update_parameters(
    W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate
)
print("Updated W1[0,0]:", W1[0,0])


def get_accurancy(A2, y):
    predictions = np.argmax(A2, axis=0)
    labels = np.argmax(y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

def model(X_train, y_train, X_test, y_test, hidden_size=128, learning_rate = 0.1,epoch=20):
    input_size = X_train.shape[0]
    output_size = y_train.shape[0]

    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epochs in  range(epoch):
        Z1, A1,Z2,A2 = forward_propagation(X_train,W1, b1, W2,b2)

        loss = compute_loss(A2, y_train)

        dW1, db1, dW2, db2 = back_propagation(X_train, y_train, Z1, A1, Z2, A2, W2)

        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 2 ==0:
            train_acc = get_accurancy(A2, y_train)
            Z1_test,A1_test, Z2_test, A2_test = forward_propagation(X_test, W1, b1, W2, b2)
            test_acc = get_accurancy(A2_test, y_test)
            print(F"Epoch {epoch}: Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return W1, b1, W2, b2

W1, b1, W2, b2 = model(X_train, y_train, X_test, y_test, hidden_size=128, learning_rate=0.1, epoch=20)


