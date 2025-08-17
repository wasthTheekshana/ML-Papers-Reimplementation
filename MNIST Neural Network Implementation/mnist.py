import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


np.random.seed(42)
tf.random.set_seed(42)


print("Loading Fashion-MNIST dataset...")
fashion_data = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_data.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")


X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(f"Reshaped training data: {X_train.shape}")
print(f"Reshaped test data: {X_test.shape}")

# One-hot encode labels
def one_hot_encode(y, num_classes=10):

    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y

y_train_onehot = one_hot_encode(y_train, 10)
y_test_onehot = one_hot_encode(y_test, 10)

print(f"One-hot training labels shape: {y_train_onehot.shape}")
print(f"One-hot test labels shape: {y_test_onehot.shape}")

# Network architecture parameters
input_size = 784
hidden_size = 128
output_size = 10

# Initialize parameters with Xavier initialization
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Activation functions
def relu(Z):

    return np.maximum(0, Z)

def relu_derivative(Z):

    return (Z > 0).astype(float)

def softmax(Z):

    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)


    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    cache = {
        'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2
    }
    return A2, cache


def compute_loss(A2, y, W1=None, W2=None, lambda_reg=0.0):

    m = y.shape[0]

    # Cross-entropy loss
    log_probs = np.log(A2 + 1e-8)
    cross_entropy_loss = -np.sum(y * log_probs) / m

    # L2 regularization
    if lambda_reg > 0 and W1 is not None and W2 is not None:
        l2_regularization = lambda_reg * (np.sum(W1**2) + np.sum(W2**2)) / (2 * m)
        return cross_entropy_loss + l2_regularization

    return cross_entropy_loss

# Backward propagation
def backward_propagation(X, y, cache, W1, W2, lambda_reg=0.0):

    m = X.shape[0]
    Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']

    # Output layer gradients
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m


    if lambda_reg > 0:
        dW1 += lambda_reg * W1 / m
        dW2 += lambda_reg * W2 / m

    return dW1, db1, dW2, db2

# Parameter update functions
def update_parameters_sgd(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

def update_parameters_momentum(W1, b1, W2, b2, dW1, db1, dW2, db2,
                              vW1, vb1, vW2, vb2, learning_rate, beta=0.9):


    vW1 = beta * vW1 + (1 - beta) * dW1
    vb1 = beta * vb1 + (1 - beta) * db1
    vW2 = beta * vW2 + (1 - beta) * dW2
    vb2 = beta * vb2 + (1 - beta) * db2


    W1 -= learning_rate * vW1
    b1 -= learning_rate * vb1
    W2 -= learning_rate * vW2
    b2 -= learning_rate * vb2

    return W1, b1, W2, b2, vW1, vb1, vW2, vb2

def update_parameters_adam(W1, b1, W2, b2, dW1, db1, dW2, db2,
                          mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2, t,
                          learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):

    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    mb2 = beta1 * mb2 + (1 - beta1) * db2


    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)


    mW1_corrected = mW1 / (1 - beta1 ** t)
    mb1_corrected = mb1 / (1 - beta1 ** t)
    mW2_corrected = mW2 / (1 - beta1 ** t)
    mb2_corrected = mb2 / (1 - beta1 ** t)


    vW1_corrected = vW1 / (1 - beta2 ** t)
    vb1_corrected = vb1 / (1 - beta2 ** t)
    vW2_corrected = vW2 / (1 - beta2 ** t)
    vb2_corrected = vb2 / (1 - beta2 ** t)


    W1 -= learning_rate * mW1_corrected / (np.sqrt(vW1_corrected) + epsilon)
    b1 -= learning_rate * mb1_corrected / (np.sqrt(vb1_corrected) + epsilon)
    W2 -= learning_rate * mW2_corrected / (np.sqrt(vW2_corrected) + epsilon)
    b2 -= learning_rate * mb2_corrected / (np.sqrt(vb2_corrected) + epsilon)

    return W1, b1, W2, b2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2


def get_accuracy(predictions, labels):

    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    return np.mean(pred_classes == true_classes)

def create_mini_batches(X, y, batch_size):

    m = X.shape[0]
    mini_batches = []

    # Shuffle the data
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]

    # Create mini-batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size]
        mini_batch_y = shuffled_y[k * batch_size:(k + 1) * batch_size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)


    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_batches * batch_size:]
        mini_batch_y = shuffled_y[num_complete_batches * batch_size:]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches

# Main training function
def train_neural_network(X_train, y_train, X_test, y_test,
                        hidden_size=128, learning_rate=0.01, epochs=50,
                        batch_size=128, optimizer='adam', lambda_reg=0.001,
                        print_cost=True):



    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)


    if optimizer == 'momentum':
        vW1, vb1, vW2, vb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
    elif optimizer == 'adam':
        mW1, mb1, mW2, mb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
        vW1, vb1, vW2, vb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
        t = 0


    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"Training neural network with {optimizer} optimizer...")
    print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {epochs}")
    print("-" * 60)

    for epoch in range(epochs):
        epoch_loss = 0
        mini_batches = create_mini_batches(X_train, y_train, batch_size)

        if optimizer == 'adam':
            t += 1

        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_y = mini_batch

            # Forward propagation
            A2, cache = forward_propagation(mini_batch_X, W1, b1, W2, b2)

            # Compute loss
            loss = compute_loss(A2, mini_batch_y, W1, W2, lambda_reg)
            epoch_loss += loss

            # Backward propagation
            dW1, db1, dW2, db2 = backward_propagation(mini_batch_X, mini_batch_y, cache, W1, W2, lambda_reg)


            if optimizer == 'sgd':
                W1, b1, W2, b2 = update_parameters_sgd(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
            elif optimizer == 'momentum':
                W1, b1, W2, b2, vW1, vb1, vW2, vb2 = update_parameters_momentum(
                    W1, b1, W2, b2, dW1, db1, dW2, db2, vW1, vb1, vW2, vb2, learning_rate)
            elif optimizer == 'adam':
                W1, b1, W2, b2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2 = update_parameters_adam(
                    W1, b1, W2, b2, dW1, db1, dW2, db2, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2, t, learning_rate)


        avg_epoch_loss = epoch_loss / len(mini_batches)
        train_losses.append(avg_epoch_loss)


        if epoch % 5 == 0 or epoch == epochs - 1:
            train_pred, _ = forward_propagation(X_train, W1, b1, W2, b2)
            train_acc = get_accuracy(train_pred, y_train)
            train_accuracies.append(train_acc)

            test_pred, _ = forward_propagation(X_test, W1, b1, W2, b2)
            test_acc = get_accuracy(test_pred, y_test)
            test_accuracies.append(test_acc)

            if print_cost:
                print(f"Epoch {epoch:3d}: Loss = {avg_epoch_loss:.4f}, "
                      f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters, train_losses, train_accuracies, test_accuracies


def plot_training_history(train_losses, train_accuracies, test_accuracies, epochs, optimizer_name):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    ax1.plot(train_losses)
    ax1.set_title(f'{optimizer_name} - Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True)


    acc_epochs = list(range(0, epochs, 5)) + ([epochs-1] if epochs-1 not in range(0, epochs, 5) else [])
    ax2.plot(acc_epochs[:len(train_accuracies)], train_accuracies, 'b-', label='Train Accuracy')
    ax2.plot(acc_epochs[:len(test_accuracies)], test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title(f'{optimizer_name} - Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_sample_predictions(X_test, y_test, parameters, class_names, num_samples=10):

    predictions, _ = forward_propagation(X_test[:num_samples], parameters['W1'], parameters['b1'],
                                       parameters['W2'], parameters['b2'])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:num_samples]


    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):

        img = X_test[i].reshape(28, 28)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {class_names[true_classes[i]]}\n'
                         f'Pred: {class_names[pred_classes[i]]}',
                         fontsize=10)
        axes[i].axis('off')


        if pred_classes[i] == true_classes[i]:
            axes[i].patch.set_edgecolor('green')
        else:
            axes[i].patch.set_edgecolor('red')
        axes[i].patch.set_linewidth(2)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(X_test, y_test, parameters, class_names):

    predictions, _ = forward_propagation(X_test, parameters['W1'], parameters['b1'],
                                       parameters['W2'], parameters['b2'])
    pred_classes = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


print("=" * 80)
print("TRAINING WITH DIFFERENT OPTIMIZERS")
print("=" * 80)


print("\n1. Training with SGD...")
params_sgd, losses_sgd, train_acc_sgd, test_acc_sgd = train_neural_network(
    X_train, y_train_onehot, X_test, y_test_onehot,
    hidden_size=128, learning_rate=0.1, epochs=30, batch_size=128,
    optimizer='sgd', lambda_reg=0.001
)


print("\n2. Training with Momentum...")
params_momentum, losses_momentum, train_acc_momentum, test_acc_momentum = train_neural_network(
    X_train, y_train_onehot, X_test, y_test_onehot,
    hidden_size=128, learning_rate=0.01, epochs=30, batch_size=128,
    optimizer='momentum', lambda_reg=0.001
)


print("\n3. Training with Adam...")
params_adam, losses_adam, train_acc_adam, test_acc_adam = train_neural_network(
    X_train, y_train_onehot, X_test, y_test_onehot,
    hidden_size=128, learning_rate=0.001, epochs=30, batch_size=128,
    optimizer='adam', lambda_reg=0.001
)


print("\n" + "=" * 80)
print("FINAL RESULTS COMPARISON")
print("=" * 80)
print(f"SGD - Final Test Accuracy: {test_acc_sgd[-1]:.4f}")
print(f"Momentum - Final Test Accuracy: {test_acc_momentum[-1]:.4f}")
print(f"Adam - Final Test Accuracy: {test_acc_adam[-1]:.4f}")


print("\nPlotting training histories...")
plot_training_history(losses_sgd, train_acc_sgd, test_acc_sgd, 30, "SGD")
plot_training_history(losses_momentum, train_acc_momentum, test_acc_momentum, 30, "Momentum")
plot_training_history(losses_adam, train_acc_adam, test_acc_adam, 30, "Adam")


best_params = params_adam


print("\nShowing sample predictions...")
plot_sample_predictions(X_test, y_test, best_params, class_names)


print("\nGenerating confusion matrix...")
plot_confusion_matrix(X_test, y_test, best_params, class_names)


predictions, _ = forward_propagation(X_test, best_params['W1'], best_params['b1'],
                                   best_params['W2'], best_params['b2'])
pred_classes = np.argmax(predictions, axis=1)

print("\nDetailed Classification Report:")
print("=" * 60)
print(classification_report(y_test, pred_classes, target_names=class_names))

