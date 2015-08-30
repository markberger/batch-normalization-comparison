import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from data import mnist
from models.mlp import MLP
from models.bnn import BNN

BATCH_SIZE = 64
HIDDEN_SIZE = 100
N_ITERS = 1000

def compare_mlps(
    X_train, Y_train, X_test, Y_test,
    hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, n_iters=N_ITERS,
):
    # Reshape if the training data is 3d tensors
    if X_train.ndim == 3:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    layers = [
        (X_train.shape[1], hidden_size),
        (hidden_size, hidden_size),
        (hidden_size, Y_train.shape[1]),
    ]

    mlp = MLP(layers)
    bnn = BNN(layers)
    y_true = np.argmax(Y_test, axis=1)

    mlp_accuracies = []
    bnn_accuracies = []

    for epoch in range(n_iters):
        # Train models
        print 'Epoch: {}'.format(epoch)
        X_train, Y_train = shuffle(X_train, Y_train)
        mlp.batch_train(X_train, Y_train, batch_size)
        bnn.batch_train(X_train, Y_train, batch_size)

        # Test models
        Y_hats = mlp.batch_predict(X_test, batch_size)
        y_guess = np.argmax(Y_hats, axis=1)
        mlp_acc = accuracy_score(y_true, y_guess)
        print 'MLP:\t{}'.format(mlp_acc)
        mlp_accuracies.append(mlp_acc)

        Y_hats = bnn.batch_predict(X_test, batch_size)
        y_guess = np.argmax(Y_hats, axis=1)
        bnn_acc = accuracy_score(y_true, y_guess)
        print 'BNN:\t{}'.format(bnn_acc)
        bnn_accuracies.append(bnn_acc)

    return mlp_accuracies, bnn_accuracies

if __name__ == '__main__':
    X_train, Y_train = mnist.training_data()
    X_test, Y_test = mnist.test_data()

    compare_mlps(X_train, Y_train, X_test, Y_test)
