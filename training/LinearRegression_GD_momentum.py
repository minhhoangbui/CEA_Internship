import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(0)

class LinearRegression():
    def __init__(self):
        self.weights = np.random.normal(0., 0.01, [2, 1])

    def generate_sample_set(self, nb_sample):
        X = np.random.rand(nb_sample, 1)
        Y = 4 + 3 * X + 0.3 * np.random.rand(nb_sample, 1)
        X = np.c_[np.ones(nb_sample), X]
        return [X, Y]

    def sgd(self, training_set, epochs, mini_batch_size, eta):
        [X_train, Y_train] = training_set
        n_train = len(Y_train)
        for k in range(epochs):
            n = range(n_train)
            random.shuffle(n)
            X_train = X_train[n]
            Y_train = Y_train[n]
            mini_batches = [X_train[i: i + mini_batch_size]
                            for i in range(0, n_train, mini_batch_size)]
            mini_batches_labels = [Y_train[j: j + mini_batch_size]
                                   for j in range(0, n_train, mini_batch_size)]
            for mini_batch, mini_batch_label in zip(mini_batches, mini_batches_labels):

                self.gradient_descent(mini_batch, mini_batch_label, mini_batch_size, eta)

    def gradient_descent(self, mini_batch, mini_batch_label, mini_batch_size, eta):
        error = mini_batch_label - np.dot(mini_batch, self.weights)
        self.weights += eta / mini_batch_size * np.dot(mini_batch.T, error)

class LinearRegression_Momentum():
    def __init__(self):
        self.weights = np.random.normal(0., 0.01, [2, 1])

    def generate_sample_set(self, nb_sample):
        X = np.random.rand(nb_sample, 1)
        Y = 4 + 3 * X + 0.3 * np.random.rand(nb_sample, 1)
        X = np.c_[np.ones(nb_sample), X]
        return [X, Y]

    def sgd(self, training_set, epochs, mini_batch_size, eta, gamma):
        [X_train, Y_train] = training_set
        n_train = len(Y_train)
        grad_old = np.empty_like(self.weights)
        for k in range(epochs):
            n = range(n_train)
            random.shuffle(n)
            X_train = X_train[n]
            Y_train = Y_train[n]
            mini_batches = [X_train[i: i + mini_batch_size]
                            for i in range(0, n_train, mini_batch_size)]
            mini_batches_labels = [Y_train[j: j + mini_batch_size]
                                   for j in range(0, n_train, mini_batch_size)]
            for mini_batch, mini_batch_label in zip(mini_batches, mini_batches_labels):
                error = mini_batch_label - np.dot(mini_batch, self.weights)
                grad_new = gamma * grad_old - eta / mini_batch_size * np.dot(mini_batch.T, error)
                self.weights -= grad_new


if __name__ == '__main__':
    lr = LinearRegression_Momentum()
    training_set = lr.generate_sample_set(100)
    lr.sgd(training_set=training_set, epochs=10, mini_batch_size=5, eta=0.05, gamma=0.1)
    x = [0, 1]
    y = lr.weights[0] + lr.weights[1] * x
    plt.plot(x, y)
    [X, Y] = training_set
    plt.scatter(X[:, 1], Y)
    plt.show()


