import numpy as np
import random

# not done
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) / (1 - sigmoid(z))


class Network():
    """
    A self-defined class which helps to initialize Neural network
    """
    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.rand(y, 1) for y in size[1:]]
        self.weights = [np.random.rand(j, i) for i, j in zip(size[:-1], size[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a) + b
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Choose the mini batch for each parameter based on Stochastic Gradient Descent
        :param training_data: numpy.ndarray
            A list of training data given to the neural network
        :param epochs: int
        :param mini_batch_size: int
            Size of each batch in one update of parameters
        :param eta: float
            Learning rate of Gradient Descent
        :param test_data: A list of test data for performance evaluation
        :return:
        """

        if test_data:
            n_test = len(test_data)
        n_training = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch=mini_batch, eta=eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data) / n_test)
            else:
                print "Epoch {0} complete".format(i)


    def update_mini_batch(self, mini_batch, eta):
        """
        Apply the Gradient Descent to update the parameter
        :param mini_batch: numpy.ndarray
            Mini batch for the approximation of the gradient of loss function
        :param eta: int
            Learning rate of Gradient Descent
        :return:
        """
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprob(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprob(self):
        return 0

