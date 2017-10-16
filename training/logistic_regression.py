import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

np.random.seed(0)


def sigmoid(z):
    """
    Compute the sigmoid function used by logistic regression
    """
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression():
    """
    A self_defined class to initialize the Logistic Regression to do the classification
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.weights = np.random.normal(0, 0.01, [3, 1]) # Each sample has 2 features plus a bias
        # Matrix of weights [3 x 1]

    def cost_function(self, training_set):
        """
        Compute codt function in Linear Regression
        """
        [X_train, y_train] = training_set
        n_train = len(y_train)
        tmp = sigmoid(np.dot(X_train, self.weights))
        return -1.0 / n_train * (np.dot(y_train.T, np.log(tmp)) +
                                np.dot((np.ones((n_train, 1)) - y_train).T, np.log(np.ones((n_train, 1)) - tmp)))

    def approximate_gradient(self, training_set):
        """
        Approximate gradient by using the finite difference method (numerically)
        """
        grad = []
        for i in range(3):
            tmp = self.weights  # Used to save the original value of weight
            self.weights[i] += 1.e-8
            tmp_p = self.cost_function(training_set)
            self.weights[i] -= 2.e-8
            tmp_n = self.cost_function(training_set)
            self.weights = tmp
            grad.append((tmp_p - tmp_n) / 2.e-8)
        grad = np.asarray(grad).reshape(3, 1)
        return grad

    def compute_gradient(self, training_set):
        """
        Compute Gradient analytically
        """
        [X_train, y_train] = training_set
        n_train = len(y_train)
        tmp = sigmoid(np.dot(X_train, self.weights))
        return 1.0 / n_train * np.dot(X_train.T, tmp - y_train)

    def gradient_check(self):
        training_set = self.generate_training_set(50)
        return np.allclose(self.approximate_gradient(training_set), self.compute_gradient(training_set), rtol=1.e-2)

    def generate_training_set(self, num_sample):
        """
        A function helps to generate training set
        """
        center = []
        I = np.array([[1, 0], [0, 1]])
        X = []
        for i in range(self.num_class):
            center.append([2.5 * i + 1, 1])
            X.extend(np.random.multivariate_normal(mean=center[i], cov=I, size=num_sample))
        X = np.asarray(X)
        y = np.asarray(num_sample * [0] + num_sample * [1])
        X = np.c_[np.ones(len(y)), X]  # Add the bias to the training set
        y = y.reshape(len(y), 1)
        return [X, y]

    def generate_test_set(self, num_sample):
        """
        A function help to generate test set
        """
        center = []
        I = np.array([[1, 0], [0, 1]])
        X = []
        for i in range(self.num_class):
            center.append([4 * i + 1, 1])
            X.extend(np.random.multivariate_normal(mean=center[i], cov=I, size=num_sample))
        X = np.asarray(X)
        y = np.asarray(num_sample * [0] + num_sample * [1])
        X = np.c_[np.ones(len(y)), X]  # Add the bias to the training set
        y = y.reshape(len(y), 1)
        return [X, y]

    def sgd(self, training_set, epochs, mini_batch_size, eta):
        """
        Method of choosing mini training set so that we could accelerate the training
        Stochastic Gradient Descent
        """
        [X_train, y_train] = training_set
        n_training = len(y_train)
        for j in xrange(epochs):
            last_weights = self.weights
            n = range(n_training)
            random.shuffle(n)
            X_train = X_train[n]
            y_train = y_train[n]
            mini_batches = [X_train[k: k + mini_batch_size]
                            for k in range(0, n_training, mini_batch_size)]
            mini_batches_labels = [y_train[i: i + mini_batch_size]
                                   for i in range(0, n_training, mini_batch_size)]

            for mini_batch, mini_batch_label in zip(mini_batches, mini_batches_labels):
                # self.gradient_descent(mini_batch, mini_batch_label, eta)
                self.update_with_hessian(mini_batch, mini_batch_label)
            current_weights = self.weights
            # if np.linalg.norm(current_weights - last_weights) < 1.e-4:
            #     break

    def gradient_descent(self, mini_batch, mini_batch_label, eta):
        n_training = len(mini_batch_label)
        grad = np.dot(mini_batch.T, sigmoid(np.dot(mini_batch, self.weights)) - mini_batch_label)
        self.weights = self.weights - eta / n_training * grad

    def update_with_hessian(self, mini_batch, mini_batch_label):
        """
        Update parameters with Newton's method. It doesn't require learning rate eta
        """
        a = sigmoid(np.dot(mini_batch, self.weights))
        print(a)
        n_train = mini_batch.shape[0]
        A = a * (1 - a)
        A = np.squeeze(np.asarray(A))
        W = np.diag(A)
        grad = np.dot(mini_batch.T, sigmoid(np.dot(mini_batch, self.weights)) - mini_batch_label)
        hessian = - 1.0 / n_train * np.dot(mini_batch.T, np.dot(W, mini_batch)) + 0.05 * np.eye(grad.shape[0])
        # TODO: Hessian itsellf is a singular matrix??? >>> Do some tricks like pseudo-inverse


        self.weights = self.weights - np.dot(inv(hessian), grad)

    def plot_sample(self, training_set):
        """
        Plot every point in training set
        """
        [X, y] = training_set
        X = X.T
        for i in range(len(y)):
            if y[i] == 0:
                plt.scatter(X[1, i], X[2, i], c='r')
            else:
                plt.scatter(X[1, i], X[2, i], c='b')

    def plot_boundary(self):
        """
        Plot the boundary using Linear Regression. Notice that there are only 3 weights
        """
        # TODO: Explore the way if there are more than 3 weights
        x = np.asarray(range(10))
        y = - self.weights[1] / self.weights[2] * x - self.weights[0] / self.weights[2]
        plt.plot(x, y)

    def evaluate(self, test_set):
        """
        Evaluate performance of the algorithm by using test set
        """
        [X_test, y_test] = test_set
        n_test = len(y_test)
        y_predict = np.dot(X_test, self.weights)
        y_predict[y_predict >= 0] = 1
        y_predict[y_predict < 0] = 0
        return np.sum(np.abs(y_predict - y_test)) / np.float(n_test)


if __name__ == '__main__':
    lr = LogisticRegression(2)
    training_set = lr.generate_training_set(200)
    test_set = lr.generate_test_set(10)
    lr.sgd(training_set=training_set, epochs=100, mini_batch_size=5, eta=0.05)
    print lr.evaluate(test_set=test_set)
    lr.plot_sample(training_set=training_set)
    lr.plot_boundary()
    plt.xlim(-3, 5)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal')
    plt.show()
