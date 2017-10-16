import numpy as np
import matplotlib.pyplot as plt
import random
import mnist

np.random.seed(0)


def softmax(z):
    """
    Compute the probability that a sample fall into each class

    """
    A = np.exp(z - np.max(z, axis=0, keepdims=True))
    return A / A.sum(axis=0)


def one_hot_encoding(nb_classes, label):
    return np.eye(nb_classes)[label]


class SoftmaxRegression():
    """
    C: number of class, in this case C = 3
    D: number of features in each sample, in this case D = 3
    N: number of sample
    Matrix of weights [D x C]
    Training data [D x N]
    Training label [C x N]
    """
    def __init__(self, nb_class):
        self.nb_class = nb_class
        self.weights = np.random.normal(0, 0.01, [784 + 1, self.nb_class])

    def generate_sample_set(self, num_sample):
        """
        A function helps to generate training set
        """
        if self.nb_class == 4:
            center = np.array([[2, 2], [7, 0], [5, 5], [4, -3]])
            # center = np.array([[3, 3], [3, -3], [-3, -3], [-3, 3]])
        else:
            raise ValueError('Not invalid')
        I = np.array([[1, 0], [0, 1]])
        X = []
        Y = []
        for i in range(self.nb_class):
            X.extend(np.random.multivariate_normal(mean=center[i], cov=I, size=num_sample))
            for k in range(num_sample):
                Y.append(one_hot_encoding(nb_classes=self.nb_class, label=i))
        X = np.asarray(X)
        X = np.c_[np.ones(X.shape[0]), X]  # Add the bias to the training set
        X = X.T
        Y = np.stack(Y).T
        return [X, Y]

    def approximate_gradient(self, training_set):
        grad = np.empty_like(self.weights)
        ws = self.weights.shape
        for i in range(ws[0]):
            for j in range(ws[1]):
                tmp = self.weights
                self.weights[i, j] += 1.e-6
                tmp_p = self.cost_function(training_set)
                self.weights[i, j] -= 2.e-6
                tmp_n = self.cost_function(training_set)
                self.weights = tmp
                grad[i, j] = (tmp_p - tmp_n) / 2.e-6
        return grad

    def compute_gradient(self, training_set):
        [X, Y] = training_set
        n_train = X.shape[1]
        error = Y - softmax(np.dot(self.weights.T, X))
        grad = - np.dot(X, error.T) / n_train
        return grad

    def gradient_check(self, training_set):
        return np.allclose(self.approximate_gradient(training_set),
                           self.compute_gradient(training_set), rtol=1.e-2)

    def plot_sample(self, training_set):
        """
        Plot every point in training set
        """
        [X, Y] = training_set
        X = X.T
        Y = Y.T
        for i in range(X.shape[0]):
            if Y[i, 0] == 1:
                plt.scatter(X[i, 1], X[i, 2], c='r')
            elif Y[i, 1] == 1:
                plt.scatter(X[i, 1], X[i, 2], c='g')
            elif Y[i, 2] == 1:
                plt.scatter(X[i, 1], X[i, 2], c='b')
            else:
                plt.scatter(X[i, 1], X[i, 2], c='y')

    def cost_function(self, training_set):
        [X_train, Y_train] = training_set
        n_train = X_train.shape[1]
        cost = 0
        for i in range(n_train):
            tmp = softmax(np.dot(self.weights.T, X_train[:, i]))
            cost += np.dot(np.transpose(Y_train[:, i]), np.log(tmp))
        return cost * -1.0 / n_train

    def sgd(self, training_set, epochs, mini_batch_size, eta):
        """
        Method of choosing mini training set so that we could accelerate the training
        Stochastic Gradient Descent
        """
        [X_train, y_train] = training_set
        n_training = X_train.shape[1]
        for j in xrange(epochs):
            last_weights = self.weights
            n = range(n_training)
            random.shuffle(n)
            X_train = X_train[:, n]
            y_train = y_train[:, n]
            mini_batches = [X_train[:, k: k + mini_batch_size]
                            for k in range(0, n_training, mini_batch_size)]
            mini_batches_labels = [y_train[:,i: i + mini_batch_size]
                                   for i in range(0, n_training, mini_batch_size)]

            for mini_batch, mini_batch_label in zip(mini_batches, mini_batches_labels):
                self.gradient_descent(mini_batch, mini_batch_label, eta)
                # self.update_with_hessian(mini_batch, mini_batch_label)
            current_weights = self.weights
            # if np.linalg.norm(current_weights - last_weights) < 1.e-4:
            #     break


    def gradient_descent(self, mini_batch, mini_batch_label, eta):
        """
        Update parameters with Gradient Descent
        """
        n_train = mini_batch.shape[1]
        error = mini_batch_label - softmax(np.dot(self.weights.T, mini_batch))
        grad = np.dot(mini_batch, error.T)
        self.weights += eta / n_train * grad

    # TODO: Cannot plot the boundary of Softmax Regression???
    def plot_boundary(self):
        W = self.weights.T
        for i in range(self.nb_class):
            x = np.asarray(range(10))
            y = - W[i, 1] / W[i, 2] * x - W[i, 0] / W[i, 2]
            plt.plot(x, y, label=str(i))

    def evaluate(self, test_set):
        """
        Evaluate the performance of Softmax Regression using test_set
        """
        [X, Y] = test_set
        n_test = X.shape[1]
        predict = softmax(self.weights.T.dot(X))
        tmp = np.argmax(predict, axis=0)
        predict = one_hot_encoding(nb_classes=self.nb_class, label=tmp)
        return np.sum(np.abs(predict - Y.T)) / np.float(2 * n_test)



if __name__ == '__main__':
    sr = SoftmaxRegression(10)
    datasets = mnist.load_mnist(train_dir='/tmp/mnist/')
    X_train = datasets.train.images
    X_train = np.c_[np.ones(X_train.shape[0]), X_train].T
    y_train = one_hot_encoding(nb_classes=10, label=datasets.train.labels).T
    training_set = [X_train, y_train]
    X_test = datasets.test.images
    X_test = np.c_[np.ones(X_test.shape[0]), X_test].T
    y_test = one_hot_encoding(nb_classes=10, label=datasets.test.labels).T
    test_set = [X_test, y_test]
    sr.sgd(training_set=training_set, epochs=50, mini_batch_size=128, eta=0.05)
    print sr.evaluate(test_set=test_set)