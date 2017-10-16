import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#not tested


def kmeans_display(X, labels):
    k = np.amax(labels) + 1
    X0 = X[labels == 0, :].T
    X1 = X[labels == 1, :].T
    X2 = X[labels == 2, :].T

    plt.scatter(X0[0,:], X0[1,:], c='b', marker='*')
    plt.scatter(X1[0,:], X1[1,:], c='r', marker='o')
    plt.scatter(X2[0,:], X2[1,:], c='g', marker='x')
    plt.show()


def kmeans_init_centroids(X, k):
    """
    Initialize the centroids by choosing randomly k rows from training data
    :param X:
    :param k:
    :return:
    """
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    return centroids


def kmean_assign_label(X, centroids):
    """
    Compute the distance between the training data and the centroids to decide the sample belongs to
    which group
    :param X:
    :param centroids:
    :return:
    """
    dist = cdist(XA=X, XB=centroids)
    return np.argmin(dist, axis=1)


def kmean_update_centroids(X, labels, k):
    """
    Update the coordinates of the centroids based on the labels of each group
    :param X:
    :param labels:
    :param k:
    :return:
    """
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        # centroids[i] = np.sum(X[labels == k], axis=0) / len(X[labels == k])
        # Consider function np.mean
        Xi = X[labels == i, :]
        centroids[i, :] = np.mean(Xi, axis=0)
    return centroids


def has_converged(centroids, new_centroids):
    """
    Compare the new centroids with the existing one, if they are nearly the same, abort the iteration
    :param centroids:
    :param new_centroids:
    :return:
    """
    return np.allclose(centroids, new_centroids)


def kmean(X, k):
    centroids = [kmeans_init_centroids(X=X, k=k)]
    labels = []
    it = 0
    while True:
        # We save all the results during the iteration, the final results are the last one
        labels.append(kmean_assign_label(X=X, centroids=centroids[-1]))
        new_centroids = kmean_update_centroids(X=X, labels=labels[-1], k=k)
        if has_converged(centroids=centroids, new_centroids=new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return centroids, labels, it


if __name__ == '__main__':
    np.random.seed(11)
    N = 10
    mean = np.array([[1, 2], [5, 2], [4, 6]])
    cov = np.array([[1, 0], [0, 1]])
    X0 = np.random.multivariate_normal(mean=mean[0], cov=cov, size=N)
    X1 = np.random.multivariate_normal(mean=mean[1], cov=cov, size=N)
    X2 = np.random.multivariate_normal(mean=mean[2], cov=cov, size=N)
    k = len(mean)
    X = np.concatenate((X0, X1, X2), axis=0)
    centroids, labels, it = kmean(X=X, k=k)
    print centroids
