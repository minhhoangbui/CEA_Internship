import numpy as np
import numpy.linalg as lg
from sklearn.decomposition import PCA

# not tested

class PCA_MH():
    def __init__(self, x, K):
        self.vectors = np.transpose(x)
        self.number = K

    def get_preprocessed(self):
        # Compute the mean vector of each feature
        mean = np.mean(a=self.vectors, axis=1)
        # Compute the preprocessed data
        preprocessed = np.zeros_like(self.vectors)
        for i in range(len(mean)):
            preprocessed[i, :] = self.vectors[i, :] - mean[i]
        return preprocessed

    def get_axis(self):
        preprocessed = self.get_preprocessed()
        # Compute the matrix of covariance
        N = len(preprocessed[0])
        S = 1.0 / N * np.dot(preprocessed, np.transpose(preprocessed))
        # Extract the eigenvalues and eigenvector from the matrix of covariance
        eigenValues, eigenVectors = lg.eig(S)
        # Sort the eigenvalues and eigenvectors from the biggest to the smallest in terms of eigenvalues
        idx = eigenValues.argsort()
        sorted(eigenValues, reverse=True)
        eigenVectors = eigenVectors[:, idx]
        for i in range(len(eigenVectors)):
            eigenVectors[i] = eigenVectors[i] * 1.0 / lg.norm(eigenVectors[i])
            eigenValues[i] *= 1.0 / lg.norm(eigenVectors[i])
        return eigenValues[:self.number], eigenVectors[:, :self.number]

    def get_new_coordinate(self):
        _, axis = self.get_axis()
        coordinates = np.dot(np.transpose(axis), self.get_preprocessed())
        return coordinates

    def get_approximation(self):
        return 0


if __name__ == '__main__':
    X = np.array([[2, 1, 1], [6, 5, 7], [7, 3, 4]])
    pca = PCA_MH(x=X, K=2)
    vl, vv = pca.get_axis()
    print vl, vv
    nw = pca.get_new_coordinate()
    print nw







