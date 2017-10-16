import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as norm

#tested


def gaussian_prob(X, mu, cov):
    return norm.pdf(x=X, mean=mu, cov=cov)


def gaussian_sampling(mu, cov, size=1):
    return np.random.multivariate_normal(mean=mu, cov=cov, size=size)


def gaussian_mixture_prob(x, mu_vec, std_vec, coefs):
    ret = 0
    for i in range(len(coefs)):
        ret += coefs[i] * gaussian_prob(x, mu_vec[i], std_vec)
    return ret


# def plot_boundary(x, mean1, mean2, std_vec, coefs):
#     xlist = np.linspace(-3, 8, 200)
#     ylist = np.linspace(-3, 8, 200)
#
#     x[0], x[1] = np.meshgrid(xlist, ylist)
#     plt.figure()
#     likelihood1 = gaussian_mixture_prob(x=x, mu_vec=mean1, std_vec=std_vec, coefs=coefs)
#     likelihood2 = gaussian_mixture_prob(x=x, mu_vec=mean2, std_vec=std_vec, coefs=coefs)
#     contour = plt.contour(x[0], x[1], np.log(likelihood1 / likelihood2), 0, colors='b')
#     plt.clabel(contour, colors='b')
#     contour_filled = plt.contourf(x[0], x[1], np.log(likelihood1 / likelihood2), 0)
#     plt.colorbar(contour_filled)
#     plt.show()


if __name__ == '__main__':
    coefs = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
    x1 = 5.0 + np.random.rand(1, 5)
    y1 = 6.0 + np.random.rand(1, 5)
    mean1 = np.append(x1, y1, axis=0).T
    x2 = 1.0 + np.random.rand(1, 5)
    y2 = 1.0 + np.random.rand(1, 5)
    mean2 = np.append(x2, y2, axis=0).T
    lambda1 = 1
    lambda2 = 10
    I = np.array([[lambda1, 0], [0, lambda2]])
    pts1 = []
    pts2 = []
    for i in range(len(coefs)):
        for j in range(int(coefs[i]*100)):
            pts1.append(gaussian_sampling(mu=mean1[i], cov=I))
            pts2.append(gaussian_sampling(mu=mean2[i], cov=I))
            # try using the parameter size while creating the sample


    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    # Do the whitening as pre-processing step
    pts1[0, :] = pts1[0, :] / np.sqrt(lambda1)
    pts2[0, :] = pts2[0, :] / np.sqrt(lambda1)
    pts1[1, :] = pts1[1, :] / np.sqrt(lambda2)
    pts1[1, :] = pts2[1, :] / np.sqrt(lambda2)

    xlist = np.linspace(-10, 16, 150)
    ylist = np.linspace(-10, 16, 150)
    X, Y = np.meshgrid(xlist, ylist)
    plt.figure()
    Z = np.empty_like(X)
    for i in range(len(X[:,0])):
        for j in range(len(X[0,:])):
            coor = np.array([X[i, j], Y[i, j]])
            likelihood1 = gaussian_mixture_prob(x=coor, mu_vec=mean1, std_vec=I, coefs=coefs)
            likelihood2 = gaussian_mixture_prob(x=coor, mu_vec=mean2, std_vec=I, coefs=coefs)
            Z[i, j] = np.log(likelihood1 / likelihood2)

    contour = plt.contour(X, Y, Z, [0.0, 100], colors='b')
    plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
    contour_filled = plt.contourf(X, Y, Z, [0.0, 100])
    plt.colorbar(contour_filled)
    plt.scatter(pts1[0,:], pts1[1,:], c='b')
    plt.scatter(pts2[0,:], pts2[1,:], c='r')
    plt.show()


