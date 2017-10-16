'''
Created on 17 mai 2017

@author: MB251995
Present 3 approach to compute external energy for the Active Contour
Common formula of external energy:
    E_ext = integral( P(grad(I)) * ds)
'''
# coding: utf-8
import numpy as np


def compute_energy_external_total(spline, img, nbr_dct=10000, normalized=True):
    '''
    The first formula to compute external energy
        P(u) = -u

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSplineCLG instance
    img: numpy.narray
        An image to compute energy
    nbr_dct: integer, optional
        Number of point de dicretization to compute the internal energy. We
        will take n values of u in the interval [0, 1] to compute the values of
        the curve at each u. Then use the the mathematic conversion to change
        the parameter to s.
    Returns
    -------
    energy : float
        External energy computed from the image
    '''

    [x, y] = np.transpose(spline(np.linspace(0.0, 1.0, nbr_dct + 1)))
    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(0.0, 1.0,
                                                               nbr_dct + 1)))
    tmp = img(y, x, grid=False) * np.hypot(x1u, y1u)
    if normalized:
        return -np.trapz(tmp, dx=1.0 / nbr_dct) / np.trapz(y=np.hypot(x1u, y1u),
                                                           dx=1.0 / nbr_dct)
    else:
        return -np.trapz(tmp, dx=1.0 / nbr_dct)


def compute_energy_external_partial(spline, img, ust, ued, nbr_dct,
                                    nbr_total=1000, normalized=False):
    '''
    The first formula to compute external energy
        P(u) = -u
    It is used to compute the jacobian with respect to a specific control point

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSplineCLG instance
    img: numpy.narray
        An image to compute energy
    ust: float
        The starting point in the knot interval when computing the partial
        energy
    ued: float
        The ending point in the knot interval when computing the partial energy
    nbr_dct: integer, optional
        Number of point de dicretization to compute the internal energy. We
        will take n values of u in the interval [ust, ued] to compute the
        values of the curve at each u. Then use the the mathematic conversion
        to change the parameter to s.
    Returns
    -------
    energy : float
        External energy computed from the image
    '''
    [x, y] = np.transpose(spline(np.linspace(ust, ued, nbr_dct + 1)))
    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(ust, ued,
                                                               nbr_dct + 1)))
    tmp = img(y, x, grid=False) * np.hypot(x1u, y1u)

    if not normalized:
        return -np.trapz(tmp, dx=(ued - ust) / nbr_dct)
    else:
        [x1u_t, y1u_t] = np.transpose(spline.derivative(1)
                                      (np.linspace(0.0, 1.0, nbr_total + 1)))
        return -np.trapz(tmp, dx=(ued - ust) / nbr_dct) / \
            np.trapz(y=np.hypot(x1u_t, y1u_t), dx=1.0 / nbr_total)


