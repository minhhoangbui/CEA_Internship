'''
Created on 16 mai 2017

@author: MB251995

Propose a method to estimate the internal energy according to the model
of Active Contour

'''
# coding: utf-8

import numpy as np


def compute_energy_internal_partial(spline, ust, ued, nbr_dct, ba_ratio=100):
    '''
    Compute the partial internal energy based on Active Contour model. It helps
    to compute the matrix jacobian in order to accelerate the optimisation.
    According to the property of spline, each control point on has its
    influence in an internal interval [ust; ued] of the knots interval. So when
    computing the jacobian of each control point, we only need to compute the
    difference in energy in that interval.
    The formula of internal energy based on Active Contour:
        E_internal(v) = alpha * int( (dv / ds)^2 + beta * (d^2v / ds^2)^2 ds

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSplineCLG instance
    ust: float
        The starting value of the internal interval
    ued: float
        The ending value of the internal interval
    nbr_dct: integer
        The number of discrete value in that interval we use to compute the
        energy
    ba_ratio: float (optional)
        The ratio of beta / alpha

    Returns
    -------
    energy : float
        Internal energy computed from the curve

    References
    ----------
    Kass, M.; Witkin, A.; Terzopoulos, D. (1988). "Snakes: Active contour
     models" (PDF). International Journal of Computer Vision. 1 (4): 321.

    '''

    # Generate the points in the curve

    # Generate the 1st and 2nd derivative of the points above with respect to u

    # v1s, v2s initialization
    # v1s is not initialized as it is alway 1 by definition
    # term_1st = v1s * ds / du; term_2nd = v2s * ds / du
    # Compute x, y , x1u, y1u, x2u, y2u
    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(ust, ued,
                                                               nbr_dct + 1)))
    [x2u, y2u] = np.transpose(spline.derivative(2)(np.linspace(ust, ued,
                                                               nbr_dct + 1)))
    v2s = (y1u * x2u - x1u * y2u) ** 2 / (x1u ** 2 + y1u ** 2) ** 3

    term_1st = np.hypot(x1u, y1u)
    term_2nd = v2s * np.hypot(x1u, y1u)

    # Compute full energy in the interval
    partial_int_energy = np.trapz(term_1st + ba_ratio * term_2nd, dx=(ued -
                                                                      ust) /
                                  nbr_dct)
    return partial_int_energy


def compute_energy_internal_total(spline, ba_ratio=100, nbr_dct=1000):
    '''
    Compute internal energy of a spline curve based on the theory of active
    contour

    E_internal(v) = alpha * int( (dv / ds)^2 + beta * (d^2v / ds^2)^2 ds

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSplineCLG instance
    ba_ratio: float
        The ratio of beta / alpha
    n_dct: integer, optional
        Number of point de dicretization to compute the internal energy. We
        will take n values of u in the interval [0, 1] to compute the values of
        the curve at each u. Then use the the mathematic conversion to change
        the parameter to s.

    Returns
    -------
    energy : float
        Internal energy computed from the curve


    References
    ----------

     Kass, M.; Witkin, A.; Terzopoulos, D. (1988). "Snakes: Active contour
     models" (PDF). International Journal of Computer Vision. 1 (4): 321.
    '''

    # Generate the points in the curve

    # Generate the 1st and 2nd derivative of the points above with respect to u

    # v1s, v2s initialization
    # v1s is not initialized as it is alway 1 by definition

    # term_1st = v1s * ds / du; term_2nd = v2s * ds / du

    # Compute x, y , x1u, y1u, x2u, y2u

    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(0, 1,
                                                               nbr_dct + 1)))
    [x2u, y2u] = np.transpose(spline.derivative(2)(np.linspace(0, 1,
                                                               nbr_dct + 1)))

    v2s = (y1u * x2u - x1u * y2u) ** 2 / (x1u ** 2 + y1u ** 2) ** 3

    term_1st = np.hypot(x1u, y1u)
    term_2nd = v2s * np.hypot(x1u, y1u)
    # Compute full energy along spline
    full_int_energy = np.trapz(term_1st + ba_ratio * term_2nd, dx=1.0 /
                               nbr_dct)

    return full_int_energy


def compute_energy_internal_total_seperate(spline, nbr_dct=1000):
    '''
    Compute internal energy of a spline curve based on the theory of active
    contour

    E_internal(v) = alpha * int( (dv / ds)^2 + beta * (d^2v / ds^2)^2 ds

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSplineCLG instance
    ba_ratio: float
        The ratio of beta / alpha
    n_dct: integer, optional
        Number of point de dicretization to compute the internal energy. We
        will take n values of u in the interval [0, 1] to compute the values of
        the curve at each u. Then use the the mathematic conversion to change
        the parameter to s.

    Returns
    -------
    energy : float
        Internal energy computed from the curve


    References
    ----------

     Kass, M.; Witkin, A.; Terzopoulos, D. (1988). "Snakes: Active contour
     models" (PDF). International Journal of Computer Vision. 1 (4): 321.
    '''

    # Generate the points in the curve

    # Generate the 1st and 2nd derivative of the points above with respect to u

    # v1s, v2s initialization
    # v1s is not initialized as it is alway 1 by definition

    # term_1st = v1s * ds / du; term_2nd = v2s * ds / du

    # Compute x, y , x1u, y1u, x2u, y2u

    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(0, 1,
                                                               nbr_dct + 1)))
    [x2u, y2u] = np.transpose(spline.derivative(2)(np.linspace(0, 1,
                                                               nbr_dct + 1)))

    v2s = (y1u * x2u - x1u * y2u) ** 2 / (x1u ** 2 + y1u ** 2) ** 3

    term_1st = np.hypot(x1u, y1u)
    term_2nd = v2s * np.hypot(x1u, y1u)

    return np.trapz(term_1st, dx=1.0 / nbr_dct), np.trapz(term_2nd, dx=1.0 /
                                                          nbr_dct)
