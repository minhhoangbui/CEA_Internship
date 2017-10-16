'''
Created on 16 mai 2017

@author: MB251995
    Provide a prototype of optimisation using scipy for the Active Contour
'''
# coding: utf-8

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from activecontours.energie.energie_interne import \
                    compute_energy_internal_total
from activecontours.energie.energie_interne import \
                    compute_energy_internal_partial
from activecontours.energie.energie_externe import \
                    compute_energy_external_total
from activecontours.energie.energie_externe import \
                    compute_energy_external_partial


def optimize_traditional(spline, method, img=None, jac=None, ba_ratio=100,
                         gamma=10, nbr=1000, eps=1.e-5):
    '''
    Minimizing the energy of the spline with respect to control points. This
    function works well with traditional optimisation of Active Contour.
    Moreover, we can check the simplified convergence of internal energy with
    this function.
    Notice: I also implement the total optimisation with image with the
    property of spline. However, due to the normalisation of the external
    energy, it does not work well.

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSpline CLG instance
    method: string
        Method of optimisation
    img: numpy.narray
        The pre-processed image in which we apply Active Contour
    jac: numpy.narray
        The pre-computed jacobian for the Gradient Descent in the optimisation.
        If jac=None, the algorithm will estimate the numerical jacobian itself.
    ba_ratio: float
        The ratio of beta to alpha
    gamma: float
        The ratio of the external energy to internal energy.
    nbr: integer
        The number of discrete points to compute the energy.
    eps: float
        The step of numerical gradient

    '''
    knots = spline.t
    deg = spline.k
    ctrl_pts_tmp = spline.get_ctrl_pts(only_free=True)
    nbr_lib = len(ctrl_pts_tmp) / 2
    n_total = nbr

    def cost(ctrl_pts):
        '''
        Compute the total energy given the new control points. It is the cost
        function of the optimisation. It also helps to compute the traditional
        jacobian.

        Parameters
        ----------
        ctrl_pts : numpy.ndarray
            The flattened vector of control points

        Returns
        -------
        ret: float
            The total energy corresponding to the given control points.If
            img=None, it returns the partial internal energy. If not, it
            returns the partial total (internal + external) energy.

        '''
        ctrl_pts = np.array(ctrl_pts)
        ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
        spline.set_ctrl_pts(ctrl_pts, only_free=True)

        if img is None:
            ret = compute_energy_internal_total(spline=spline,
                                                ba_ratio=ba_ratio,
                                                nbr_dct=nbr)
        else:
            ret = compute_energy_internal_total(spline=spline,
                                                ba_ratio=ba_ratio,
                                                nbr_dct=nbr) + \
                    gamma * compute_energy_external_total(spline=spline,
                                                          img=img,
                                                          nbr_dct=nbr,
                                                          normalized=True)

        return ret

    def jacobian_traditional_2sides(dt_pts):
        '''
        Given the control point, this function helps to contruct the jacobian
            f'(x) = (f(x + h) - f(x - h)) / (2 * h)

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The vector of control point

        Returns
        -------
        ret : numpy.ndarray (n-1,)
            The jacobian matrix

        '''
        dt_pts = dt_pts.flatten()
        ret = np.zeros_like(dt_pts)
        dt_p = np.zeros_like(dt_pts)
        dt_m = np.zeros_like(dt_pts)
        for i, data in enumerate(dt_pts):
            dt_p[:] = dt_pts
            dt_m[:] = dt_pts
            dt_p[i] = data + eps
            dt_m[i] = data - eps
            # reshaping of dt_p and dt_m: it will be done automatically in cost
            ret[i] = (cost(dt_p) - cost(dt_m)) / (2 * eps)
        return ret

    def jacobian_traditional_1side(dt_pts):
        '''
        Given the control point, this function helps to contruct the jacobian
            f'(x) = (f(x + h) - f(x)) / h

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The vector of control point

        Returns
        -------
        ret : numpy.ndarray (n-1,)
            The jacobian matrix

        '''
        dt_pts = dt_pts.flatten()
        cost_act = cost(dt_pts)
        ret = np.zeros_like(dt_pts)
        dt_p = np.zeros_like(dt_pts)
        for i, data in enumerate(dt_pts):
            dt_p[:] = dt_pts
            dt_p[i] = data + eps
            ret[i] = (cost(dt_p) - cost_act) / eps
        return ret

    def cost_partial(ctrl_pts, nth):
        '''
        The function is used to compute the jacobian given the control point
        and the order of the varialbe. Firstly, we choose the internal interval
        and compute the partial energy. It helps to compute the simplified
        jacobian

        Parameters
        ----------
        ctrl_pts : numpy.narray
            The flattened vector of control points
        nth: integer
            The order of the control point of which we want to compute the
            partial energy

        Returns
        -------
        partial_energy : float
            The partial energy corresponding to variables nth. If img=None, it
            returns the partial internal energy. If not, it returns the partial
            total (internal + external) energy.

        References
        ----------
        Les Piegl, Wayne Tiller, The NURBS book p.84

        '''
        ctrl_pts = np.array(ctrl_pts)
        ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
        spline.set_ctrl_pts(ctrl_pts, only_free=True)
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            if img is None:
                ret1 = compute_energy_internal_partial(spline=spline,
                                                       ust=ust1, ued=ued1,
                                                       nbr_dct=n_dct1) + \
                    compute_energy_internal_partial(spline=spline,
                                                    ust=ust2, ued=ued2,
                                                    nbr_dct=n_dct2)
            else:
                ret1 = compute_energy_internal_partial(spline=spline,
                                                       ust=ust1, ued=ued1,
                                                       nbr_dct=n_dct1) + \
                    compute_energy_internal_partial(spline=spline,
                                                    ust=ust2, ued=ued2,
                                                    nbr_dct=n_dct2) + gamma * \
                    compute_energy_external_partial(spline=spline, img=img,
                                                    ust=ust1, ued=ued1,
                                                    nbr_dct=n_dct1) + gamma * \
                    compute_energy_external_partial(spline=spline, img=img,
                                                    ust=ust2, ued=ued2,
                                                    nbr_dct=n_dct2)
            return ret1
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            if img is None:
                ret2 = compute_energy_internal_partial(spline, ust, ued, n_dct)
            else:
                ret2 = compute_energy_internal_partial(spline=spline,
                                                       ust=ust, ued=ued,
                                                       nbr_dct=n_dct) + gamma * \
                    compute_energy_external_partial(spline=spline, img=img,
                                                    ust=ust, ued=ued,
                                                    nbr_dct=n_dct)
            return ret2

    def jacobian_spline_2sides(dt_pts):
        '''
        Given the control point, this function helps to contruct the jacobian.
        It uses the property of spline to accelerate the computation.
            f'(x) = (f(x + h) - f(x - h)) / (2 * h)

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The vector of control point

        Returns
        -------
        ret : numpy.ndarray
            The jacobian matrix

        '''
        dt_pts = dt_pts.flatten()
        ret = np.zeros_like(dt_pts)
        dt_p = np.zeros_like(dt_pts)
        dt_m = np.zeros_like(dt_pts)
        for i, data in enumerate(dt_pts):
            dt_p[:] = dt_pts
            dt_m[:] = dt_pts
            dt_p[i] = data + eps
            dt_m[i] = data - eps
            ret[i] = (cost_partial(dt_p, i // 2) - cost_partial(dt_m, i // 2)) \
                / (2 * eps)
        return ret

    def jacobian_spline_1side(dt_pts):
        '''
        Given the control point, this function helps to contruct the jacobian.
        It uses the property of spline to accelerate the computation.
            f'(x) = (f(x + h) - f(x)) / h

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The vector of control point

        Returns
        -------
        ret : numpy.ndarray
            The jacobian matrix

        '''
        dt_pts = dt_pts.flatten()
        cost_pts = np.zeros(len(dt_pts) / 2)
        for i in range(len(dt_pts) / 2):
            cost_pts[i] = cost_partial(dt_pts, i)
        ret = np.zeros_like(dt_pts)
        dt_p = np.zeros_like(dt_pts)
        for i, data in enumerate(dt_pts):
            dt_p[:] = dt_pts
            dt_p[i] = data + eps
            ret[i] = (cost_partial(dt_p, i // 2) - cost_pts[i // 2]) / eps
        return ret

    def print_callback(var):
        '''
        After each iteration, the algorithm will give us a result var. This
        function helps to illustrate the convergence

        Parameters
        ----------
        var : numpy.ndaray
            The vector that we receive from the algorithm after each iteration

        '''

        print cost(var)
        var = np.array(var)
        var = np.reshape(var, ((len(var) / 2), 2))
        spline.set_ctrl_pts(var, only_free=True)
        spline.plot_spline(label=None)
        plt.draw()
        plt.pause(0.5)

    if jac == 'f':
        jacob = jacobian_traditional_1side
    elif jac == 's':
        jacob = jacobian_spline_1side
    else:
        jacob = None

    res = minimize(fun=cost, x0=spline.get_ctrl_pts(only_free=True),
                   jac=jacob,
                   callback=None,
                   method=method,
                   options={'disp': False, 'maxiter': 500})

    return res


def optimize_simplified(spline, method, img, ba_ratio=100, gamma=10,
                        nbr=10000, eps=1.e-5):
    '''
    Minimizing the energy of the spline with respect to control points. It is
    only for Active Contour with simplified jacobian for total energy (internal
    + external).

    Parameters
    ----------
    spline : BSplineCLG
        A well-defined BSpline CLG instance
    method: string
        Method of optimisation
    img: numpy.narray
        The pre-processed image in which we apply Active Contour
    jac: numpy.narray
        The pre-computed jacobian for the Gradient Descent in the optimisation.
        If jac=None, the algorithm will estimate the numerical jacobian itself.
    ba_ratio: float
        The ratio of beta to alpha
    gamma: float
        The ratio of the external energy to internal energy.
    nbr: integer
        The number of discrete points to compute the energy.
    eps: float
        The step of numerical gradient

    '''
    knots = spline.t
    deg = spline.k
    ctrl_pts_tmp = spline.get_ctrl_pts(only_free=True)
    nbr_lib = len(ctrl_pts_tmp) / 2
    n_total = nbr

    def cost(ctrl_pts):
        '''
        Compute the total energy given the new control points. It acts as the
        cost function of the optimisation.

        Parameters
        ----------
        ctrl_pts : numpy.ndarray
            The flattened vector of control points

        Returns
        -------
        ret: float
            The total energy corresponding to the given control points.

        '''
        ctrl_pts = np.array(ctrl_pts)
        ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
        spline.set_ctrl_pts(ctrl_pts, only_free=True)
        ret = compute_energy_internal_total(spline=spline,
                                            ba_ratio=ba_ratio,
                                            nbr_dct=nbr) + \
            gamma * compute_energy_external_total(spline=spline,
                                                  img=img,
                                                  nbr_dct=nbr,
                                                  normalized=True)

        return ret

    def ext_partial(spline, nth):
        '''
        The function is used to compute the partial external energy given the
        control point and the order of the varialbe. Firstly, we choose the
        internal interval and compute the partial energy. It helps to compute
        the simplified jacobian.

        Parameters
        ----------
        ctrl_pts : numpy.narray
            The flattened vector of control points
        nth: integer
            The order of the control point of which we want to compute the
            partial energy
        Returns
        -------
        ret1(ret2) : float
            Partial external energy corresponding the specified control point
        '''
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))

            ret1 = gamma * \
                compute_energy_external_partial(spline=spline, img=img,
                                                ust=ust1, ued=ued1,
                                                nbr_dct=n_dct1,
                                                normalized=False) + gamma * \
                compute_energy_external_partial(spline=spline, img=img,
                                                ust=ust2, ued=ued2,
                                                nbr_dct=n_dct2,
                                                normalized=False)
            return ret1
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            ret2 = gamma * \
                compute_energy_external_partial(spline=spline, img=img,
                                                ust=ust, ued=ued,
                                                nbr_dct=n_dct,
                                                normalized=False)
            return ret2

    def int_partial(spline, nth):
        '''
        The function is used to compute the partial internal energy given the
        control point and the order of the varialbe. Firstly, we choose the
        internal interval and compute the partial energy. It helps to compute
        the simplified jacobian.

        Parameters
        ----------
        ctrl_pts : numpy.narray
            The flattened vector of control points
        nth: integer
            The order of the control point of which we want to compute the
            partial energy
        Returns
        -------
        ret1(ret2) : float
            Partial internal energy corresponding the specified control point
        '''
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            ret1 = compute_energy_internal_partial(spline=spline,
                                                   ust=ust1, ued=ued1,
                                                   nbr_dct=n_dct1) + \
                compute_energy_internal_partial(spline=spline,
                                                ust=ust2, ued=ued2,
                                                nbr_dct=n_dct2)
            return ret1
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            ret2 = compute_energy_internal_partial(spline=spline,
                                                   ust=ust, ued=ued,
                                                   nbr_dct=n_dct)
            return ret2

    def jacobian_spline_1side(dt_pts):
        '''
        Given the control point, this function helps to contruct the jacobian.
        It uses the property of spline to accelerate the computation.
            f'(x) = (f(x + h) - f(x)) / h.
        However, due to the use of normalisation in the external energy, we
        have to modify the formula. The detail is in the report (VI.G)

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The vector of control point

        Returns
        -------
        ret : numpy.ndarray
            The jacobian matrix

        '''
        dt_pts = dt_pts.flatten()
        ctrl_pts = np.array(dt_pts)
        ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
        spline.set_ctrl_pts(ctrl_pts, only_free=True)
        l1 = spline.get_length()
        e1t = compute_energy_external_total(spline, img, normalized=False)
        e1i = np.zeros(len(dt_pts) / 2)
        e1e = np.zeros(len(dt_pts) / 2)
        for i in range(len(dt_pts) / 2):
            e1i[i] = int_partial(spline, i)
            e1e[i] = ext_partial(spline, i)
        ret = np.zeros_like(dt_pts)
        dt_p = np.zeros_like(dt_pts)
        for i, data in enumerate(dt_pts):
            dt_p[:] = dt_pts
            dt_p[i] = data + eps
            ctrl = np.array(dt_p)
            ctrl = np.reshape(ctrl, ((len(ctrl) / 2), 2))
            spline.set_ctrl_pts(ctrl, only_free=True)
            l2 = spline.get_length()
            ret[i] = ((ext_partial(spline, i // 2) - e1e[i // 2] -
                       e1t * (l2 - l1) / l1) / l2 +
                      int_partial(spline, i // 2) - e1i[i // 2]) / eps
        return ret

    def print_callback(var):
        print cost(var)
        var = np.array(var)
        var = np.reshape(var, ((len(var) / 2), 2))
        spline.set_ctrl_pts(var, only_free=True)
        spline.plot_spline(label=None)
        plt.draw()
        plt.pause(0.5)

    res = minimize(fun=cost, x0=spline.get_ctrl_pts(only_free=True),
                   jac=jacobian_spline_1side,
                   callback=None,
                   method=method,
                   options={'disp': False, 'maxiter': 50})
    return res

