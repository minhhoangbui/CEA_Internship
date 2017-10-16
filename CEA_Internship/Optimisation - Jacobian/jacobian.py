'''
Created on 22 juin 2017
Describe the approaches of calculating the jacobian matrix.

@author: MB251995
'''
import numpy as np
import copy
from activecontours.energie.energie_interne import \
                        compute_energy_internal_partial
from activecontours.energie.energie_interne import \
                        compute_energy_internal_total
from activecontours.energie.energie_externe import \
                        compute_energy_external_total
from activecontours.energie.energie_externe import \
                        compute_energy_external_partial


def jacobian_full_1_side(spline_o, img, gamma=1):
    '''
    Given a well-defined spline, this function compute the jacobian with
    respect to the control point based on the formula:
        f'(x) = (f(x + h) - f(x)) / h
    This computation doesn't employ the property of spline
    Parameters
    ----------
    spline : BSplineCLG
        A well-defined spline curve

    Returns
    -------
    mat : numpy.ndarray
        The jacobian matrix
    '''
    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -6.0
    dat_p = np.zeros_like(dat)

    def cost(ctrl):
        ctrl = np.array(ctrl)
        ctrl = np.reshape(ctrl, ((len(ctrl) / 2), 2))
        spline.set_ctrl_pts(ctrl, only_free=True)
        if img is None:
            ret = compute_energy_internal_total(spline=spline,
                                                nbr_dct=1000)
        else:
            ret = compute_energy_internal_total(spline=spline,
                                                nbr_dct=1000) + \
                    gamma * compute_energy_external_total(spline=spline,
                                                          img=img,
                                                          nbr_dct=1000)
        return ret
    cost_act = cost(dat)
    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_p[i] = data + eps
        mat[i] = (cost(dat_p) - cost_act) / eps
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def jacobian_spl_1_side(spline_o, img, gamma=1):
    '''
    Given a well-defined spline, this function compute the jacobian with
    respect to the control point based on the formula:
        f'(x) = (f(x + h) - f(x)) / h
    This computation employs the property of spline.
    This method does not fit the computation of total jacobian because of the
    normalisation
    Parameters
    ----------
    spline : BSplineCLG
        A well-defined spline curve

    Returns
    -------
    mat : numpy.ndarray
        The jacobian matrix
    '''
    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -6.0
    dat_p = np.zeros_like(dat)
    cost_pts = np.zeros(len(ctrl_pts))

    def cost(ctrl_pts, nth):
        knots = spline.t
        deg = spline.k
        nbr_lib = len(ctrl_pts) / 2
        n_total = 1000
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            ctrl_pts = np.array(ctrl_pts)
            ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
            spline.set_ctrl_pts(ctrl_pts, only_free=True)
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
                    gamma * compute_energy_external_partial(spline=spline,
                                                            img=img,
                                                            ust=ust1, ued=ued1,
                                                            nbr_dct=n_dct1) + \
                    compute_energy_internal_partial(spline=spline,
                                                    ust=ust2, ued=ued2,
                                                    nbr_dct=n_dct2) + \
                    gamma * compute_energy_external_partial(spline=spline,
                                                            img=img,
                                                            ust=ust2, ued=ued2,
                                                            nbr_dct=n_dct2)
            return ret1
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            ctrl_pts = np.array(ctrl_pts)
            ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
            spline.set_ctrl_pts(ctrl_pts, only_free=True)
            if img is None:
                ret2 = compute_energy_internal_partial(spline=spline,
                                                       ust=ust, ued=ued,
                                                       nbr_dct=n_dct)
            else:
                ret2 = compute_energy_internal_partial(spline=spline,
                                                       ust=ust, ued=ued,
                                                       nbr_dct=n_dct) + \
                    gamma * compute_energy_external_partial(spline=spline,
                                                            img=img,
                                                            ust=ust, ued=ued,
                                                            nbr_dct=n_dct)
            return ret2

    for i in range(len(cost_pts)):
        cost_pts[i] = cost(dat, i)

    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_p[i] = data + eps
        mat[i] = (cost(dat_p, i // 2) - cost_pts[i // 2]) / eps
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def jacobian_spl_2_sides(spline_o, img, gamma=1):
    '''
    Given a well-defined spline, this function compute the jacobian with
    respect to the control point based on the formula:
        f'(x) = (f(x + h) - f(x - h)) / (2 * h)
    This computation employs the property of spline.
    This method does not fit the computation of total jacobian because of the
    normalisation.
    Parameters
    ----------
    spline_o : BSplineCLG
        A well-defined spline curve

    Returns
    -------
    mat : numpy.ndarray
        The jacobian matrix
    '''
    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -6.0
    dat_p = np.zeros_like(dat)
    dat_m = np.zeros_like(dat)

    def cost(ctrl_pts, nth):
        nth = nth // 2
        knots = spline.t
        deg = spline.k
        nbr_lib = len(ctrl_pts) / 2
        n_total = 1000
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            ctrl_pts = np.array(ctrl_pts)
            ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
            spline.set_ctrl_pts(ctrl_pts, only_free=True)
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
            ctrl_pts = np.array(ctrl_pts)
            ctrl_pts = np.reshape(ctrl_pts, ((len(ctrl_pts) / 2), 2))
            spline.set_ctrl_pts(ctrl_pts, only_free=True)
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
    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_m[:] = dat
        dat_p[i] = data + eps
        dat_m[i] = data - eps
        mat[i] = (cost(dat_p, i) - cost(dat_m, i)) / (2.0 * eps)
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def jacobian_full_2_sides(spline_o, img, gamma=1):
    '''
    Given a well-defined spline, this function compute the jacobian with
    respect to the control point based on the formula:
        f'(x) = (f(x + h) - f(x - h)) / (2 * h)
    This computation doesn't employ the property of spline
    Parameters
    ----------
    spline_o : BSplineCLG
        A well-defined spline curve

    Returns
    -------
    mat : numpy.ndarray
        The jacobian matrix
    '''
    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -6.0
    dat_p = np.zeros_like(dat)
    dat_m = np.zeros_like(dat)

    def cost(ctrl):
        ctrl = np.array(ctrl)
        ctrl = np.reshape(ctrl, ((len(ctrl) / 2), 2))
        spline.set_ctrl_pts(ctrl, only_free=True)
        if img is None:
            ret = compute_energy_internal_total(spline=spline,
                                                nbr_dct=1000)
        else:
            ret = compute_energy_internal_total(spline=spline,
                                                nbr_dct=1000) + \
                    gamma * compute_energy_external_total(spline=spline,
                                                          img=img,
                                                          nbr_dct=1000)
        return ret

    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_m[:] = dat
        dat_p[i] = data + eps
        dat_m[i] = data - eps
        mat[i] = (cost(dat_p) - cost(dat_m)) / (2.0 * eps)
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def jacobian_spl_1_side_external(spline_o, img, gamma=1):
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
    knots = spline_o.t
    deg = spline_o.k
    ctrl_pts = spline_o.get_ctrl_pts(only_free=True)
    nbr_lib = len(ctrl_pts) / 2
    n_total = 10000

    def ext_partial(spl, nth):
        if knots[nth] < 0:
            ust1 = 0.0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            return compute_energy_external_partial(spline=spl,
                                                   img=img,
                                                   ust=ust1, ued=ued1,
                                                   nbr_dct=n_dct1,
                                                   normalized=False) + \
                compute_energy_external_partial(spline=spl,
                                                img=img,
                                                ust=ust2, ued=ued2,
                                                nbr_dct=n_dct2,
                                                normalized=False)
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            return compute_energy_external_partial(spline=spl,
                                                   img=img,
                                                   ust=ust, ued=ued,
                                                   nbr_dct=n_dct,
                                                   normalized=False)

    def int_partial(spl, nth):
        if knots[nth] < 0:
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1.0
            n_dct1 = (ued1 - ust1) * n_total
            n_dct2 = (ued2 - ust2) * n_total
            n_dct1 = int(np.ceil(n_dct1))
            n_dct2 = int(np.ceil(n_dct2))
            ret1 = compute_energy_internal_partial(spline=spl,
                                                   ust=ust1, ued=ued1,
                                                   nbr_dct=n_dct1) + \
                compute_energy_internal_partial(spline=spl,
                                                ust=ust2, ued=ued2,
                                                nbr_dct=n_dct2)
            return ret1
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            n_dct = (ued - ust) * n_total
            n_dct = int(np.ceil(n_dct))
            ret2 = compute_energy_internal_partial(spline=spl,
                                                   ust=ust, ued=ued,
                                                   nbr_dct=n_dct)
            return ret2

    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -5.0
    dat_p = np.zeros_like(dat)
    e1e = np.zeros(len(ctrl_pts))
    e1i = np.zeros(len(ctrl_pts))
    l1 = spline.get_length()
    e1t = compute_energy_external_total(spline=spline, img=img,
                                        normalized=False)

    for i in range(len(e1e)):
        e1e[i] = ext_partial(spline, i)
        e1i[i] = int_partial(spline, i)

    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_p[i] = data + eps
        ctrl = np.array(dat_p)
        ctrl = np.reshape(ctrl, ((len(ctrl) / 2), 2))
        spline.set_ctrl_pts(ctrl, only_free=True)
        e2e = ext_partial(spline, i // 2)
        l2 = spline.get_length()
        mat[i] = gamma * (e2e - e1e[i // 2] - e1t * (l2 - l1) / l1) / l2 / eps \
            + (int_partial(spline, i // 2) - e1i[i // 2]) / eps
    mat = mat.reshape((len(mat) / 2, 2))
    return mat
