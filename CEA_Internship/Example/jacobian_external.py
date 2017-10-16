'''
Created on 11 juil. 2017
Justify the similarity between simplified external jacobian and the traditional
one.
@author: MB251995
'''
import numpy as np
from activecontours.energie.energie_externe import \
                        compute_energy_external_partial
from activecontours.energie.energie_externe import \
                        compute_energy_external_total
from activecontours.energie.energie_interne import compute_energy_internal_partial
from activecontours.energie.energie_interne import compute_energy_internal_total
import copy
from activecontours.spline.bspline_scipy_derivee import BSpline_periodic
from scipy.interpolate import splprep, RectBivariateSpline
from scipy.interpolate import splprep, BSpline, RectBivariateSpline
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
from pyleti.geometry._generic import GeomCollection
from pyleti.image.proc import fake_images as fk_img
from pyleti.cvision import lscad
from pyleti.geometry.fake_sgeom import iso_circle as circle
import matplotlib.pyplot as plt
import timeit


def jacobian_full_1_side(spline_o, img):
    '''
    Construct a external jacobian with traditional method
    
    Parameters
    ----------
    spline_o : BSpline_periodic
        A well-defined BSpline
    img: numpy.narray
        An image
    
    Returns
    -------
    mat : numpy.ndarray
        Matrix of jacobian
    '''

    spline = copy.deepcopy(spline_o)
    ctrl_pts = spline.get_ctrl_pts(only_free=True)
    dat = ctrl_pts.flatten()
    mat = np.zeros_like(dat)
    eps = 10.0 ** -5.0
    dat_p = np.zeros_like(dat)

    def cost(ctrl):
        ctrl = np.array(ctrl)
        ctrl = np.reshape(ctrl, ((len(ctrl) / 2), 2))
        spline.set_ctrl_pts(ctrl, only_free=True)

        return compute_energy_external_total(spline=spline, img=img,
                                             nbr_dct=10000,
                                             normalized=True) + \
            compute_energy_internal_total(spline=spline, nbr_dct=10000)

    cost_act = cost(dat)

    for i, data in enumerate(dat):
        dat_p[:] = dat
        dat_p[i] = data + eps
        mat[i] = (cost(dat_p) - cost_act) / eps
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def jacobian_spl_1_side(spline_o, img):
    '''
    Construct a external jacobian with simplified method
    
    Parameters
    ----------
    spline_o : BSpline_periodic
        A well-defined BSpline
    img: numpy.narray
        An image
    
    Returns
    -------
    mat : numpy.ndarray
        Matrix of jacobian
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
        mat[i] = (e2e - e1e[i // 2] - e1t * (l2 - l1) / l1) / l2 / eps + \
            (int_partial(spline, i // 2) - e1i[i // 2]) / eps
    mat = mat.reshape((len(mat) / 2, 2))
    return mat


def initialize():
    # initialize the spline
    c = np.array([0, 0])
    R = 100.0
    nbr_init = 30
    phi = np.linspace(0, 2.0 * np.pi, nbr_init)
    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[nbr_init - 1] = x_init[0]
    y_init[nbr_init - 1] = y_init[0]
    tck, _ = splprep([x_init, y_init], s=0, per=1)
    spl = BSpline_periodic(tck[0], np.transpose(tck[1]), tck[2],
                           is_periodic=True)
    # create an image
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    my_sgeom = circle(100, 0, 0)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    tmp = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)

    x, y = tmp.arrf.sampling_vectors()
    img = RectBivariateSpline(x, y, np.flipud(tmp.arr))
    return spl, img


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    spl, img = initialize()
    st1 = timeit.default_timer()
    for i in range(10):
        jacobian_full_1_side(spline_o=spl, img=img)
    ed1 = timeit.default_timer()
    st2 = timeit.default_timer()
    for i in range(10):
        jacobian_spl_1_side(spline_o=spl, img=img)
    ed2 = timeit.default_timer()
    print ed1 - st1
    print ed2 - st2
#     print np.abs(jacobian_full_1_side(spline_o=spl, img=img) -
#                  jacobian_spl_1_side(spline_o=spl, img=img)) / \
#                  jacobian_full_1_side(spline_o=spl, img=img)
