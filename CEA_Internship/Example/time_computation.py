'''
Created on 4 juil. 2017
Computing the computation time of internal and external energy with the help
of interpolation tools like:
- RectBivariateSpline
- RegularGridInterpolator
@author: MB251995
'''
from activecontours.energie.energie_externe import compute_energy_external_total
from activecontours.energie.energie_interne import compute_energy_internal_total
from activecontours.spline.bspline_scipy_derivee import BSpline_periodic
import numpy as np
from scipy.interpolate import splprep
import timeit
from pyleti.geometry import SGeom
from pyleti.cvision import lscad
from pyleti.image.proc import fake_images as fk_img
from shapely.geometry.geo import box
from scipy import interpolate


def initialize(R):
    c = np.array([256, 256])
    n = 20
    phi = np.linspace(0, 2 * np.pi, n + 1)
    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[n] = x_init[0]
    y_init[n] = y_init[0]
    dat = np.array([x_init, y_init])
    tck, _ = splprep(dat, s=0, per=1, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSpline_periodic(knots, ctrl_pts, deg, is_periodic=True)
    return spl


def create_RVSpine():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    my_sgeom = SGeom(box(-100, -50, 100, 50))
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    return img_xy

def create_RGInterp():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    my_sgeom = SGeom(box(-100, -50, 100, 50))
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RegularGridInterpolator((x, y), np.flipud(img.arr))
    return img_xy


def compute_internal():
    st = timeit.default_timer()
    for i in range(50):
        compute_energy_internal_total(initialize(50))
    ed = timeit.default_timer()
    print ed - st


def compute_external():
    spl = initialize(50)
    img_r = create_RVSpine()
    st = timeit.default_timer()
    for i in range(10):
        compute_energy_external_total(spline=spl,
                                      img=img_r,
                                      nbr_dct=100)
    ed = timeit.default_timer()
    print ed - st

def compare_regu_rectbivar():
    reg = create_RGInterp()
    rect = create_RVSpine()
    st1 = timeit.default_timer()
    for i in range(1000):
        reg([245, 244])
    ed1 = timeit.default_timer()
    st2 = timeit.default_timer()
    for i in range(1000):
        rect(245, 244)
    ed2 = timeit.default_timer()
    print ed1 - st1
    print ed2 - st2


def time_RVSpline():
    spl = initialize(50)
    img = create_RVSpine()
    st = timeit.default_timer()
    for i in range(10):
        [x, y] = np.transpose(spl(np.linspace(0, 1, 1000)))
        tmp = img(y, x, grid=False)
        - np.sum(tmp)
    ed = timeit.default_timer()
    print ed - st


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    compare_regu_rectbivar()

