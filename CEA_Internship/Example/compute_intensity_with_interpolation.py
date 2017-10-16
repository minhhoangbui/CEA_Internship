'''
Created on 3 juil. 2017
Some tests to master the interpolation with image from pyleti in order to
compute the energy during the convergence 
@author: MB251995
'''
import unittest
import numpy as np
from shapely.geometry.geo import box
import matplotlib.pyplot as plt
from pyleti import conf
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
from pyleti.geometry._generic import GeomCollection
from pyleti.cvision import lscad
from pyleti.image.proc import fake_images as fk_img
from scipy.interpolate import interp2d, RectBivariateSpline, RegularGridInterpolator, splprep
from activecontours.spline.bspline_scipy_derivee import BSplineCLG

def comparer():
    '''
    Create an image and compare the interpolation from different methods
    '''
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
    x_s, y_s = img.arrf.sampling_vectors()
    r, c = img.xy_to_rc(100, 50)
    interp = interp2d(x_s, y_s, np.flipud(img.arr))
    bivar1 = RectBivariateSpline(x=x_s, y=y_s, z=np.flipud(img.arr))
    bivar2 = RectBivariateSpline(x=y_s, y=x_s, z=np.flipud(img.arr))
    print interp(100, 50)
    print img.arr[r, c]
    print bivar1(50, 100), bivar1(100, 50)
    print bivar2(50, 100), bivar2(100, 50)
    img.plot()
    plt.scatter(100, 50)
    plt.show()

def interp_array():
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
    x_s, y_s = img.arrf.sampling_vectors()
    bivar = RectBivariateSpline(x=x_s, y=y_s, z=np.flipud(img.arr))
    x_n = np.linspace(0, 100, 10)
    y_n = np.linspace(100, 200, 10)
    print bivar(x_n, y_n, grid=False)
    print bivar(0, y_n)


def initialize(R):
    '''
    Initialize the BSpline having the form of circle which has a radius R
    
    Parameters
    ----------
    R : float
        Radius of the circle
    
    Returns
    -------
    spl : BSpline_periodic
        A well-defined spline curve
    '''
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
    spl = BSplineCLG(knots, ctrl_pts, deg, is_periodic=True)
    return spl


def regu_interp():
    # Comparison between RegularGridInterpolator and RectBivariateSpline
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
    x_s, y_s = img.arrf.sampling_vectors()
    reg = RegularGridInterpolator((x_s, y_s), np.flipud(img.arr))
    bivar = RectBivariateSpline(x=x_s, y=y_s, z=np.flipud(img.arr))
    spl = initialize(40)
    pts = spl(np.linspace(0, 1, 10))
    print pts[0]
    print bivar(245, 244)
    print reg([245, 244])


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    regu_interp()
