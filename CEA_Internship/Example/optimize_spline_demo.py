'''
Created on 28 juil. 2017
Using the simplified jacobian for the optimisation with scipy

@author: MB251995
'''
import numpy as np
from scipy.interpolate import splprep
from activecontours.spline.bspline_scipy_derivee import BSpline_periodic
from pyleti import conf
from shapely.geometry.geo import Polygon
import matplotlib.pyplot as plt
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
from pyleti.geometry._generic import GeomCollection
from pyleti.cvision import lscad
from pyleti.image.proc import fake_images as fk_img
from scipy.interpolate import interp2d, RectBivariateSpline
from pyleti.geometry.fake_sgeom import iso_circle as circle, sin_rough_iso_circle as rand_circle
import timeit
from scipy import interpolate
from pyleti.mathutils.core import distance_between_curves
from shapely.geometry.geo import box
from shapely.geometry.multipolygon import MultiPolygon
from pyleti.mathutils import spline
from pyleti.mathutils.core import distance_between_curves
import copy
from activecontours.energie.energie_interne import compute_energy_internal_total
from activecontours.optimisation.optimisation_module import optimize_simplified


def create_circle(R):
    '''
    Create a image in which the maximum pixels form a circle having a radius R
    
    Parameters
    ----------
    R : float
        Radius of the circle
    
    Returns
    -------
    img_xy : RectBivariate object
        A narray whose values at non-integer coordinates are available thanks
        to interpolation
    imsh : ImShape object
        A object which helps to display the contour and the image
    '''
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    my_sgeom = circle(R, 0, 0)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)

    return img_xy, imsh


def create_vague(R, per=np.pi / 3):
    '''
    Create a image in which the maximum pixels form a circular wave having a 
    radius R
    
    Parameters
    ----------
    R : float
        Radius of the circle
    per: float
        Periodicity of the wave
    
    Returns
    -------
    img_xy : RectBivariate object
        A narray whose values at non-integer coordinates are available thanks
        to interpolation
    imsh : ImShape object
        A object which helps to display the contour and the image
    '''
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    r_nb = 200
    rough_dict = {'lwr': 10}
    my_sgeom = rand_circle(R, 0, 0,
                           r_nb, rough_dict, per)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)

    return img_xy, imsh


def create_circle_local(R, threshold=50):
    '''
    Create a image in which the maximum pixels form a circle having a radius R.
    Next we will create a constant zone in which all the pixel are set to 0

    Parameters
    ----------
    R : float
        Radius of the circle
    threshold: float, optional
        Every pixel of which the value is less than this threshold is set to 0
    Returns
    -------
    img_xy : RectBivariate object
        A narray whose values at non-integer coordinates are available thanks
        to interpolation
    imsh : ImShape object
        A object which helps to display the contour and the image
    '''
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    my_sgeom = circle(R, 0, 0)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)

    img.arr[img.arr < threshold] = 0
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)
    return img_xy, imsh


def spline_initialize(R, n_init):
    '''
    Initialize the BSpline having the form of circle which has a radius R
    
    Parameters
    ----------
    R : float
        Radius of the circle
    n_init: integer
        Number of initial points which are used to create spline curve
    
    Returns
    -------
    spl : BSpline_periodic
        A well-defined spline curve
    '''
    c = np.array([0, 0])
    phi = np.linspace(0, 2 * np.pi, n_init + 1)
    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[n_init] = x_init[0]
    y_init[n_init] = y_init[0]
    dat = np.array([x_init, y_init])
    tck, _ = splprep(dat, s=0, per=1, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSpline_periodic(knots, ctrl_pts, deg, is_periodic=True)
    return spl


def optimize_image():
    '''
    A module of optimisation for the Active Contour. However, it is only 
    effective for the simplified method of jacobian
    '''
    spl = spline_initialize(R=200, n_init=10)
    img, imsh = create_vague(R=50)
    imsh.plot(label='Vrai Contour')
    opti = optimize_simplified(spline=spl, method='slsqp', img=img,
                               ba_ratio=100, gamma=10)
    lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
    spl.set_ctrl_pts(lst, only_free=True)
    spl.plot_spline(label='Courbe finale')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    optimize_image()
