# coding: utf-8
'''
Created on 28 juin 2017
Use the optimisation to find the different forms of contour in image with
traditional method of jacobian 
@author: MB251995
'''
import numpy as np
from scipy.interpolate import splprep
from activecontours.spline.bspline_scipy_derivee import BSpline_periodic
from activecontours.optimisation.optimisation_module import optimize_traditional as optimize
from pyleti import conf
from shapely.geometry.geo import Polygon
import matplotlib.pyplot as plt
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
from pyleti.geometry._generic import GeomCollection
from pyleti.cvision import lscad
from pyleti.image.proc import fake_images as fk_img
from activecontours.energie.energie_externe import compute_energy_external_total
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


def create_vague(R, per=np.pi / 3, lwr=40):
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
    rough_dict = {'lwr': lwr}
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

def create_noisy_circle(R, sigma):
    '''
    Create a image in which the maximum pixels form a circle having a radius R.
    However, we add a Gaussian noise with muy = 0
    
    Parameters
    ----------
    R : float
        Radius of the circle
    sigma: float
        Variance of the Gaussian noise
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

    noise = np.random.normal(0, sigma, img.arr.shape).astype(float)
    img.arr = img.arr + noise
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


def create_vague_local(R, threshold=50, per=np.pi / 3):
    '''
    Create a image in which the maximum pixels form a circular wave having a 
    radius R
    
    Parameters
    ----------
    R : float
        Radius of the circle
    threshold: float, optional
        Every pixel of which the value is less than this threshold is set to 0
    per: float, optional
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
    img.arr[img.arr < threshold] = 0
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)

    return img_xy, imsh


def create_rect():
    '''
    Create a image in which the maximum pixels form a rectangular
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
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(img.arr))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)

    return img_xy, imsh


def create_2ob():
    '''
    Create a image in which the maximum pixels form 2 objects
    '''
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    R = 25
    my_sgeom_1 = circle(R, -100, 0)
    my_sgeom_2 = SGeom(box(0, 0, 100, 50))
    my_geom = MultiPolygon([my_sgeom_1.geom, my_sgeom_2.geom])
    my_sgeom = SGeom(my_geom, label='circle_square')
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


def compute_external(R):
    c = np.array([256, 256])
    n = 25
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
    img, imsh = create_vague()
    return compute_energy_external_total(spl, img=img)


def optimize_image(jaco):
    '''
    A module of optimisation for the Active Contour. However, it is only 
    effective for the traditional method of jacobian
    '''
    spl = spline_initialize(R=200, n_init=7)
    img, imsh = create_vague(50)
    imsh.plot(label='Vrai contour')
    spl.plot_spline(label='Courbe initiale')
    st = timeit.default_timer()
    opti = optimize(spline=spl, method='slsqp', img=img,
                    gamma=75., ba_ratio=100,
                    jac=jaco)
#     ed = timeit.default_timer()
#     print ed - st
    lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
    spl.set_ctrl_pts(lst, only_free=True)
    spl.plot_spline(label='Courbe finale')
    [x, y] = lst.T
    plt.scatter(x, y)
    plt.title(u"Nombre de points de contrôle = 7" + r', $\gamma = 75$' +
              r', $\beta/\alpha = 100$')
#     plt.title(u'Rugosité = 60')
    plt.legend()
    plt.show()


def distance_vs_gamma(ba, n_init):
    '''
    Computing the distance between the target and the final curve  with respect
    to gamma. In this function we fix the beta/alpha and number of initial 
    points
    '''
    spl = spline_initialize(R=200, n_init=n_init)
    img, imsh = create_vague(R=50)
    gamma_lst = np.linspace(40, 200, 20)
    dist = np.zeros_like(gamma_lst)
    pts_ref = get_pts_from_sgeom_vague(R=50)
    plt.rcParams["font.family"] = "serif"
    for i, dat in enumerate(gamma_lst):
        spl_c = copy.deepcopy(spl)
        opti = optimize(spline=spl_c, method='slsqp', img=img,
                        gamma=dat, ba_ratio=ba,
                        jac=None)
        lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
        spl_c.set_ctrl_pts(lst, only_free=True)
        pts = spl_c(np.linspace(0, 1, 10000))
        dist[i] = distance_between_curves(pts_ref, pts)
    plt.plot(gamma_lst, dist)
    plt.title(r"$\beta/\alpha = $" + str(ba) +
              u', nombre de points de contrôle = ' + str(n_init))
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Distance')
    plt.show()


def distance_vs_n_init(ba, gamma):
    '''
    Computing the distance between the target and the final curve  with respect
    to number of initial points. In this function we fix the beta/alpha and
    gamma
    '''
    nbr_lst = np.linspace(5, 40, 20)
    img, _ = create_vague(R=50)
    dist = np.zeros_like(nbr_lst)
    nbr_lst = nbr_lst.astype(int)
    pts_ref = get_pts_from_sgeom_vague(R=50)
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(nbr_lst):
        spl = spline_initialize(R=200, n_init=data)
        opti = optimize(spline=spl, method='slsqp', img=img,
                        gamma=gamma, ba_ratio=ba,
                        jac=None)
        lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
        spl.set_ctrl_pts(lst, only_free=True)
        pts = spl(np.linspace(0, 1, 10000))
        dist[i] = distance_between_curves(pts_ref, pts)
    plt.plot(nbr_lst, dist)
    plt.title(r"$\beta/\alpha = $" + str(ba) + r', $\gamma = $' + str(gamma))
    plt.xlabel(u'Nombre de points de contrôle')
    plt.ylabel('Distance')
    plt.show()


def distance_vs_ba(n_init, gamma):
    '''
    Computing the distance between the target and the final curve  with respect
    to beta/alpha. In this function we fix the number of initial points and
    gamma
    '''
    ba_lst = np.linspace(100, 1000, 11)
    img, _ = create_vague(R=50)
    spl = spline_initialize(R=200, n_init=n_init)
    dist = np.zeros_like(ba_lst)
    pts_ref = get_pts_from_sgeom_vague(R=50)
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(ba_lst):
        spl_c = copy.deepcopy(spl)
        opti = optimize(spline=spl_c, method='slsqp', img=img,
                        jac=None, ba_ratio=data, gamma=gamma)
        lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
        spl.set_ctrl_pts(lst, only_free=True)
        pts = spl(np.linspace(0, 1, 10000))
        dist[i] = distance_between_curves(pts_ref, pts)
    plt.plot(ba_lst, dist)
    plt.title(u"Nombre de points de contrôle = " + str(n_init) + r', $\gamma = $' +
              str(gamma))
    plt.xlabel(r'$\beta / \alpha$')
    plt.ylabel('Distance')
    plt.show()


def distance_vs_period(n_init, gamma, ba):
    '''
    Computing the distance between the target and the final curve  with respect
    to periodicity in case we use the image of circular wave. In this function
    we fix the number of initial points, gamma, beta/alpha
    '''
    per_lst = 2.0 * np.pi / np.arange(6, 13)
    spl = spline_initialize(R=200, n_init=n_init)
    dist = np.zeros_like(per_lst)
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(per_lst):
        spl_c = copy.deepcopy(spl)
        pts_ref = get_pts_from_sgeom_vague(R=50, per=data)
        img, _ = create_vague(R=50, per=data)
        opti = optimize(spline=spl_c, method='slsqp', img=img,
                        jac=None, ba_ratio=data, gamma=gamma)
        lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
        spl.set_ctrl_pts(lst, only_free=True)
        pts = spl(np.linspace(0, 1, 10000))
        dist[i] = distance_between_curves(pts_ref, pts)
    plt.plot(per_lst, dist)
    plt.title(u"Nombre de points de contrôle = " + str(n_init) + r', $\gamma = $' +
              str(gamma) + r', $\beta/\alpha = $' + str(ba))
    plt.xlabel('Period')
    plt.ylabel('Distance')
    plt.show()


def distance_vs_roughness(n_init, gamma, ba):
    '''
    Computing the distance between the target and the final curve  with respect
    to roughness of circular wave. In this function we fix the number of 
    initial points, gamma, beta/alpha and the peiodicity
    '''
    lwr_lst = np.linspace(10, 50, 10)
    spl = spline_initialize(R=200, n_init=n_init)
    dist = np.zeros_like(lwr_lst)
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(lwr_lst):
        spl_c = copy.deepcopy(spl)
        pts_ref = get_pts_from_sgeom_vague(R=50, lwr=data)
        img, _ = create_vague(R=50, lwr=data)
        opti = optimize(spline=spl_c, method='slsqp', img=img,
                        jac=None, ba_ratio=data, gamma=gamma)
        lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
        spl.set_ctrl_pts(lst, only_free=True)
        pts = spl(np.linspace(0, 1, 10000))
        dist[i] = distance_between_curves(pts_ref, pts)
    plt.plot(lwr_lst, dist)
    plt.title(u"Nombre de points de contrôle = " + str(n_init) + r', $\gamma = $' +
              str(gamma) + r', $\beta/\alpha = $' + str(ba))
    plt.xlabel(u'Rugosité')
    plt.ylabel('Distance')
    plt.show()


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


def get_pts_from_sgeom_vague(R, per=np.pi / 3, lwr=40):
    '''
    Get a referenced points from Sgeom object in order to compute the distance
    between the target and the final curve. In this case, it is from circular
    wave image
    '''
    r_nb = 200
    rough_dict = {'lwr': lwr}
    my_sgeom = rand_circle(R, 0, 0,
                           r_nb, rough_dict, per)
    xy_arr = my_sgeom.get_xy()[0]
    x_arr = xy_arr[:, 0]
    y_arr = xy_arr[:, 1]
    tck, _ = spline.build_spline(x_arr, y_arr, per=True, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSpline_periodic(knots, ctrl_pts, deg, is_periodic=True)
    return spl(np.linspace(0, 1, 10000))

def get_pts_from_sgeom_circle(R):
    '''
    Get a referenced points from Sgeom object in order to compute the distance
    between the target and the final curve. In this case, it is from image of
    circle
    '''
    sgeom = circle(R, 0, 0)
    xy_arr = sgeom.get_xy()[0]
    x_arr = xy_arr[:, 0]
    y_arr = xy_arr[:, 1]
    tck, _ = spline.build_spline(x_arr, y_arr, per=True, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSpline_periodic(knots, ctrl_pts, deg, is_periodic=True)
    return spl(np.linspace(0, 1, 10000))


def distance(ba, gamma, n_init, per=np.pi / 3):
    '''
    Computing the distance between the final curve and the contour with respect
    to parameters like ba_ratio, gamma, number of initial points
    '''
    ref_pts = get_pts_from_sgeom_circle(R=50)
    spl = spline_initialize(R=200, n_init=n_init)
    img, _ = create_circle(R=50)
    opti = optimize(spline=spl, method='slsqp', img=img,
                    jac=None, ba_ratio=ba, gamma=gamma)
    lst = np.reshape(opti.x, (len(opti.x) / 2, 2))
    spl.set_ctrl_pts(lst, only_free=True)
    pts = spl(np.linspace(0, 1, 3000))
    dist = distance_between_curves(ref_pts, pts)
    print dist


if __name__ == '__main__':
#     distance_vs_roughness(n_init=20, gamma=75, ba=100)
    optimize_image(jaco=None)

