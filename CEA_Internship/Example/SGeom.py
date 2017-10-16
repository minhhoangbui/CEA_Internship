'''
Created on 28 juin 2017
Create some images with different contours with SGeom
@author: MB251995
'''

from pyleti import conf
from shapely.geometry.geo import Polygon
from shapely.geometry.geo import box
import matplotlib.pyplot as plt
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
import numpy as np
from pyleti.cvision import lscad
from pyleti.geometry._generic import GeomCollection
from pyleti.image.proc import fake_images as fk_img
from pyleti.geometry.fake_sgeom import iso_circle as circle, sin_rough_iso_circle as rand_circle
from shapely.geometry.multipolygon import MultiPolygon

def create_circle_1():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    R = 25
    my_sgeom = circle(R, 0, 0)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)

    img.plot()
    plt.show()


def create_circle_2():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    R = 25
    c = np.array([0, 0])
    phi = np.linspace(0, 2 * np.pi, 1000)
    x = c[0] + R * np.cos(phi)
    y = c[1] + R * np.sin(phi)
    dat = np.array([x, y]).T
    my_sgeom = SGeom(Polygon(dat))
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)

    img.plot()
    plt.show()

def create_rect():
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
    img.plot()
    plt.show()


def plot_coll():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    R = 25
    c = np.array([0, 0])
    phi = np.linspace(0, 2 * np.pi, 1000)
    x = c[0] + R * np.cos(phi)
    y = c[1] + R * np.sin(phi)
    dat = np.array([x, y]).T
    my_sgeom = SGeom(Polygon(dat))
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)
    imsh.plot()
    plt.show()


def create_fluctuation():
    img_shape = (512, 512)
    frame_bounds = [-256, 256, -256, 256]
    R = 25
    r_nb = 150
    rough_dict = {'lwr': 10}
    r_per = np.pi / 3
    my_sgeom = rand_circle(R, 0, 0,
                           r_nb, rough_dict, r_per)
    strat = lscad.MountainStrategy()
    param_dict = {'hmax': 100., 'sigma': (0, 0)}
    img = fk_img.img_from_sgeom(img_shape,
                                frame_bounds,
                                my_sgeom,
                                z_map_strategy=strat,
                                param_dict=param_dict)
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)
    imsh.plot()
    plt.show()

def create_2ob():
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
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)
    imsh.plot()
    plt.show()


def create_noisy_circle(R):
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
    mu, sigma = 0, 10
    noise = np.random.normal(mu, sigma, img.arr.shape).astype(float)
    noisy_img = img.arr + noise
    x, y = img.arrf.sampling_vectors()
    img_xy = interpolate.RectBivariateSpline(x, y, np.flipud(noisy_img))
    gcoll = GeomCollection()
    gcoll.set_geom(0, my_sgeom)
    imsh = ImShape(image=img, geom_coll=gcoll)
    imsh.plot()
    plt.show()


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    create_noisy_circle(50)

