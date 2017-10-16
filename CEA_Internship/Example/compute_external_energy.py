'''
Created on 6 juil. 2017
Compute the external energy with respect to some parameters

@author: MB251995
'''
import numpy as np
import operator
from scipy import interpolate
from pyleti.geometry.fake_sgeom import sin_rough_iso_circle as rand_circle
from scipy.interpolate import RectBivariateSpline, splprep
from activecontours.spline.bspline_scipy_derivee import BSplineCLG
import matplotlib.pyplot as plt
from pyleti.geometry import SGeom
from pyleti.cvision.imshape import ImShape
from pyleti.geometry._generic import GeomCollection
from pyleti.image.proc import fake_images as fk_img
from pyleti.cvision import lscad
from pyleti.geometry.fake_sgeom import iso_circle as circle
from pyleti.mathutils.spline import equispaced_sampling
from activecontours.energie.energie_interne import compute_energy_internal_total
from activecontours.energie.energie_externe import compute_energy_external_partial, compute_energy_external_total
from activecontours.optimisation.jacobian import jacobian_full_1_side, jacobian_spl_1_side


def create_image(R):
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


def spline_initialize(R):
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
    c = np.array([0, 0])
    n = 10
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


def compute_energy_total(spline, img, nbr_dct):
    # Compute the external energy. More details are given in the src
    [x, y] = np.transpose(spline(np.linspace(0, 1, nbr_dct + 1)))
    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(0, 1,
                                                               nbr_dct + 1)))
    tmp = img(y, x, grid=False) * np.hypot(x1u, y1u)
    return -np.trapz(y=tmp, dx=1.0 / nbr_dct) / np.trapz(y=np.hypot(x1u, y1u),
                                                         dx=1.0 / nbr_dct)


def compute_energy_partial(spline, img, ust, ued, nbr_dct, nbr_total=1000):
    # Compute the partial external energy. More details are given in the src
    [x, y] = np.transpose(spline(np.linspace(ust, ued, nbr_dct + 1)))
    [x1u, y1u] = np.transpose(spline.derivative(1)(np.linspace(ust, ued,
                                                               nbr_dct + 1)))
    [x_s, y_s] = np.transpose(spline.derivative(1)(np.linspace(0, 1,
                                                               nbr_total + 1)))
    tmp = img(y, x, grid=False) * np.hypot(x1u, y1u)
    return -np.trapz(y=tmp, dx=(ued - ust) / nbr_dct) / \
        np.trapz(y=np.hypot(x_s, y_s), dx=1.0 / nbr_total)


def changement_discretisation():
    # How the external energy changes with respect to the level of discretization
    n_dct = np.linspace(start=10, stop=2000, num=100, dtype=np.int16)
    energy = np.zeros(100)
    plt.rcParams["font.family"] = "serif"
    spl = spline_initialize(100)
    img, imsh = create_image(30)
    for i, data in enumerate(n_dct):
        energy[i] = compute_energy_external_partial(spline=spl, img=img,
                                                    ust=0, ued=1.0,
                                                    nbr_dct=data)
    fig = plt.figure()
    fig.canvas.set_window_title("Dicretisation")
    plt.plot(n_dct, energy)
    plt.xlabel('Nombre de point de discretisation')
    plt.ylabel('Energie')
    plt.legend()
    plt.show()


def changement_rayon_externe():
    # How the external energy changes with respect to the radius given that we
    # know the optimal radius
    n_dct = 1000
    img, imsh = create_image(30)
    R = np.linspace(20, 40, 100)
    energy = np.zeros(100)
    for i, dat in enumerate(R):
        spl = spline_initialize(dat)
        energy[i] = compute_energy_external_total(spline=spl, img=img,
                                                  nbr_dct=n_dct)
    fig = plt.figure()
    fig.canvas.set_window_title("Changement de rayon")
    plt.xlabel("Rayon")
    plt.ylabel("Energie")
    plt.plot(R, energy)
    plt.show()


def changement_rayon_total():
    # How the total energy changes with respect to the radius given that we
    # know the optimal radius
    n_dct = 1000
    img, _ = create_image(30)
    R = np.linspace(20, 40, 1000)
    e_ext = np.zeros(1000)
    e_int = np.zeros(1000)
    e_total = np.zeros(1000)
    dict = {}
    gamma = 50
    for i, data in enumerate(R):
        spl = spline_initialize(data)
        e_int[i] = compute_energy_internal_total(spline=spl, ba_ratio=400,
                                                 nbr_dct=n_dct)
        e_ext[i] = compute_energy_external_total(spl, img, n_dct)
        e_total[i] = e_int[i] + gamma * e_ext[i]
        dict[data] = e_total[i]
    print min(dict.iteritems(), key=operator.itemgetter(1))[0]
    fig = plt.figure()
    fig.canvas.set_window_title("Rayon")
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text('Le changement avec gamma = ' + str(gamma))

    ax1.plot(R, e_int, 'r', label='Energie interne')
    ax1.legend(fontsize='large')
    ax2.plot(R, e_ext, 'g', label='Energie externe')
    ax2.legend(fontsize='large')
    ax3.plot(R, e_total, 'b', label='Energie totale')
    ax3.legend(fontsize='large')

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
    ax.set_xlabel('Rayon')
    ax.set_ylabel("Energie")
    plt.show()

if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    changement_rayon_total()
