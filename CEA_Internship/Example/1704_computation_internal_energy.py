'''
Created on 21 juin 2017
Some tests which shows the influence of the number of initializing point,
discretising point and the size of curve to the precision of the computation
E_internal(v) = alpha * int( (dv / ds)^2 + beta * (d^2v / ds^2)^2 ds
@author: MB251995
'''
import numpy as np
from activecontours.energie.energie_interne import compute_energy_internal_total_seperate
from scipy.interpolate import splprep, BSpline
import matplotlib.pyplot as plt


def test_nbr_init():
    '''
    Verify the influence of the number of initialized point to the relative
    error between the numerical computation and analytical computation
    '''
    # n_pts: The number of initialized point
    n_pts = np.linspace(start=100, stop=3000, num=30, dtype=np.int16)
    # c: center of the circle
    # R: Initial radius of the circle
    c = np.array([0, 0])
    R = 100.0
    # en1, en2, en: The numerical result of the first and the second term
    # and the total internal energy
    en1 = np.zeros(30)
    en2 = np.zeros(30)
    en = np.zeros(30)
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(n_pts):
        phi = np.linspace(0., 2.0 * np.pi, data)
        x_init = c[0] + R * np.cos(phi)
        y_init = c[1] + R * np.sin(phi)
        x_init[data - 1] = x_init[0]
        y_init[data - 1] = y_init[0]
        tck, _ = splprep([x_init, y_init], s=0)
        spl = BSpline(tck[0], np.transpose(tck[1]), tck[2])
        en1[i], en2[i] = compute_energy_internal_total_seperate(spl)
        en[i] = en1[i] + en2[i]
    # e_a1, e_a2, en: The analytical result of the first and the second term of
    # internal energy
    e_a1 = 2.0 * np.pi * R
    e_a2 = 2.0 * np.pi / R
    fig = plt.figure()
    fig.canvas.set_window_title("Initialisation")
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text('Les erreurs relatives avec R = 100.0')

    ax1.plot(n_pts, 100 * np.absolute(en1 - e_a1) / e_a1, 'r',
             label=r'$\frac{\partial v}{\partial s}$')
    ax1.legend(fontsize='large')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(n_pts, 100 * np.absolute(en2 - e_a2) / e_a2, 'g',
             label=r'$\frac{\partial^2 v}{\partial s^2}$')
    ax2.legend(fontsize='large')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.plot(n_pts, 100 * np.absolute(en1 + en2 - e_a1 - e_a2) /
             (e_a1 + e_a2), 'b', label="Energie interne")
    ax3.legend()
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
    ax.set_xlabel('Nombre de point d\'initialisation')
    ax.set_ylabel("Erreurs")
    plt.savefig(r'\\bkshiva\LCshareNFS002\500.14-Litho\CLG\projects\2017\MB-METRO\MS-1703-ACTIVE_CONTOURS\Figure\Computation of internal energy\Changement_nbr_init.png')
    plt.show()


def test_nbr_dct():
    '''
    Verify the influence of the number of discretized point to the relative
    error between the numerical computation and analytical computation
    '''
    # n_pts: The number of initialized point
    n_init = 1000
    # c: center of the circle
    # R: Initial radius of the circle
    c = np.array([0, 0])
    R = 100.0
    phi = np.linspace(0., 2.0 * np.pi, n_init)

    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[n_init - 1] = x_init[0]
    y_init[n_init - 1] = y_init[0]
    tck, _ = splprep([x_init, y_init], s=0)
    spl = BSpline(tck[0], np.transpose(tck[1]), tck[2])

    n = np.linspace(start=10, stop=2000, num=100, dtype=np.int16)
    # e_a1, e_a2, en: The analytical result of the first and the second term
    # of internal energy
    e_a1 = 2.0 * np.pi * R
    e_a2 = 2.0 * np.pi / R
    # en1, en2, en: The numerical result of the first and the second term
    # of internal energy
    en1 = np.zeros(len(n))
    en2 = np.zeros(len(n))
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(n):
        en1[i], en2[i] = compute_energy_internal_total_seperate(spline=spl,
                                                                nbr_dct=data)

    fig = plt.figure()
    fig.canvas.set_window_title("Dicretisation")
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text('Les erreurs relatives avec R = 100.0')

    ax1.plot(n, 100 * np.abs(en1 - e_a1) / e_a1, 'r',
             label=r'$\frac{\partial v}{\partial s}$')
    ax1.legend(fontsize='large')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(n, 100 * np.abs(en2 - e_a2) / e_a2, 'g',
             label=r'$\frac{\partial^2 v}{\partial s^2}$')
    ax2.legend(fontsize='large')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.plot(n, 100 * np.abs(en1 + en2 - e_a1 - e_a2) / (e_a1 + e_a2), 'b',
             label="Energie interne")
    ax3.legend()
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
    ax.set_xlabel("Logarithme de nombre de point de discretisation")
    ax.set_ylabel("Erreurs")
    plt.savefig(r'\\bkshiva\LCshareNFS002\500.14-Litho\CLG\projects\2017\MB-METRO\MS-1703-ACTIVE_CONTOURS\Figure\Computation of internal energy\Changement_nbr_dct.png')
    plt.show()


def test_R():
    '''
    Verify the variability of internal energy with respect to radius
    '''
    # n_pts: The number of initialized point
    n_pts = 5000
    # c: center of the circle
    # R: Initial radius of the circle
    c = np.array([0, 0])
    R = np.array([10, 20, 50, 70, 100, 120, 150, 180, 200, 230, 250, 300, 350,
                  400, 450, 500])
    phi = np.linspace(0., 2.0 * np.pi, n_pts)
    x_init = np.zeros((len(R), n_pts))
    y_init = np.zeros((len(R), n_pts))
    # e_a1, e_a2, en: The analytical result of the first and the second term of
    # internal energy
    en1 = np.zeros(len(R))
    en2 = np.zeros(len(R))
    en = np.zeros(len(R))
    plt.rcParams["font.family"] = "serif"
    for i, data in enumerate(R):
        x_init[i, :] = c[0] + data * np.cos(phi)
        y_init[i, :] = c[1] + data * np.sin(phi)
        x_init[i, n_pts - 1] = x_init[i, 0]
        y_init[i, n_pts - 1] = y_init[i, 0]
        tck, _ = splprep([x_init[i, :], y_init[i, :]], s=0)
        spl = BSpline(tck[0], np.transpose(tck[1]), tck[2])
        en1[i], en2[i] = compute_energy_internal_total_seperate(spl)
        en[i] = en1[i] + 10000 * en2[i]

    fig = plt.figure("Rayon")
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text("L'energie avec n = 3000, \n beta = 10000 * alpha")

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')

    ax1.plot(R, en1, 'b', label=r'$\frac{\partial v}{\partial s}$')
    ax2.plot(R, en2, 'g', label=r'$\frac{\partial^2 v}{\partial s^2}$')
    ax3.plot(R, en, 'r', label="Energie interne")
    ax1.legend(fontsize='large')
    ax2.legend(fontsize='large')
    ax3.legend()

    ax.set_xlabel("Rayon")
    ax.set_ylabel("Energie")
    plt.savefig(r'\\bkshiva\LCshareNFS002\500.14-Litho\CLG\projects\2017\MB-METRO\MS-1703-ACTIVE_CONTOURS\Figure\Computation of internal energy\Changement_R.png')
    plt.show()


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    test_R()
