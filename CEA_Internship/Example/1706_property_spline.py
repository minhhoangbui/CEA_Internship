'''
Created on 27 juin 2017
Given the BSpline and coordinates (x, y), the module illustrates the property
of local pertubation of BSpline if we move a control point to the given
coordinates
@author: MB251995
'''
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep
from activecontours.spline.bspline_scipy_derivee import BSplineCLG


def main():
    # Initialize the spline
    c = np.array([100, 100])
    R = 50
    ctrl_points_nb = 9
    phi = np.linspace(0, 2 * np.pi, ctrl_points_nb + 1)
    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[ctrl_points_nb] = x_init[0]
    y_init[ctrl_points_nb] = y_init[0]
    dat = np.array([x_init, y_init])
    tck, _ = splprep(dat, s=0, per=1, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSplineCLG(knots, ctrl_pts, deg, is_periodic=True)
    go_on = True
    while go_on:
        # Choose the control point that you want to move
        spl = BSplineCLG(knots, ctrl_pts, deg, is_periodic=True)
        nb_dct_total = 1000
        [x, y] = np.transpose(spl(np.linspace(0, 1, nb_dct_total)))
        plt.plot(x, y)
        nth = raw_input('Enter the nth point: ')
        nth = int(nth)
        nbr_lib = len(spl.get_ctrl_pts(only_free=True))
        if nth > nbr_lib:
            raise ValueError('Too big')
        lib = np.transpose(spl.get_ctrl_pts(only_free=True))
        all = plt.scatter(lib[0], lib[1], s=40)
        # Decide the range of influence of the choosen point and plot the figure
        if knots[nth] < 0:
            # For these point satisfying this condition, each of them affect two
            # internal parts in the curve
            ust1 = 0
            ued1 = knots[nth + deg + 1]
            ust2 = knots[nth + nbr_lib]
            ued2 = 1
            nb_dct_1 = (ued1 - ust1) * nb_dct_total
            nb_dct_2 = (ued2 - ust2) * nb_dct_total
            [xth1, yth1] = np.transpose(spl(np.linspace(ust1, ued1, nb_dct_1)))
            [xth2, yth2] = np.transpose(spl(np.linspace(ust2, ued2, nb_dct_2)))
            plt.plot(xth1, yth1, 'r')
            plt.plot(xth2, yth2, 'r')
            lib_pts = spl.get_ctrl_pts(only_free=True)
            old = plt.scatter(lib_pts[nth, 0], lib_pts[nth, 1], s=40, color='y')
            lib_pts[nth] = [110, 250]
            new = plt.scatter(lib_pts[nth, 0], lib_pts[nth, 1], s=40, color='r')
            spl.set_ctrl_pts(lib_pts, only_free=True)
            [xth_a1, yth_a1] = np.transpose(spl(np.linspace(ust1, ued1, nb_dct_1)))
            [xth_a2, yth_a2] = np.transpose(spl(np.linspace(ust2, ued2, nb_dct_2)))
            plt.plot(xth_a1, yth_a1, 'g')
            plt.plot(xth_a2, yth_a2, 'g')
            plt.scatter(spl(ust2)[0], spl(ust2)[1], s=40, color='c',
                        marker="*")
            plt.scatter(spl(ued1)[0], spl(ued1)[1], s=40, color='m',
                        marker="*")
        else:
            ust = knots[nth]
            ued = knots[nth + deg + 1]
            nb_dct = (ued - ust) * nb_dct_total
            [xth, yth] = np.transpose(spl(np.linspace(ust, ued, nb_dct)))
            plt.plot(xth, yth, 'r')
            lib_pts = spl.get_ctrl_pts(only_free=True)
            old = plt.scatter(lib_pts[nth, 0], lib_pts[nth, 1], s=40, color='y')
            lib_pts[nth] = [110, 250]
            new = plt.scatter(lib_pts[nth, 0], lib_pts[nth, 1], s=40, color='r')
            spl.set_ctrl_pts(lib_pts, only_free=True)
            [xth_a, yth_a] = np.transpose(spl(np.linspace(ust, ued, nb_dct)))
            plt.plot(xth_a, yth_a, 'g')
            plt.scatter(spl(ust)[0], spl(ust)[1], s=70, color='c',
                        marker="*")
            plt.scatter(spl(ued)[0], spl(ued)[1], s=70, color='m', marker="*")
        plt.legend((all, old, new), ('All control point',
                                     'Old moving point',
                                     'New moving point'))
        plt.savefig(r'\\bkshiva\LCshareNFS002\500.14-Litho\CLG\projects\2017\MB-METRO\MS-1703-ACTIVE_CONTOURS\Figure\MovingControlPoint.png')
        plt.show()
        exit = raw_input('Quitter ou pas ?: ')
        if exit == 'y' or exit == 'Y':
            go_on = False


if __name__ == '__main__':
    # pylint: disable=I0011,C0103
    main()
