'''
Created on 25 juil. 2017

Illustrate the value of basis functions with respect to knots.

@author: MB251995
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, BSpline

if __name__ == '__main__':
    # Initialize the spline
    c = np.array([100, 100])
    R = 50
    n = 10
    phi = np.linspace(0, 2 * np.pi, n)
    x_init = c[0] + R * np.cos(phi)
    y_init = c[1] + R * np.sin(phi)
    x_init[n - 1] = x_init[0]
    y_init[n - 1] = y_init[0]
    dat = np.array([x_init, y_init])
    tck, _ = splprep(dat, s=0, per=0, k=3)
    knots = tck[0]
    ctrl_pts = np.transpose(tck[1])
    deg = tck[2]
    spl = BSpline(knots, ctrl_pts, deg)
    fig, ax = plt.subplots()
    # Compute the value of each basis function
    for knot_ind in np.arange(len(knots)):
        knots_basis = spl.t[knot_ind: knot_ind + deg + 2]
        b = spl.basis_element(knots_basis, extrapolate=False)
        x = np.linspace(knots_basis[0], knots_basis[-1], 50)
        ax.plot(x, b(x), lw=3)
        ax.plot([knots_basis[0]] * 2, [0, 1], 'k--')
    plt.xlabel('Noeud')
    plt.title('Valeur de fonction de base en fonction de u')
    plt.show()
