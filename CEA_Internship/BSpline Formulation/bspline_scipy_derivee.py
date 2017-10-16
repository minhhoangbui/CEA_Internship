'''
Created on 16 mai 2017

@author: MB251995

Derive a subclass of BSpline that is optimised for the convergence of
Active Contour in case of periodic splines
'''
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


class BSpline_periodic(BSpline):
    '''
    A Subclass of BSpline from scipy which allow to control the periodicity of
    the spline

    Parameters
    ----------
    knots : numpy.array(m + 1)
        knots = [u0, u1, ..., um]
        Knots vector of the BSpline: size (m + 1)
    ctrl_pts: numpy.narray(n + 1)
        ctrl_pts = [[p0x, p0y], ..., [pnx, pny]]
        A set of control points of the BSpline: size (n + 1)
        Warning following sizing constraint shall be respected:
        m = n + degree + 1
    degree: integer
        Degree of the spline
    is_periodic: boolean
        Indication of the periodicity of spline

    '''

    def __init__(self, knots, ctrl_pts, degree, is_periodic=False):
        BSpline.__init__(self, knots, ctrl_pts, degree)
        # self.t corresponds to the knots vector
        # self.c corresponds to the control points
        # self.k corresponds to the degree of the curve
        self.is_periodic = is_periodic
        if is_periodic:
            # Check control points
            # Suppose that we have (n + 1) control points {P[0], ...,P[n]}
            # Condition for proper spline periodicity:
            # P[i - 1] == P[n + i - k] with i  = 1...k
            if (self.c[(len(self.c) - self.k):] == self.c[:self.k]).all():
                # It means that if the vector {P[n + 1 - k],..., P[n]} ==
                # {P[0],..., P[k - 1]} then:
                pass
            else:
                raise ValueError("Not satisfied")

            # Check knots vector
            # Suppose that we have (m + 1) knots {u[0], ..., u[m]} and degree k
            # Condition of proper spline periodiciy:
            # u[k - i] == u[m - k - i] - 1 and
            # u[m - k + i] == u[k + i] + 1 for i = 1,...,k
            m = len(self.t) - 1
            for i in range(1, self.k + 1):
                if (self.t[self.k - i] != self.t[m - self.k - i] - 1 or
                        self.t[m - self.k + i] != self.t[self.k + i] + 1):
                    raise ValueError("Not satisfied")

    def get_ctrl_pts(self, only_free=False):
        '''
        Get the coordinates of the control points as the input for the
        optimisation.
        In case of the periodic spline curve and only_free is True, it will
        return only the independant control points. In the other case, it
        will return all the control points


        Parameters
        ----------
        only_free : boolean, optional
            only_free = True: Return only the independant control points in
            case of periodic spline
            If not, return all the points

        Returns
        -------
        self.c : numpy.narray
            The returned control points

        References
        ----------
        Curve and Surface Fitting with Splines - Paul Dierckx, p.11


        '''
        if only_free:
            if self.is_periodic:
                n = len(self.c) - 1
                # remove duplicated final control points:
                # {P[n + 1 - k],..., P[n]}
                r_idx = np.linspace(n - self.k + 1, n, self.k)
                return np.delete(self.c, r_idx, axis=0)
            else:
                return self.c
        else:
            return self.c

    def set_ctrl_pts(self, new_ctrl, only_free=False):
        '''
        Set the new control points for the spline

        Parameters
        ----------
        new_ctrl: numpy.narray()
            New control points about to be set. The dimension must be agreed
        only_free: boolean, optional
            only_free = True: Replace only the independant control points in
            case of periocity
            only_free = False: Replace all the control points

        References
        ----------
        Curve and Surface Fitting with Splines - Paul Dierckx, p.11

        '''
        if not self.is_periodic:
            # If the spline is not periodic, the dimension of the new control
            # point must agree to the old one
            if len(new_ctrl) == len(self.c):
                self.c = new_ctrl
            else:
                raise ValueError("The dimension must agree")
        else:
            # If the curve is periodic,
            if not only_free:
                # In case we want to update all the control point (not
                # necessary), we have to check the conditions of dimension and
                # periodcity
                if len(new_ctrl) == len(self.c):
                    pass
                else:
                    raise ValueError("The dimension must agree")
                if (new_ctrl[len(new_ctrl) - self.k:] ==
                        new_ctrl[:self.k]).all():
                    pass
                else:
                    raise ValueError("Not satisfied")
                self.c = new_ctrl
            else:
                # Check the dimension
                if len(new_ctrl) == len(self.c) - self.k:
                    self.c = np.empty_like(self.c)
                    self.c[:len(new_ctrl)] = new_ctrl[:]
                    self.c[len(self.c) - self.k:] = self.c[: self.k]
                else:
                    raise ValueError("The dimension must agree")

    def plot_spline(self, label, n=1000):
        '''
        Plot a well-defined spline

        Parameters
        ----------
        n : integer, optional
            Number of discrete points in u we compute in order to plot the
            digital curve

        '''
        x = np.zeros(n)
        y = np.zeros(n)
        [x[:], y[:]] = np.transpose(self(np.linspace(0, 1, n)))
        plt.plot(x, y, label=label)
        plt.xlabel("x")
        plt.ylabel("y")

    def get_length(self, nbr=1000):
        [x1u_t, y1u_t] = np.transpose(self.derivative(1)
                                      (np.linspace(0.0, 1.0, nbr + 1)))
        return np.trapz(y=np.hypot(x1u_t, y1u_t), dx=1.0 / nbr)
