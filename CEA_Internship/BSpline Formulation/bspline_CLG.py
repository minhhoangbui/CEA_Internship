'''
Created on 28 mars 2017
A Spline class based on the algorithm of "The NURBS Book" by Les Piegl and
Wayne Tiller
@author: MB251995
'''
# coding: utf-8
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


class Spline():
    '''
    A Spline class which allows us to describe a spline curve as well as
    computing its derivatives and contruct a interpolation in form of
    spline. Approximation is still in progress

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

    '''

    def __init__(self, knots=None, deg=None, ctrl_pts=None):
        '''
        Initializing the parameters of the spine curve
        '''
        self.knots = knots
        self.degree = deg
        self.control_points = ctrl_pts

    def _find_span(self, u):
        '''
        Find the span in the knot vector to which u corresponds

        Parameters
        ----------
        u : float
            A value in the inteval [0, 1]

        Returns
        -------
        mid : integer
            u in the interval [ U[mid]; U[mid + 1]

        References:
        The NURBS Book - Les Piegl, Wayne Tiller p.68
        '''
        m = len(self.knots) - 1
        if u == self.knots[m]:
            return m - self.degree - 1
        low = self.degree
        high = m
        mid = (low + high) / 2
        while (u < self.knots[mid] or u >= self.knots[mid + 1]):
            if (u < self.knots[mid]):
                high = mid
            else:
                low = mid
            mid = (low + high) / 2
        return mid

    def _compute_Bfunc(self, u):
        '''
        Compute the basis function corresponding to u

        Parameters
        ----------
        u: float
            A value in the interval [0, 1] that we want to compute the basis
            function corresponding to

        Returns
        -------
        N :  numpy.ndarray
            An array which includes all the basis coefficients of spline which
            correspond to the the degree given

        References:
        The NURBS Book - Les Piegl, Wayne Tiller p.70
        '''
        span = self._find_span(u)
        N = np.zeros(self.degree + 1)
        N[0] = 1
        left = np.zeros(self.degree + 1)
        right = np.zeros(self.degree + 1)

        for j in range(1, self.degree + 1):
            left[j] = u - self.knots[span + 1 - j]
            right[j] = self.knots[span + j] - u
            saved = 0
            for k in range(j):
                temp = N[k] / (right[k + 1] + left[j - k])
                N[k] = saved + right[k + 1] * temp
                saved = left[j - k] * temp
            N[j] = saved
        return N

    def compute_point(self, u):
        '''
        Compute the coordinates of the point that u corresponds to

        Parameters
        ----------
        u : float
            A value in the interval [0, 1] that we want to compute the
            coordinates of the corresponding point

        Returns
        -------
        s : numpy.ndarray
            The coordinates of the correspoding point


        '''
        span = self._find_span(u)
        N = self._compute_Bfunc(u)
        s = 0
        for i in range(self.degree + 1):
            s += N[i] * self.control_points[span - self.degree + i]
        return s

    def compute_derivative_basis(self, order, u):
        '''
        Compute the derivatives of basis function of the degree 'order'
        corresponding to u

        Parameters
        ----------
        u : float
            A value in the interval [0, 1] that we want to find the derivative
        order: integer
            The order of derivative that we want to find
        Returns
        -------
        ders : numpy.ndarray
            The array that includes all the derivative of basis function

        References:
        The NURBS Book - Les Piegl, Wayne Tiller p.72
        '''
        p = self.degree
        i = self._find_span(u)
        ders = np.zeros((order + 1, p + 1))
        tmp1 = np.zeros((p + 1, p + 1))
        tmp2 = np.zeros((2, p + 1))
        tmp1[0, 0] = 1
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        for j in range(1, p + 1):
            left[j] = u - self.knots[i + 1 - j]
            right[j] = self.knots[i + j] - u
            saved = 0.0
            for r in range(j):
                tmp1[j, r] = right[r + 1] + left[j - r]
                tmp = tmp1[r, j - 1] / tmp1[j, r]
                tmp1[r, j] = saved + right[r + 1] * tmp
                saved = left[j - r] * tmp
            tmp1[j, j] = saved

        for j in range(p + 1):
            ders[0, j] = tmp1[j, p]

        for r in range(0, p + 1):
            s1 = 0
            s2 = 1
            tmp2[0, 0] = 1.0
            for k in range(1, order + 1):
                d = 0.0
                rk = r - k
                pk = p - k
                if r >= k:
                    tmp2[s2, 0] = tmp2[s1, 0] / tmp1[pk + 1, rk]
                    d = tmp2[s2, 0] * tmp1[rk, pk]

                if rk >= -1:
                    j1 = 1
                else:
                    j1 = -rk

                if r - 1 <= pk:
                    j2 = k - 1
                else:
                    j2 = p - r

                for j in range(j1, j2 + 1):
                    tmp2[s2, j] = (tmp2[s1, j] - tmp2[s1, j - 1]) / \
                                        tmp1[pk + 1, rk + j]
                    d += tmp2[s2, j] * tmp1[rk + j, pk]

                if r <= pk:
                    tmp2[s2, k] = -tmp2[s1, k - 1] / tmp1[pk + 1, r]
                    d += tmp2[s2, k] * tmp1[r, pk]
                ders[k, r] = d
                j = s1
                s1 = s2
                s2 = j
        r = p
        for k in range(1, order + 1):
            for j in range(p + 1):
                ders[k, j] *= r
            r *= (p - k)
        return ders

    def compute_derivative(self, order, u):
        span = self._find_span(u)
        der_N = self.compute_derivative_basis(order, u)[order]
        s = 0
        for i in range(self.degree + 1):
            s += der_N[i] * self.control_points[span - self.degree + i]
        return s

    def get_length(self, u):
        '''
        Compute the length of the curve corresponding to [0, u]

        Parameters
        ----------
        u : float
            A value in the interval [0, 1]

        Returns
        -------
        length : float
            The length of the curve corresponding to [0, u]
        '''
        uk = np.linspace(0, u, 100)
        ck = np.zeros((100, 2))
        for i, item in enumerate(uk):
            ck[i, :] = self.compute_point(item)
        length = 0
        for i in range(99):
            length += sqrt((ck[i, 0] - ck[i + 1, 0]) ** 2 +
                           (ck[i, 1] - ck[i + 1, 1]) ** 2)
        return length

    def draw_curve(self):
        '''
        Draw the spline curve given

        '''
        inp = np.linspace(0, 1, 100)
        out = np.zeros((len(inp), 2))

        for i, item in enumerate(inp):
            out[i] = self.compute_point(item)
        out = out.T
        length = 0
        for i in range(1, len(out)):
            length += sqrt((out[i, 0] - out[i - 1, 0]) ** 2 +
                           (out[i, 1] - out[i - 1, 1]) ** 2)
        print " The length of this curve is {:f}".format(length)

        for i, pts in enumerate(self.control_points):
            plt.plot(pts[0], pts[1], 'go')
        plt.plot(out[0], out[1], 'r-')
        plt.show()

    def interpolate(self, dt_pts, per=0):
        '''
        Compute the interpolating curve given the data point

        Parameters
        ----------
        dt_pts : numpy.ndarray
            The coordinates of the data points that we want to interpolate
        per: boolean
            It indicates that we want to interpolate a periodic curve or not.
            The periodic part in still in progress

        Returns
        -------
        We will modify the parameters of spline curve itself

        References:
        The NURBS Book - Les Piegl, Wayne Tiller p.369
        '''
        if per == 0:
            m = len(dt_pts) + self.degree

            # generate uk
            d = 0
            n = len(dt_pts) - 1
            for i in range(1, n + 1):
                d += sqrt((dt_pts[i, 0] - dt_pts[i - 1, 0]) ** 2 +
                          (dt_pts[i, 1] - dt_pts[i - 1, 1]) ** 2)
            uk = np.zeros(len(dt_pts))
            uk[n] = 1
            for i in range(1, n):
                uk[i] = uk[i - 1] + sqrt(
                                    (dt_pts[i, 0] - dt_pts[i - 1, 0]) ** 2 +
                                    (dt_pts[i, 1] - dt_pts[i - 1, 1]) ** 2) / d

            # generate knots vector
            self.knots = np.zeros(m + 1)
            for i in range(m - self.degree, m + 1):
                self.knots[i] = 1
            for j in range(1, n - self.degree + 1):
                v = 0
                for i in range(j, j + self.degree):
                    v += uk[i] / self.degree
                self.knots[j + self.degree] = v

            # generate matrix
            mat = np.zeros((len(dt_pts), len(dt_pts)))
            for k in range(n + 1):
                span = self._find_span(uk[k])
                mat[k, span - self.degree: span + 1] = \
                    self._compute_Bfunc(uk[k])

            x = solve(mat, dt_pts.T[0])
            y = solve(mat, dt_pts.T[1])
            self.control_points = np.array([x, y]).T
        else:
            pass

    def approximate(self, dt_pts, num_ctrl):
        pass




