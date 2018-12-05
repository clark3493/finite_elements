import os
import sys

import numpy as np

from sympy import diff, lambdify, Matrix
from sympy.abc import eta, xi

SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import ccw
from ._gauss import gauss_points, weighted_integration


class ShellPlanar(object):

    ngp = 2

    def __init__(self, eid, nodes, E, nu):
        self.eid = eid

        self._nodes = None
        self.nodes = nodes

        self.E = E
        self.nu = nu

    @property
    def B(self):
        return np.array([lambdify((xi, eta), Bi) for Bi in self._B]).reshape(self._B.shape)

    @property
    def C(self):
        E = self.E
        nu = self.nu
        return E / (1. - nu**2) * np.array([[1., nu, 0.],
                                            [nu, 1., 0.],
                                            [0., 0., (1.-nu)/2.]])

    @property
    def gauss_points(self):
        return gauss_points(self.ngp)

    @property
    def J(self):
        return np.array([lambdify((xi, eta), Ji) for Ji in self._J]).reshape(self._J.shape)

    @property
    def k_local_integrand(self):
        integrand = np.array([lambdify((xi, eta), ki) for ki in self._k_local_integrand])
        return integrand.reshape(self._k_local_integrand.shape)

    @property
    def K(self):
        z, w = self.gauss_points
        return weighted_integration(self.k_local_integrand, z, w, ndims=2)

    @property
    def N(self):
        return np.array([lambdify((xi, eta), Ni) for Ni in self._N]).reshape(self._N.shape)

    @property
    def ndof(self):
        return sum((node.ndof for node in self.nodes))

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        if not ccw([node.x for node in value]):
            raise ValueError("ShellPlanar nodes must be provided in counter-clockwise order")
        self._nodes = value

    @property
    def x(self):
        return np.array([self.nodes[0].x,
                         self.nodes[1].x,
                         self.nodes[2].x,
                         self.nodes[3].x])

    @property
    def _B(self):
        gN4Q = self._gradN4Q_element_coords
        B = np.zeros((3, 8), dtype=object)
        for i in range(4):
            B[0, 2*i] = gN4Q[0, i]
            B[1, 2*i+1] = gN4Q[1, i]
            B[2, 2*i] = gN4Q[1, i]
            B[2, 2*i+1] = gN4Q[0, i]
        return Matrix(B)

    @property
    def _gradN4Q_element_coords(self):
        return self._J.inv() * self._gradN4Q_gauss_coords

    @property
    def _gradN4Q_gauss_coords(self):
        row1 = [diff(Ni, xi) for Ni in self._N4Q]
        row2 = [diff(Ni, eta) for Ni in self._N4Q]
        return Matrix([row1, row2])

    @property
    def _J(self):
        return Matrix(self._gradN4Q_gauss_coords * self.x)

    @property
    def _k_local_integrand(self):
        return self._B.transpose() * self.C * self._B * self._J.det()

    @property
    def _N(self):
        N = np.zeros((2, 8), dtype=object)
        for i in range(4):
            N[0, 2*i] = self._N4Q[i]
            N[1, 2*i+1] = self._N4Q[i]

        return Matrix(N)

    @property
    def _N4Q(self):
        return Matrix([1./4. * (1.-xi)*(1.-eta),
                       1./4. * (1.+xi)*(1.-eta),
                       1./4. * (1.+xi)*(1.+eta),
                       1./4. * (1.-xi)*(1.+eta)])
