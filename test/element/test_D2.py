import os
import sys

import numpy as np
import unittest

from sympy import Matrix
from sympy.abc import xi, eta

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from element import Node
from element.D2 import ShellPlanar


class Element2DTestCase(unittest.TestCase):

    def setup_shell_planar1(self):
        node1 = Node([0., 1.], nid=1, ndof=2)
        node2 = Node([0., 0.], nid=2, ndof=2)
        node3 = Node([2., 0.5], nid=3, ndof=2)
        node4 = Node([2., 1.], nid=4, ndof=2)

        nodes = (node1, node2, node3, node4)

        shell = ShellPlanar(1, nodes, E=3.e7, nu=0.3)

        return nodes, shell

    def test_shell_planar1_C(self):
        nodes, shell = self.setup_shell_planar1()
        expected = 3.3e7 * np.array([[1., 0.3, 0.],
                                     [0.3, 1., 0.],
                                     [0., 0.,  0.35]])
        actual = shell.C
        self.assertTrue(np.allclose(expected, actual, atol=5e4))

    def test_shell_planar1_gradN4Q_gaussian_sym(self):
        nodes, shell = self.setup_shell_planar1()
        expected = 0.25 * Matrix([[eta-1, 1-eta, 1+eta, -eta-1],
                                  [xi-1, -xi-1, 1+xi, 1-xi]])
        actual = shell._gradN4Q_gauss_coords
        self.assertTrue(expected.equals(actual))

    def test_shell_planar1_Jsym(self):
        nodes, shell = self.setup_shell_planar1()
        expected = Matrix([[0., 0.125*eta - 0.375],
                           [1., 0.125*xi  + 0.125]])
        actual = shell._J
        self.assertTrue(expected.equals(actual))


    def test_shell_planar1_K_LOCAL(self):
        nodes, shell = self.setup_shell_planar1()
        expected = 1.e7 * np.array([[ 1.49, -0.74, -0.66,  0.16, -0.98,  0.65,  0.15, -0.08],
                                    [-0.74,  2.75,  0.24, -2.46,  0.66, -1.68, -0.16,  1.39],
                                    [-0.66,  0.24,  1.08,  0.33,  0.15, -0.16, -0.56, -0.41],
                                    [ 0.16, -2.46,  0.33,  2.60, -0.08,  1.39, -0.41, -1.53],
                                    [-0.98,  0.66,  0.15, -0.08,  2.00, -0.82, -1.18,  0.25],
                                    [ 0.65, -1.68, -0.16,  1.39, -0.82,  3.82,  0.33, -3.53],
                                    [ 0.15, -0.16, -0.56, -0.41, -1.18,  0.33,  1.59,  0.25],
                                    [-0.08,  1.39, -0.41, -1.53,  0.25, -3.53,  0.25,  3.67]])
        actual = shell.K_LOCAL
        self.assertTrue(np.allclose(expected, actual, atol=1.e5))

    def test_shell_planar_Nsym(self):
        nodes, shell = self.setup_shell_planar1()
        N = [1./4. * (1.-xi)*(1.-eta),
             1./4. * (1.+xi)*(1.-eta),
             1./4. * (1.+xi)*(1.+eta),
             1./4. * (1.-xi)*(1.+eta)]
        expected = Matrix([[N[0], 0., N[1], 0., N[2], 0., N[3], 0.],
                           [0., N[0], 0., N[1], 0., N[2], 0., N[3]]])
        actual = shell._N
        self.assertTrue(expected.equals(actual))


if __name__ == '__main__':
    unittest.main()
