import os
import sys
SRC_DIR = os.path.abspath(r"..\..\src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import unittest

from element import Node
from element.D1 import Rod


class OneDimensionalElementTestCase(unittest.TestCase):

    def setup_rod1(self):
        node1 = Node([1., 0., 0.], nid=1)
        node2 = Node([0., 1., 0.], nid=2)
        rod = Rod(1, [node1, node2], E=1.*10e11, area=0.02)
        return rod, [node1, node2]

    def setup_rod2(self):
        node1 = Node([1., 0.], nid=1, ndof=2)
        node2 = Node([0., 1.], nid=2, ndof=2)
        rod = Rod(1, [node1, node2], E=1.*10e11, area=0.02)
        return rod, [node1, node2]

    def test_rod1_gauss_point_values(self):
        rod, nodes = self.setup_rod1()
        z, w = rod.gauss_points
        self.assertTrue(np.allclose(0., z))

    def test_rod1_gauss_point_weights(self):
        rod, nodes = self.setup_rod1()
        z, w = rod.gauss_points
        self.assertEqual(2., w)

    def test_rod1_length(self):
        rod, nodes = self.setup_rod1()
        self.assertEqual(np.sqrt(2.), rod.length)

    def test_rod1_rotation_matrix(self):
        rod, nodes = self.setup_rod1()
        expected = 1. / np.sqrt(2.) * np.array([[-1.,  1.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0., -1.,  1.,  0.]])
        self.assertTrue(np.allclose(expected, rod.R))

    def test_rod1_stiffness_matrix(self):
        rod, nodes = self.setup_rod1()
        expected = rod.k / 2. * np.array([[ 1., -1.,  0., -1.,  1.,  0.],
                                          [-1.,  1.,  0.,  1., -1.,  0.],
                                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                                          [-1.,  1.,  0.,  1., -1.,  0.],
                                          [ 1., -1.,  0., -1.,  1.,  0.],
                                          [ 0.,  0.,  0.,  0.,  0.,  0.]])
        self.assertTrue(np.allclose(expected, rod.K))

    def test_rod2_length(self):
        rod, nodes = self.setup_rod2()
        self.assertEqual(np.sqrt(2.), rod.length)

    def test_rod2_rotation_matrix(self):
        rod, nodes = self.setup_rod2()
        expected = 1. / np.sqrt(2.) * np.array([[-1.,  1.,  0.,  0.],
                                                [ 0.,  0., -1.,  1.]])
        self.assertTrue(np.allclose(expected, rod.R))

    def test_rod2_stiffness_matrix(self):
        rod, nodes = self.setup_rod2()
        expected = rod.k / 2. * np.array([[ 1., -1., -1.,  1.],
                                          [-1.,  1.,  1., -1.],
                                          [-1.,  1.,  1., -1.],
                                          [ 1., -1., -1.,  1.]])
        self.assertTrue(np.allclose(expected, rod.K))


if __name__ == '__main__':
    unittest.main()
