import os
import sys
SRC_DIR = os.path.abspath(r"..\..\src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import unittest

from element._gauss import gauss_points, _M, _P


class GaussPointTestCase(unittest.TestCase):

    def test_gauss_points_1d_z1(self):
        z, W = gauss_points(1, ndims=1)
        self.assertTrue(np.allclose([0.], z))

    def test_gauss_points_1d_W1(self):
        z, W = gauss_points(1, ndims=1)
        self.assertEqual([2.], W)

    def test_gauss_points_1d_z2(self):
        z, W = gauss_points(2, ndims=1)
        self.assertTrue(np.allclose([-1/np.sqrt(3), 1/np.sqrt(3)], z))

    def test_gauss_points_1d_W2(self):
        z, W = gauss_points(2, ndims=1)
        self.assertTrue(np.allclose([1., 1.], W))

    def test_M(self):
        M = _M(2)
        M_values = np.zeros((2, 4))
        for j in range(len(M[:, 0])):
            for i in range(len(M[0,:])):
                M_values[j, i] = M[j, i](2)
        expected = np.array([[1., 2., 4., 8.], [1., 2., 4., 8.]])
        self.assertTrue(np.allclose(expected, M_values))

    def test_P(self):
        self.assertTrue(np.allclose([2., 0., 2./3., 0.], _P(4)))


if __name__ == '__main__':
    unittest.main()
