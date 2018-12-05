import os
import sys
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import unittest

from utils import *


class UtilsTestCase(unittest.TestCase):

    def test_ccw_returns_true(self):
        points = [(1., 0.), (1., 1.), (0., 1.), (-1., 1.), (-1., 0.), (-1., -1.), (0., -1.), (1., -1.)]
        self.assertTrue(ccw(points))

    def test_ccw_returns_false(self):
        points = [(1., 0.), (1., 1.), (0., 1.), (-1., 1.), (-1., -1.), (-1., 0.), (0., -1.), (1., -1.)]
        self.assertFalse(ccw(points))

    def test_distance(self):
        a = [20., 30., 0.]
        b = [10., 20., 10.]
        d = distance(a, b)
        self.assertEqual(np.sqrt(3*10**2.), d)


if __name__ == '__main__':
    unittest.main()
