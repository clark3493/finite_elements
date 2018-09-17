import sys
sys.path.insert(0, '../../src')

import unittest

from element.utils import *


class UtilsTestCase(unittest.TestCase):

    def test_distance(self):
        a = [20., 30., 0.]
        b = [10., 20., 10.]
        d = distance(a, b)
        self.assertEqual(np.sqrt(3*10**2.), d)


if __name__ == '__main__':
    unittest.main()
