import os
import sys
SRC_DIR = os.path.abspath(r"..\..\src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import unittest

from solve import KUFSolver


class KUFSolverTestCase(unittest.TestCase):

    def compare(self, a, b):
        self.assertTrue(np.all(np.equal(a, b)))

    @staticmethod
    def kuf_set1():
        u = [None, 1., None]
        f = [2, None, 3]
        K = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        return K, u, f

    @staticmethod
    def kuf_set2():
        u = [None, 0., None]
        f = [None, 0., -10.]
        K = [[-1., 1., 0.], [2., -1., -1.], [-1., 0., 1.]]
        return K, u, f

    def test_incorrect_uf_definition_raises_value_error(self):
        u = [None, 4, None]
        f = [6, None, None]
        K = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertRaises(ValueError, KUFSolver, K, u, f)

    def test_solve_f2(self):
        K, u, f = self.kuf_set2()

        solver = KUFSolver(K, u, f)

        expected = np.array([10., 0., -10])
        self.compare(expected, solver.f)

    def test_solve_u2(self):
        K, u, f = self.kuf_set2()

        solver = KUFSolver(K, u, f)

        expected = np.array([-10., 0, -20.])
        self.compare(expected, solver.u)

    def test_sort_f1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        expected = np.array([[None], [2], [3]])
        self.compare(expected, solver._f)

    def test_sort_indices1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        expected = [1, 0, 2]
        self.compare(expected, solver._u_indices)

    def test_sort_K1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        expected = np.array([[5, 4, 6], [2, 1, 3], [8, 7, 9]])
        self.compare(expected, solver._K)

    def test_sort_u1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        expected = np.array([[1.], [None], [None]])
        self.compare(expected, solver._u)

    def test_unsort_f1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        solver._f[0] = 99
        solver._unsort()

        expected = np.array([2, 99, 3])
        self.compare(expected, solver.f)

    def test_unsort_u1(self):
        K, u, f = self.kuf_set1()

        solver = KUFSolver(K, u, f, auto=False)
        solver._sort()

        solver._u[1] = 10
        solver._u[2] = 99
        solver._unsort()

        expected = np.array([10, 1, 99])
        self.compare(expected, solver.u)


if __name__ == '__main__':
    unittest.main()
