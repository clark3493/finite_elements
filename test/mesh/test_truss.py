import sys
sys.path.insert(0, r'..\..\src')

import numpy as np
import unittest

from element import Node
from element.D1 import Rod
from mesh import Truss


class MyTestCase(unittest.TestCase):

    def setup_truss1(self):
        nodeA = Node(1, [0., 0.], ndof=2)
        nodeB = Node(2, [1., 0.], ndof=2)
        nodeC = Node(3, [1., 1.], ndof=2)
        nodeD = Node(4, [0., 1.], ndof=2)

        nodes = [nodeA, nodeB, nodeC, nodeD]

        E = 1.e11
        area = .02

        element1 = Rod(1, [nodeA, nodeD], E=E, area=area)
        element2 = Rod(2, [nodeD, nodeC], E=E, area=area)
        element3 = Rod(3, [nodeC, nodeB], E=E, area=area)
        element4 = Rod(4, [nodeD, nodeB], E=E, area=area)

        elements = [element1, element2, element3, element4]
        displacements = {1: [0., 0.],
                         2: [0., 0.],
                         3: [None, None],
                         4: [None, None]}

        loads = {1: [None, None],
                 2: [None, None],
                 3: [10., 0.],
                 4: [ 0., 0.]}

        return nodes, elements, displacements, loads

    def test_truss1_displacement_solution(self):
        nodes, elements, displacements, loads = self.setup_truss1()
        truss = Truss(nodes, elements, displacements, loads)

        expected = {1: 1.e-7 * np.array([0., 0.]),
                    2: 1.e-7 * np.array([0., 0.]),
                    3: 1.e-7 * np.array([0.2414213562373097, 0.]),
                    4: 1.e-7 * np.array([.19142135623730963, 0.05])}

        for key in expected:
            self.assertTrue(np.allclose(expected[key], truss.displacements[key]))

    def test_truss1_L_matrices(self):
        nodes, elements, displacements, loads = self.setup_truss1()
        truss = Truss(nodes, elements, displacements, loads, auto=False)

        L1_expected = np.zeros((4, 8))
        L1_expected[0, 0] = 1.
        L1_expected[1, 1] = 1.
        L1_expected[2, 6] = 1.
        L1_expected[3, 7] = 1.

        L2_expected = np.zeros((4, 8))
        L2_expected[0, 6] = 1.
        L2_expected[1, 7] = 1.
        L2_expected[2, 4] = 1.
        L2_expected[3, 5] = 1.

        L3_expected = np.zeros((4, 8))
        L3_expected[0, 4] = 1.
        L3_expected[1, 5] = 1.
        L3_expected[2, 2] = 1.
        L3_expected[3, 3] = 1.

        L4_expected = np.zeros((4, 8))
        L4_expected[0, 6] = 1.
        L4_expected[1, 7] = 1.
        L4_expected[2, 2] = 1.
        L4_expected[3, 3] = 1.

        self.assertTrue(np.allclose(truss.L[1], L1_expected))
        self.assertTrue(np.allclose(truss.L[2], L2_expected))
        self.assertTrue(np.allclose(truss.L[3], L3_expected))
        self.assertTrue(np.allclose(truss.L[4], L4_expected))


if __name__ == '__main__':
    unittest.main()
