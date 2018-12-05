import os
import sys

import numpy as np

from numpy import transpose

SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from solve import KUFSolver


class ShellPlanarModel(object):

    def __init__(self, elements, displacements, loads, auto=True):
        self.elements = elements

        # convert supplied displacement and load vectors to numpy arrays
        self.displacements = {k: np.array(d) for k, d in displacements.items()}
        self.loads = {k: np.array(l) for k, l in loads.items()}

        for node in self.nodes:
            if node.id not in self.displacements:
                self.displacements[node.id] = np.array([None] * node.ndof)
            if node.id not in self.loads:
                self.loads[node.id] = np.array([0.] * node.ndof)

        self.ndof = sum((node.ndof for node in self.nodes))

        self.L = self._define_transformation_matrices()

        self.auto = auto
        if self.auto:
            self.solve()

    @property
    def f(self):
        f = None
        for i, node in enumerate(self.nodes):
            load = self.loads[node.id]
            if i == 0:
                f = load.reshape((node.ndof, 1))
            else:
                f = np.vstack((f, load.reshape((node.ndof, 1))))
        return f

    @property
    def K(self):
        K = np.zeros((self.ndof, self.ndof))
        for element in self.elements:
            L = self.L[element.eid]
            K += transpose(L).dot(element.K).dot(L)
        return K

    @property
    def nodes(self):
        nodes = []
        for element in self.elements:
            nodes += element.nodes
        return sorted(list(set(nodes)), key=lambda x: x.id)

    @property
    def u(self):
        u = None
        for i, node in enumerate(self.nodes):
            displacement = self.displacements[node.id]
            if i == 0:
                u = displacement.reshape((node.ndof, 1))
            else:
                u = np.vstack((u, displacement.reshape((node.ndof, 1))))
        return u

    def solve(self):
        K = self.K
        u = self.u
        f = self.f
        solver = KUFSolver(K, u, f)

        self.displacements = self._vector_to_node_dict(solver.u)
        self.loads = self._vector_to_node_dict(solver.f)

    def _define_transformation_matrices(self):
        nodes = self.nodes
        elements = self.elements

        L = {}
        node_ids = [node.id for node in nodes]

        for element in elements:
            dof_indices = []

            for element_node in element.nodes:
                node_index = node_ids.index(element_node.id)
                previous_dof = sum((node.ndof for node in nodes[:node_index]))
                dof_indices += [previous_dof + i for i in range(element_node.ndof)]

            L[element.eid] = self._define_transformation_matrix(element, dof_indices)
        return L

    def _define_transformation_matrix(self, element, indices):
        L = np.zeros((element.ndof, self.ndof))
        for i, index in enumerate(indices):
            L[i, index] = 1
        return L

    def _vector_to_node_dict(self, vector):
        total_dof = 0
        vector_dict = {}
        v = vector.reshape((len(vector),))
        for node in self.nodes:
            vector_dict[node.id] = np.array(v[total_dof:total_dof+node.ndof], dtype=float)
            total_dof += node.ndof
        return vector_dict
