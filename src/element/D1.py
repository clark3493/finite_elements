import os
import sys

import numpy as np

from math import ceil
from numpy import transpose
from sympy import diff, lambdify
from sympy.abc import x

SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from post.element.post_D1 import PostRod
from utils import collinear, distance
from ._gauss import gauss_points, weighted_integration
from ._node import Node


class AbstractElement1D(object):
    """
    An abstract class to hold method which are common to multiple types of 1D elements.

    The user has the option to specify a number of nodes to be linearly
    spaced between the ends of the element in addition to the provided global
    nodes.

    Parameters
    ----------
    eid: int
        Element ID.
    nodes: list(Node)
        Global nodes which make up the element.
    n_local_nodes: int, optional
        Number of local nodes to add in additional to global nodes. Default=0.

    Notes
    -----
    .. [1] The ends of the element MUST be the first and last nodes in the provided
           list of nodes.
    """
    # TODO: IMPLEMENT PROPERTY CACHEING TO DECREASE COMPUTATION TIME
    def __init__(self, eid, nodes, n_local_nodes=0):
        self.eid = eid
        self._global_nodes = nodes

        self.length = distance(nodes[0].x, nodes[-1].x)

        self._element_nodes = []
        self.set_local_nodes(n_local_nodes)

    @property
    def B(self):
        """Abstract placeholder for subclass implementation of shape function derivatives."""
        raise NotImplementedError("This property must be implemented by a subclass")

    @property
    def local_nodes(self):
        """
        Nodes which were created locally upon element instantiation.

        Returns
        -------
        local_nodes: list(Node)
        """
        return self._element_nodes

    @property
    def gauss_points(self):
        """
        Element gauss points and weights.

        Returns
        -------
        z: list(float)
            Gauss points ranging from -1 to 1.
        w: list(float)
            Weights corresponding to each Gauss point.
        """
        return gauss_points(self.ngp, ndims=1)

    @property
    def global_nodes(self):
        """
        Nodes which were provided externally to define the element.

        Returns
        -------
        global_nodes: list(Node)

        Raises
        ------
        ValueError
            If global_nodes is set with nodes which are not collinear.
        """
        return self._global_nodes

    @global_nodes.setter
    def global_nodes(self, value):
        if not collinear([v.x for v in value]):
            raise ValueError("Nodes supplied to 1D element must be collinear")
        self._global_nodes = value

    @property
    def J(self):
        """Abstract placeholder for subclass implementation of local Jacobian"""
        raise NotImplementedError("This property must be overridden by a subclass")

    @property
    def K(self):
        """
        Element stiffness matrix in element coordinate system.

        Returns
        -------
        K: numpy.ndarry
        """
        R = self.R
        K_LOCAL = self.K_LOCAL
        return transpose(R).dot(K_LOCAL).dot(R)

    @property
    def K_LOCAL(self):
        """Abstract placeholder for subclass implementation of local stiffness matrix"""
        raise NotImplementedError("This property must be overridden by a subclass")

    @property
    def N(self):
        """
        Vector of shape functions for each node.

        Each shape function is a compiled symbolic function which can be
        called by passing in a float or integer.

        Returns
        -------
        N: list(function)
        """
        return np.array([lambdify(x, Ni) for Ni in self._N])

    @property
    def ndof(self):
        """
        Total number of elemental degrees of freedom.

        Returns
        -------
        ndof: int
        """
        return sum((node.ndof for node in self.nodes))

    @property
    def ngp(self):
        """
        The number of gauss points required to integrate the element exactly.

        ngp = (p + 1) / 2 = n / 2, where:
            p = degree of polynomial integration
            n = number of nodes in the element.

        Returns
        -------
        ngp: int
        """
        return ceil(len(self.nodes) / 2)

    @property
    def nodes(self):
        """
        All elemental nodes, including global and local nodes.

        Returns
        -------
        nodes: list(Node)

        Notes
        -----
        .. [1] Nodes are sorted in ascending order according to their distance
               from the first global node.
        """
        n = self.global_nodes + self.local_nodes
        x = [distance(node.x, self.global_nodes[0].x) / self.length for node in n]
        return [node for _, node in sorted(zip(x, n))]

    @property
    def R(self):
        """
        Local coordinate system to element coordinate system transformation matrix.

        Vectors in the local coordinate system can be converted to the element
        coordinate system by pre-multiplying by R.

        Returns
        -------
        R: numpy.ndarray
        """
        n = self.ndof // 2
        r = np.zeros((len(self.nodes), self.ndof))
        for i in range(len(self.nodes)):
            r[i, n*i:n*(i+1)] = [(x2 - x1) / self.length for x1, x2 in zip(self.nodes[0].x, self.nodes[-1].x)]
        return r

    @property
    def z(self):
        """
        Coordinates of each node mapped to the Gaussian integration range (-1 to 1).

        The position of each node is defined by a single value.

        Returns
        -------
        z: list(float)
        """
        return [2 * distance(node.x, self.nodes[0].x) / self.length - 1. for node in self.nodes]

    def set_local_nodes(self, n):
        """
        Adds additional n nodes to the element on top of the nodes provided externally.

        The nodes are linearly spaced between the first and last global node.

        Parameters
        ----------
        n: int
            Number of nodes to add.
        """
        if n > 0:
            x1 = self.global_nodes[0].x
            x2 = self.global_nodes[-1].x
            coords = [x1 + i*(x2 - x2)/(n+1) for i in np.linspace(1, n)]
            element_nodes = [Node(xi) for xi in coords]
            self._element_nodes = element_nodes

    def to_post(self, post_nodes):
        raise NotImplementedError("This method must be overridden by a sub-class")

    @property
    def _N(self):
        """
        Symbolic shape functions for each node.

        Returns
        -------
        _N: symbolic function of x
        """
        N = []
        for i, zi in enumerate(self.z):
            Ni = 1.
            for j, zj in enumerate(self.z):
                if i != j:
                    Ni = Ni * (x - zj) / (zi - zj)
            N.append(Ni)
        return N


class Rod(AbstractElement1D):
    """
    A 1-dimensional finite element which supports axial loading only.

    The nodes which define the Rod may be 1, 2 or 3 dimensional.

    Parameters
    ----------
    eid: int
        Element ID.
    nodes: list(Node)
        The two nodes which define the Rod.
    E: float
        Young's modulus.
    area: float
        Cross sectional area.
    """
    def __init__(self, eid, nodes, E, area):
        super().__init__(eid, nodes)
        self.E = E
        self.area = area

    @property
    def B(self):
        """
        Vector of shape function 1st derivatives.

        Each shape function derivative is a compiled symbolic function which
        can be called by passing in a float or integer.

        Returns
        -------
        B: list(function)
        """
        return np.array([lambdify(x, diff(Ni)) for Ni in self._N])

    @property
    def J(self):
        """
        Element Jacobian.

        For this element, the Jacobian is simply a scalar value.

        Returns
        -------
        J: float
        """
        return distance(self.nodes[0].x, self.nodes[-1].x) / 2.

    @property
    def k(self):
        """
        Element scalar stiffness value.

        Returns
        -------
        k: float
        """
        return self.area * self.E / self.length

    @property
    def K_LOCAL(self):
        """
        Element stiffness matrix in the local coordinate system.

        The local coordinate system is one dimensional, aligned along
        the element's axis.

        Returns
        -------
        K_LOCAL: numpy.ndarray (ndim=2)
        """
        z, w = self.gauss_points
        Bz = weighted_integration(self.B, z, w)
        I = np.outer(Bz, Bz)
        return self.J * self.E * self.area * I

    def to_post(self, post_nodes):
        return PostRod(self, post_nodes)
