import numpy as np

from numpy import transpose
from .utils import distance


class Rod(object):
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
        Young's modulus
    area: float
        Cross sectional area

    Attributes
    ----------
    K: array_like (ndim=2)
        Element sitffness matrix in the element coordinate system.
    ndof: int
        Total elemental degrees of freedom (sum of nodal degrees of freedom).
    R: array_like (ndim=2)
        Element rotation matrix for rotation from local to element coordinate system.
    """

    K_LOCAL = np.array([[1., -1.], [-1., 1.]])
    """
    Unitary element stiffness matrix in the local coordinate system.
    
    The local coordinate system is one dimensional for each node and is aligned 
    along the element's axis.
    
    dtype: ndarray (shape=(2,2))
    """

    NNODE = 2
    """
    Number of nodes which define the element.
    
    dtype: int
    """

    def __init__(self, eid, nodes, E=None, area=None):
        self.id = eid
        self.nodes = nodes
        self.E = E
        self.area = area
        self.length = distance(nodes[0].x, nodes[1].x)
        self.k = E * area / self.length

    @property
    def K(self):
        """Element stiffness matrix in the element coordinate system."""
        R = self.R
        k = self.k
        K_LOCAL = self.K_LOCAL
        return transpose(R).dot(k * K_LOCAL).dot(R)

    @property
    def ndof(self):
        """Total number of elemental degrees of freedom."""
        return sum((node.ndof for node in self.nodes))

    @property
    def nodes(self):
        """The two nodes which define the location of the Rod."""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        if len(value) != 2:
            raise ValueError("Rod can only be defined by 2 nodes")
        elif not value[0].ndof == value[1].ndof:
            raise ValueError("Rod nodes must have the same number of degrees of freedom.")
        self._nodes = value

    @property
    def R(self):
        """Local coordinate system to element coordinate system transformation matrix."""
        ndof = self.ndof
        r = np.zeros((2, self.ndof))
        r[0, 0:ndof//2] = [(x2 - x1) / self.length for x1, x2 in zip(self.nodes[0].x, self.nodes[1].x)]
        r[1, ndof//2:ndof] = [(x2 - x1) / self.length for x1, x2 in zip(self.nodes[0].x, self.nodes[1].x)]
        return r
