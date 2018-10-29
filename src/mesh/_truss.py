import dill
import sys
import numpy as np

from numpy import transpose

sys.path.insert(0, '..')
from solve import KUFSolver


class Truss(object):
    """
    An object for structural truss problems with elements that support axial load only.

    The items corresponding to the displacement and load dictionary should be vectors of the
    same length as the number of degrees of freedom as the corresponding node. The dictionary
    keys are the element ID's. Nodes with unknown displacement or load in any or all
    directions should have a 'None' object in place of a defined float value.

    Parameters
    ----------
    nodes: list(Node)
        The nodes which define the structural degrees of freedom.
    elements: list(Rod)
        The elements which define the nodal connectivity and stiffness.
    displacements: dict{int: array_like}
        Enforced displacements for each node (if any).
    loads: dict{int: array_like}
        Applied loads for each node (if any).
    auto: bool, optional.
        Flag to automatically solve the system after initializing the Truss. Default=True.
    save: str, optional
        Name of output file to store results for post-processing.

    Attributes
    ----------
    f: array_like (shape=(N, 1))
        Global truss force vector.
    K: array_like (shape=(N, N))
        Global truss stiffness matrix.
    u: array_like (shape=(N, 1))
        Global truss displacement vector.

    Notes
    -----
    .. [1] While the elements of the truss only support load in dimension,
       nodal coordinates may be supplied in 1, 2 or 3 dimensions.
    .. [2] Nodes with 0 applied load should have defined 0. loads in the supplied
       load vector. 'None' for nodal load indicates a reaction force that needs
       to be solved for.
    """
    def __init__(self, nodes, elements, displacements, loads, auto=True, save=None):
        self.nodes = nodes
        self.elements = elements

        # convert supplied displacement and load vectors to numpy arrays
        self.displacements = {k: np.array(d) for k, d in displacements.items()}
        self.loads = {k: np.array(l) for k, l in loads.items()}

        self.save = save

        self.ndof = sum((node.ndof for node in nodes))
        """
        Total number of degrees of freedom for the truss system.
        
        dtype: int
        """

        self.L = self._define_transformation_matrices(nodes, elements)
        """
        The global to elemental transformation matrix for each element.
        
        The keys of the dictionary are the element ID's and the items are the corresponding
        transformation matrix. Each matrix is defined such that L.dot(u_global) results in
        the elemental displacement vector.
        
        dtype: dict{int: array_like}
        """

        self.auto = auto
        if auto:
            self.solve()

    @property
    def f(self):
        f = None
        for i, node in enumerate(self.nodes):
            load = self.loads[node.id]
            if i == 0:
                f = load.reshape((node.ndof, 1))
            else:
                f = np.vstack([f, load.reshape((node.ndof, 1))])
        return f

    @property
    def K(self):
        K = np.zeros((self.ndof, self.ndof))
        for element in self.elements:
            L = self.L[element.id]
            K += transpose(L).dot(element.K).dot(L)
        return K

    @property
    def u(self):
        u = None
        for i, node in enumerate(self.nodes):
            displacement = self.displacements[node.id]
            if i == 0:
                u = displacement.reshape((node.ndof, 1))
            else:
                u = np.vstack([u, displacement.reshape((node.ndof, 1))])
        return u

    def save_to_file(self):
        """Pickle the truss object for data persistence."""
        post_nodes = {node.id: node.to_post(self.displacements[node.id]) for node in self.nodes}
        post_elements = {element.id: element.to_post([post_nodes[node.id] for node in element.nodes]) \
                         for element in self.elements}

        post_items = {'nodes': post_nodes, 'elements': post_elements, 'loads': self.loads}
        with open(self.save, 'wb') as f:
            dill.dump(post_items, f)

    def solve(self):
        """
        Solve the truss system of equations and store the resulting displacement and load data.

        The Truss instance's 'displacement' and 'loads' attributes are updated in place.
        """
        K = self.K
        u = self.u
        f = self.f
        solver = KUFSolver(K, u, f)

        self.displacements = self._vector_to_node_dict(solver.u)
        self.loads = self._vector_to_node_dict(solver.f)

        if self.save is not None:
            self.save_to_file()

    def _define_transformation_matrices(self, nodes, elements):
        """
        Define the global to elemental transformation matrix for each truss element.

        Parameters
        ----------
        nodes: list(Node)
            All truss nodes.
        elements: list(Rod)
            All truss elements.

        Returns
        -------
        L: dict{int: ndarray}
            Dictionary of the transformation matrices for each element, with element ID's as keys.
        """
        # initialize the tranformation matrix dictionary
        L = {}
        # store the nodal ID's in their relative order for future reference
        node_ids = [node.id for node in nodes]

        for element in elements:
            # list for storing the index of each elemental node within the global node list
            dof_indices = []

            # iterate through each of the element's nodes
            for element_node in element.nodes:
                # get the global node index
                node_index = node_ids.index(element_node.id)
                # determine the number of degrees of freedom defined prior to the current node
                previous_dof = sum((node.ndof for node in nodes[:node_index]))
                # store indices corresponding to the nodal degrees of freedom
                dof_indices += [previous_dof + i for i in range(element_node.ndof)]
            # save the transformation matrix
            L[element.id] = self._define_transformation_matrix(element, dof_indices)
        return L

    def _define_transformation_matrix(self, element, indices):
        """
        Define the transformation matrix for an element based on given global DOF indices.

        The list of provided indicies should indicate the position of the elements nodal degrees
        of freedom within the global truss displacement vector.

        Parameters
        ----------
        element: Rod
            The element whose transformation matrix is to be calculated.
        indices: list(int)
            Relative position of the elemental degrees of freedom within the global displacement vector.

        Returns
        -------
        L: ndarray
            The elemental transformation amtrix.
        """
        L = np.zeros((element.ndof, self.ndof))
        for i, index in enumerate(indices):
            L[i, index] = 1
        return L

    def _vector_to_node_dict(self, vector):
        """
        Convert a global vector into a dictionary with EID keys and corresponding portions of the global vector.

        Parameters
        ----------
        vector: array_like
            The global data array.

        Returns
        -------
        dict{int: ndarray}
            Dictionary with EID keys and portions of the global vector corresponding to each element as values.
        """
        total_dof = 0
        vector_dict = {}
        v = vector.reshape((len(vector),))
        for node in self.nodes:
            vector_dict[node.id] = np.array(v[total_dof:total_dof+node.ndof], dtype=float)
            total_dof += node.ndof
        return vector_dict


