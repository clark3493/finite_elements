import numpy as np

from numpy import transpose
from scipy.linalg import lu_factor, lu_solve


class KUFSolver(object):
    """
    An object to solve standard f=Ku systems of equations for finite element problems.

    Elements of the displacement (u) and load (f) vectors which are initially unknown should be represented
    by a Python None object. The known vs unknown parameters do not need to be provided in any particular
    order. The solver internally organizes the data as required and then stores the solution in the order
    that was originally provided.

    Parameters
    ----------
    K: array_like, (m, n)
        Stiffness matrix.
    u: array_like, (n)
        Displacement vector.
    f: array_like, (m)
        Load vector.
    auto: bool, optional
        Flag to automatically solve the system of equations after defining the object. Default=True.

    Examples
    --------
    >>> u = [None, 0., None]
    >>> f = [None, 0., -10.]
    >>> K = [[-1., 1., 0.], [2., -1., -1.], [-1., 0., 1.]]
    >>> solver = KUFSolver(K, u, f)
    >>> solver.u
    array([-10.0, 0.0, -20.0], dtype=object)
    """

    def __init__(self, K, u, f, auto=True):
        # save shapes of vectors
        self._u_shape = np.array(u).shape
        self._f_shape = np.array(f).shape

        # store matrix/vectors and reshape input vectors into vertical vectors
        self.K = np.array(K)
        self.u = np.array(u).reshape((len(u), 1))
        self.f = np.array(f).reshape((len(u), 1))

        # internal variable to hold sorted matrices/vectors
        self._K = None
        self._u = None
        self._f = None

        # solve the problem if the auto flag is not turned off
        if auto:
            self.solve()

    def solve(self):
        """Solve the given system of equations."""
        # sort into distinct sections for defined and undefined displacement
        self._sort()

        self._solve()

        # unsort displacements and loads into their original vector positions
        self._unsort()

    @property
    def _f_f(self):
        """The portion of the force vector beyond the number of defined displacement elements."""
        return self._f[self._ndef_u:]

    @property
    def _K_EF(self):
        """The rows of K corresponding to defined displacements and columns corresponding to undefined disp."""
        return self._K[:self._ndef_u, self._ndef_u:]

    @property
    def _K_F(self):
        """The rows and columns of K corresponding to undefined displacements."""
        return self._K[self._ndef_u:, self._ndef_u:]

    @property
    def _u_e(self):
        """The elements of u corresponding to defined displacements."""
        return self._u[:self._ndef_u]

    def _calc_f(self):
        """Calculate the value of unknown loads with the displacement vector and corresponding stiffness matrix row."""
        for i in np.arange(len(self.f)):
            if self._f[i][0] is None:
                self._f[i][0] = self._K[i, :].dot(self._u)

    def _solve(self):
        """Solve the system of equations for unknown displacements and loads after sorting."""
        # extract sub matrices for readability
        f_f = self._f_f
        K_F = self._K_F
        K_EF = self._K_EF
        u_e = self._u_e

        # solve for the unknown displacements using LU decomposition
        u_f = lu_solve(lu_factor(K_F), f_f - transpose(K_EF).dot(u_e))
        self._u[self._ndef_u:] = u_f

        # calculate unknown loads
        self._calc_f()

    def _sort(self):
        """Sort the stiffness, displacement and load arrays such that all known displacements are stored together."""
        self._sort_u()
        self._sort_f()

    def _sort_f(self):
        f = self.f
        K = self._K.copy()

        # get the indices where the loads have been specified/unspecified
        f_defined = np.where(f != None)[0]
        f_undefined = np.where(f == None)[0]
        if len(f_undefined) != self._ndef_u:
            raise ValueError("All defined displacements must have corresponding defined loads and vice versa.")

        # move all undefined loads to beginning of vector
        self._f = np.vstack([f[f_undefined], f[f_defined]])

        for j, ind in enumerate(f_undefined):
            self._K[j, :] = K[ind, :]
        for j, ind in enumerate(f_defined):
            self._K[j + len(f_undefined), :] = K[ind, :]

        # save indices to resort data in order it was originally provided
        self._f_indices = np.hstack([f_undefined, f_defined])

    def _sort_u(self):
        u = self.u

        #  get _u_indices where displacements have been specified/unspecified
        u_defined = np.where(u != None)[0]
        u_undefined = np.where(u == None)[0]

        # move all defined displacements to beginning of vector
        self._u = np.vstack([u[u_defined], u[u_undefined]])

        self._K = np.zeros((len(self.f), len(u)))
        for i, ind in enumerate(u_defined):
            self._K[:, i] = self.K[:, ind]
        for i, ind in enumerate(u_undefined):
            self._K[:, i + len(u_defined)] = self.K[:, ind]

        # save _u_indices to resort data in order it was originally provided
        self._u_indices = np.hstack([u_defined, u_undefined])
        self._ndef_u = len(u_defined)

    def _unsort(self):
        """Reorganize the data into the order it was originally provided in"""
        u = self._u.copy()
        f = self._f.copy()

        for i, j in enumerate(self._u_indices):
            u[j] = self._u[i]
        for i, j in enumerate(self._f_indices):
            f[j] = self._f[i]

        self.u = u.reshape(self._u_shape)
        self.f = f.reshape(self._f_shape)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
