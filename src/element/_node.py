import numpy as np


class Node(object):
    """
    A nodal element for use in finite element analyses.

    If fewer coordinates are supplied than the number of nodal degrees
    of freedom, then 0's are filled in from the last supplied
    coordinate.

    Parameters
    ----------
    nid: int
        Node ID
    x: array_like (ndim=1)
        Node coordinates
    ndof: int, optional
        Number of nodal degrees of freedom. Default=3.
    """
    def __init__(self, nid, x, ndof=3):
        self.id = nid
        self._ndof = ndof

        self.x = x

    @property
    def ndof(self):
        """Number of nodal degrees of freedom."""
        return self._ndof

    @ndof.setter
    def ndof(self, value):
        self._ndof = value
        # adjust nodal degrees of freedom if necessary
        self.x = np.trim_zeros(self.x, trim='b')

    @property
    def x(self):
        """Nodal coordinates."""
        return self._x

    @x.setter
    def x(self, value):
        if len(value) > self.ndof:
            raise ValueError("%d nodal coordinates provided but node is only %d dimensional" % (len(value), self.ndof))
        else:
            self._x = np.array(value + [0.] * (self.ndof-len(value)))

