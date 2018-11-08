import numpy as np


def collinear(points):
    """
    Determine if the given points are all collinear.

    Collinearity is determined by verifying that the cross product
    of xi and X is zero for all xi, where:
        - xi is the i'th coordinate minus the first coordinate
        - X is the last coordinate minus the first coordinate

    Point coordinates may be of arbitrary dimension but the number
    of dimensions for each point must be the same.

    Parameters
    ----------
    points: array_like (ndim=1)
        List of coordinates to check.

    Returns
    -------
    are_collinear: bool
        Boolean indicating if the points are all collinear.
    """
    cross = [np.cross(point-points[0], points[-1]-points[0]) for point in points]
    return np.all(c == 0. for c in cross)


def distance(a, b):
    """
    Evaluate the Einstein summation on the given 1D vectors (Euclidian norm).

    Parameters
    ----------
    a: array_like (ndim=1)
        The first vector
    b: array_like (ndim=1)
        The second vector

    Returns
    -------
    length: float
        Euclidian norm of the given vectors
    """
    a_min_b = np.array(a) - np.array(b)
    return np.sqrt(np.einsum('i,i', a_min_b, a_min_b))


