import numpy as np

from math import asin, atan2, cos, sin


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


