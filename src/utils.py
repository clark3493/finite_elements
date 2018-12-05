import numpy as np

from math import atan2, pi


def ccw(points, allow_duplicates=False):
    """
    Determine if the given set of 2D points are provided in counter-clockwise order.
    Parameters
    ----------
    points: list(float), len=2
        The list of 2D coordinates.
    allow_duplicates: bool, optional
        Flag to interpret duplicate points as counter-clockwise. Default=False.

    Returns
    -------
    ccw: bool
        Flag indicating if all points are in counter-clockwise order.

    Notes
    -----
    .. [1] This function may return erroneous results if the given set of points is not convex.
    """
    # determine the centroid of all the provided points
    coords = [[pt[0] for pt in points], [pt[1] for pt in points]]
    centroid = [sum(xlist) / len(xlist) for xlist in coords]

    points_continuous = points + [points[0]]

    ccw = True
    for i in range(len(points)):
        pt1 = points_continuous[i]
        pt2 = points_continuous[i+1]

        if i == 0:
            angle0 = atan2(pt1[1] - centroid[1], pt1[0] - centroid[0])

        angle1 = atan2(pt1[1] - centroid[1], pt1[0] - centroid[0]) - angle0
        angle2 = atan2(pt2[1] - centroid[1], pt2[0] - centroid[0]) - angle0

        angle1 = angle1 + 2*pi if angle1 < 0. else angle1
        angle2 = angle2 + 2*pi if angle2 <= 0. else angle2

        if allow_duplicates:
            if angle2 < angle1:
                ccw = False
                break
        else:
            if angle2 <= angle1:
                ccw = False
                break
    return ccw


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


