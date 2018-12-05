import numpy as np

from functools import lru_cache
from scipy import integrate
from scipy.optimize import fsolve


@lru_cache(maxsize=16)
def gauss_points(ngp, integration='rectilinear'):
    """
    Compute the Gauss point values and weights for an arbitrary number of Gauss points.

    Parameters
    ----------
    ngp: int
        Number of gauss points to compute.
    integration: str, {'rectilinear', 'triangular'}
        Type of Gaussian integration to be performed with the gauss points.

    Returns
    -------
    z: list(float)
        Gauss point values.
    w: list(float)
        Gauss point weights.

    References
    ----------
    .. [1] Fish, Jacob and Belytschko, Ted. "A First Course in Finite Elements."
           John Wiley & Sons, 2007. Chapters 4.6, 7.8.
    """
    if integration == 'rectilinear':
        return _gauss_points_1d(ngp)
    else:
        raise NotImplementedError("Gauss quadrature for triangular domains has not been implemented.")


def weighted_integration(func, values, weights, ndims=1):
    """
    Perform weighted numerical integration on a function with given values and weights.

    Input func can either be a function or an array of functions.

    For integration over a number of dimensions greater than 1, the integration
    if performed for the same values and weights in each direction.

    Parameters
    ----------
    func: function or array_like(function)
        The function or array of functions to perform
    values: list(float)
        The values at which to evaluate the function.
    weights: list(float)
        Weights corresponding to each value.
    ndims: int
        Number of dimensions over which to perform integration.

    Returns
    -------
    I: float or list(float)
        Integrated function or vector of integrated functions.
    """
    vfunc = np.array(func).flatten()
    if ndims == 1:
        v = np.array([_weighted_integration_1d(f, values, weights) for f in vfunc])
    elif ndims == 2:
        v = np.array([_weighted_integration_2d(f, values, weights) for f in vfunc])
    else:
        raise NotImplementedError("Weighted (Gaussian) integration for more than two dimensions has not been implemented")
    return v.reshape(func.shape)


@lru_cache(maxsize=16)
def _gauss_points_1d(ngp):
    """
    Compute 1-dimensional Gauss point values and weights for numerical integration.

    Returns
    -------
    z: list(float)
        Gauss point values.
    w: list(float)
        Gauss point weights.

    References
    ----------
    .. [1] Fish, Jacob and Belytschko, Ted. "A First Course in Finite Elements."
           John Wiley & Sons, 2007. Chapter 4.6.
    """
    M = _M(ngp)
    MT = M.transpose()
    P = _P(ngp*2)

    def equations(variables):
        eta = variables[:ngp]
        W = variables[ngp:]
        return [np.sum([MT[j, i](eta[i]) * W[i] for i in range(ngp)] + [-P[j]]) for j in range(ngp*2)]

    solution = fsolve(equations, np.linspace(-1., 1., ngp*2))
    return solution[:ngp], solution[ngp:]


@lru_cache(maxsize=16)
def _M(ngp):
    """
    Matrix of Gauss point polynomial functions.

    The matrix columns are increasing polynomials of the Gauss variable
    starting with 1.
    Ex, for ngp=2:
        _M = [1   z   z**2   z**3
              1   z   z**2   z**3]

    Returns
    -------
    _M: numpy.ndarry (ndim=2, shape=(ngp, 2*ngp))
    """
    M = np.empty((ngp, ngp*2), dtype=object)
    for i in range(ngp*2):
        def f(eta, ii=i):
            return eta**ii
        M[:, i] = [f for _ in range(ngp)]
    return M


@lru_cache(maxsize=16)
def _P(n):
    """
    Calculate the vector of n integrated polynomials from -1 to 1.

    Returns the evaluated n polynomial term integrals starting with x^0.
    For example, _P(4) would return the integral from -1 to 1 of:
        [1, x, x^2, x^3]

    Parameters
    ----------
    n: int
        The number of polynomial terms to include.

    Returns
    -------
    P: ndarray(ndim=1, dtype=float)
        The integrated values.

    Notes
    -----
    Function inputs are cached so that calculations are not repeated on each call.
    """
    return [integrate.quad(lambda x: x**i, -1., 1.)[0] for i in range(n)]


def _weighted_integration_1d(func, values, weights):
    """
    Perform weighted numerical integration of a function in 1 dimension.

    Parameters
    ----------
    func: function
        The function to integrate.
    values: list(float)
        The values at which to evaluate the function.
    weights: list(float)
        The weights corresponding to each value.

    Returns
    -------
    sum: float
        Result of the numerical integration.
    """
    return sum([func(v)*w for v, w in zip(values, weights)])


def _weighted_integration_2d(func, values, weights):
    """
    Perform weighted numerical integration of a function in 2 dimensions.

    The same values and weights are used for integration in each direction.

    func: function
        The function to integrate.
    values: list(float)
        The values at which to evaluate the function.
    weights: list(float)
        The weights corresponding to each value.

    Returns
    -------
    sum: float
        Result of the numerical integration.
    """
    sum = 0.
    for i in range(len(values)):
        for j in range(len(values)):
            sum += func(values[i], values[j]) * weights[i] * weights[j]
    return sum
