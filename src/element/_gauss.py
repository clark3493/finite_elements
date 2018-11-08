import numpy as np

from functools import lru_cache
from scipy import integrate
from scipy.optimize import fsolve


@lru_cache(maxsize=16)
def gauss_points(ngp, ndims=1):
    """
    Compute the Gauss point values and weights for an arbitrary number of Gauss points.

    Returns
    -------
    z: list(float)
        Gauss point values.
    w: list(float)
        Gauss point weights.
    """
    if ndims == 1:
        return _gauss_points_1d(ngp)
    else:
        raise NotImplementedError("Gauss quadrature for greater than 1 dimension has not been implemented.")


def weighted_integration(func, values, weights):
    """
    Perform weighted numerical integration on a function with given values and weights.

    Input func can either be a function or a vector of functions.

    Parameters
    ----------
    func: function or list(function)
        The function or list of functions to perform
    values: list(float)
        The values at which to evaluate the function.
    weights: list(float)
        Weights correspoding to each value.

    Returns
    -------
    I: float or list(float)
        Integrated function or vector of integrated functions.
    """
    try:
        v = [weighted_integration(f, values, weights) for f in func]
        return np.array(v).reshape(func.shape)
    except TypeError:
        return sum([func(v)*w for v, w in zip(values, weights)])


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
