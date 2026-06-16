"""
Module containing evaluation functions for classical designs.

These functions compute standard design quality metrics (D-, A-,
G-efficiency, estimation variance, alias matrix) for designs that
have no random-effect / covariance structure — only the simple
X'X information matrix is needed.
"""

import numpy as np
import pandas as pd

from ...utils.design import encode_design
from .utils import _factor_encoding_params


def _prepare_X(Y, factors, Y2X):
    """Normalize, encode, and compute the model matrix X."""
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded DataFrame"
    Y = Y.copy()

    col_names = [str(f.name) for f in factors]
    effect_types, coords, _ = _factor_encoding_params(factors)

    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    Ynp = Y[col_names].astype(float).to_numpy()
    Yenc = encode_design(Ynp, effect_types, coords)
    X = Y2X(Yenc)
    return X


def d_efficiency(Y, factors, Y2X):
    """
    Compute the D-efficiency of a design.

    D-efficiency is defined as ``(det(X'X))^{1/p} / n`` scaled to a
    percentage, where *n* is the number of runs and *p* the number of
    model parameters.  For an orthogonal +/-1 coded design (``X'X = n I``)
    this equals 100 %, the theoretical bound.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.

    Returns
    -------
    d_eff : float
        D-efficiency as a percentage (0-100+).
    """
    X = _prepare_X(Y, factors, Y2X)
    n, p = X.shape

    det_val = np.linalg.det(X.T @ X)
    if det_val <= 0:
        return 0.0
    return (det_val ** (1.0 / p)) / n * 100.0


def a_efficiency(Y, factors, Y2X):
    """
    Compute the A-efficiency of a design.

    A-efficiency is ``p / trace((X'X)^{-1})`` expressed as a percentage.
    Lower trace means smaller average variance of parameter estimates.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.

    Returns
    -------
    a_eff : float
        A-efficiency as a percentage (0-100+).
    """
    X = _prepare_X(Y, factors, Y2X)
    n, p = X.shape

    M = X.T @ X
    if np.linalg.matrix_rank(X) < p:
        return 0.0

    trace_val = np.trace(np.linalg.inv(M))
    return (p / trace_val) * 100.0 / n


def estimation_variance(Y, factors, Y2X):
    """
    Compute the variance of each parameter estimate.

    Returns the diagonal of ``(X'X)^{-1}``, which gives the variance
    of each parameter estimator (assuming unit error variance).

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.

    Returns
    -------
    var : np.ndarray
        1-D array of parameter estimation variances. If the model is not
        estimable, all variances are infinite.
    """
    X = _prepare_X(Y, factors, Y2X)
    p = X.shape[1]
    if np.linalg.matrix_rank(X) < p:
        return np.full(p, np.inf)

    Minv = np.linalg.inv(X.T @ X)
    return np.diag(Minv)


def alias_matrix(Y, factors, Y2X_primary, Y2X_potential):
    """
    Compute the alias matrix between primary and potential model terms.

    The alias matrix ``A = (X1'X1)^{-1} X1' X2`` shows how potential
    terms (columns of X2) bias the estimates of the primary terms
    (columns of X1) when the potential terms are active but omitted
    from the fitted model.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X_primary : callable
        Y2X for the primary (fitted) model.
    Y2X_potential : callable
        Y2X for the potential (omitted) terms.

    Returns
    -------
    A : np.ndarray
        Alias matrix of shape ``(p_primary, p_potential)``. If the
        primary model is not estimable, all entries are undefined
        (``np.nan``).
    """
    X1 = _prepare_X(Y, factors, Y2X_primary)
    X2 = _prepare_X(Y, factors, Y2X_potential)
    if np.linalg.matrix_rank(X1) < X1.shape[1]:
        return np.full((X1.shape[1], X2.shape[1]), np.nan)

    return np.linalg.solve(X1.T @ X1, X1.T @ X2)


def design_info(Y, factors, Y2X, n_samples=10000):
    """
    Compute a summary of design quality metrics.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.
    n_samples : int
        Number of random samples for G-efficiency.

    Returns
    -------
    info : dict
        Dictionary with keys ``'n_runs'``, ``'n_params'``,
        ``'d_efficiency'``, ``'a_efficiency'``,
        and ``'estimation_variance'``.
    """
    X = _prepare_X(Y, factors, Y2X)
    n, p = X.shape

    return {
        "n_runs": n,
        "n_params": p,
        "d_efficiency": d_efficiency(Y, factors, Y2X),
        "a_efficiency": a_efficiency(Y, factors, Y2X),
        "estimation_variance": estimation_variance(Y, factors, Y2X),
    }
