"""
Module containing evaluation functions for classical designs.

These functions compute standard design quality metrics (D-, A-,
G-efficiency, average prediction variance, estimation variance, alias
matrix) for designs that have no random-effect / covariance structure —
only the simple X'X information matrix is needed.

The metrics use the same definitions as the optimal-design metrics in
:py:mod:`pyoptex.doe.fixed_structure.metric` without random effects
(``V = I``), so a classical and an optimal design can be compared
directly:

* ``d_efficiency = 100 * Dopt / n``
* ``average_prediction_variance = -Iopt``
"""

import numpy as np
import pandas as pd

from ...utils.design import encode_design
from ..utils.init import init_single_unconstrained
from .utils import _factor_encoding_params


def _prepare(Y, factors, Y2X):
    """Normalize the design and compute the model matrix X. Returns (normalized design, X)."""
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded DataFrame"
    Y = Y.copy()

    col_names = [str(f.name) for f in factors]
    effect_types, coords, _ = _factor_encoding_params(factors)

    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    Ynp = Y[col_names].astype(float).to_numpy()
    Yenc = encode_design(Ynp, effect_types, coords)
    X = Y2X(Yenc)
    return Ynp, X


def _prepare_X(Y, factors, Y2X):
    """Normalize, encode, and compute the model matrix X."""
    return _prepare(Y, factors, Y2X)[1]


def _region_samples(factors, Y2X, n_samples):
    """
    Model matrix of *n_samples* random points of the design region: continuous
    factors uniform in their [min, max] range, categorical factors at a random
    level. This is the same sampling as the moments matrix of
    :py:class:`Iopt <pyoptex.doe.fixed_structure.metric.Iopt>`.
    """
    assert n_samples >= 1, f"n_samples must be at least 1 (got {n_samples})"
    effect_types, coords, colstart = _factor_encoding_params(factors)
    coords = [None if f.is_continuous else c for f, c in zip(factors, coords)]
    run = np.zeros((n_samples, colstart[-1]), dtype=np.float64)
    run = init_single_unconstrained(colstart, coords, run, effect_types)
    return Y2X(run)


def _in_region(Ynp, factors):
    """Mask of the design points whose continuous factors lie within [min, max]."""
    cont = np.array([f.is_continuous for f in factors])
    return np.all(np.abs(Ynp[:, cont]) <= 1 + 1e-9, axis=1)


def _prediction_variance(X, Xs):
    """Prediction variance ``f(x)' (X'X)^{-1} f(x)`` for every row of Xs."""
    return np.sum(Xs * np.linalg.solve(X.T @ X, Xs.T).T, axis=1)


def _g_efficiency(X, Xs):
    """G-efficiency (%) from the model matrix X and region samples Xs, see :py:func:`g_efficiency`."""
    n, p = X.shape
    if np.linalg.matrix_rank(X) < p:
        return 0.0
    return p / (n * np.max(_prediction_variance(X, Xs))) * 100.0


def _average_prediction_variance(X, Xs):
    """Average prediction variance from X and region samples Xs, see :py:func:`average_prediction_variance`."""
    p = X.shape[1]
    if np.linalg.matrix_rank(X) < p:
        return np.inf
    moments = Xs.T @ Xs / Xs.shape[0]
    return float(np.trace(np.linalg.solve(X.T @ X, moments)))


def _d_efficiency(X):
    """D-efficiency (%) from the model matrix X, see :py:func:`d_efficiency`."""
    n, p = X.shape
    # Log-determinant avoids overflow / underflow of det(X'X), which grows like n^p.
    sign, logdet = np.linalg.slogdet(X.T @ X)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p)) / n * 100.0


def _a_efficiency(X):
    """A-efficiency (%) from the model matrix X, see :py:func:`a_efficiency`."""
    n, p = X.shape
    if np.linalg.matrix_rank(X) < p:
        return 0.0

    trace_val = np.trace(np.linalg.inv(X.T @ X))
    return (p / trace_val) * 100.0 / n


def _estimation_variance(X):
    """Parameter estimation variances from the model matrix X, see :py:func:`estimation_variance`."""
    p = X.shape[1]
    if np.linalg.matrix_rank(X) < p:
        return np.full(p, np.inf)

    return np.diag(np.linalg.inv(X.T @ X))


def d_efficiency(Y, factors, Y2X):
    """
    Compute the D-efficiency of a design.

    D-efficiency is defined as ``(det(X'X))^{1/p} / n`` scaled to a
    percentage, where *n* is the number of runs and *p* the number of
    model parameters.  For an orthogonal +/-1 coded design (``X'X = n I``)
    this equals 100 %, the theoretical bound.  It equals ``100 * Dopt / n``
    with :py:class:`Dopt <pyoptex.doe.fixed_structure.metric.Dopt>`
    without random effects.

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
    return _d_efficiency(_prepare_X(Y, factors, Y2X))


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
    return _a_efficiency(_prepare_X(Y, factors, Y2X))


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
    return _estimation_variance(_prepare_X(Y, factors, Y2X))


def g_efficiency(Y, factors, Y2X, n_samples=10000):
    """
    Compute the G-efficiency of a design.

    G-efficiency is ``p / (n * max d(x))`` expressed as a percentage,
    where ``d(x) = f(x)' (X'X)^{-1} f(x)`` is the prediction variance
    (unit error variance) and the maximum is taken over the design
    region: every continuous factor within its ``[min, max]`` range and
    every categorical factor at any level.  The maximum is estimated
    from *n_samples* random points plus the design points that lie in
    the region, which covers the common case where the maximum is
    attained at a design point (e.g. the corners of a factorial).

    The bound is 100 %, attained by e.g. a 2^k design with a
    main-effects model.  Values slightly above 100 % are possible when
    design points lie outside the factor ranges (e.g. the axial points
    of a rotatable CCD), as these are not part of the region.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.
    n_samples : int
        Number of random points of the design region. Uses numpy's
        global random generator, so
        :py:func:`set_seed <pyoptex._seed.set_seed>` applies.

    Returns
    -------
    g_eff : float
        G-efficiency as a percentage (0-100). Zero if the model is not
        estimable.
    """
    Ynp, X = _prepare(Y, factors, Y2X)
    Xs = np.vstack([_region_samples(factors, Y2X, n_samples), X[_in_region(Ynp, factors)]])
    return _g_efficiency(X, Xs)


def average_prediction_variance(Y, factors, Y2X, n_samples=10000):
    """
    Compute the average prediction variance of a design (the I-criterion).

    The average of ``d(x) = f(x)' (X'X)^{-1} f(x)`` (unit error variance)
    over the design region, estimated as ``trace((X'X)^{-1} M)`` with the
    moments matrix ``M`` from *n_samples* random points of the region.
    Lower is better.  It equals ``-Iopt`` with
    :py:class:`Iopt <pyoptex.doe.fixed_structure.metric.Iopt>` without
    random effects, and the ratio of two designs' values is their
    relative I-efficiency.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    factors : list(Factor)
        The factor definitions.
    Y2X : callable
        Function converting the encoded design matrix to the model matrix.
    n_samples : int
        Number of random points of the design region. Uses numpy's
        global random generator, so
        :py:func:`set_seed <pyoptex._seed.set_seed>` applies.

    Returns
    -------
    apv : float
        Average prediction variance. Infinite if the model is not
        estimable.
    """
    X = _prepare_X(Y, factors, Y2X)
    return _average_prediction_variance(X, _region_samples(factors, Y2X, n_samples))


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
        Number of random points of the design region for the
        G-efficiency and the average prediction variance.

    Returns
    -------
    info : dict
        Dictionary with keys ``'n_runs'``, ``'n_params'``,
        ``'d_efficiency'``, ``'a_efficiency'``, ``'g_efficiency'``,
        ``'average_prediction_variance'`` and ``'estimation_variance'``.
    """
    Ynp, X = _prepare(Y, factors, Y2X)
    n, p = X.shape
    Xs = _region_samples(factors, Y2X, n_samples)

    return {
        "n_runs": n,
        "n_params": p,
        "d_efficiency": _d_efficiency(X),
        "a_efficiency": _a_efficiency(X),
        "g_efficiency": _g_efficiency(X, np.vstack([Xs, X[_in_region(Ynp, factors)]])),
        "average_prediction_variance": _average_prediction_variance(X, Xs),
        "estimation_variance": _estimation_variance(X),
    }
