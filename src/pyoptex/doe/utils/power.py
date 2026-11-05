"""
Module containing power computations shared by all design types.
"""

import numpy as np
from scipy import stats


def t_test_power(var, df, effect_size=1.0, alpha=0.05):
    """
    Power of the two-sided t-test of each parameter, given the variance of
    its estimate (in units of the error variance) and the degrees of freedom.

    This is the part shared by every design type: without random effects
    ``var`` is the diagonal of ``(X'X)^{-1}`` and ``df = n - p`` (see
    :py:func:`power <pyoptex.doe.classical.evaluate.power>`); with random
    effects a Kenward-Roger approximation provides an adjusted variance and
    per-parameter degrees of freedom, which can be passed as arrays.

    The test statistic follows a noncentral t distribution with *df* degrees
    of freedom and noncentrality ``effect_size / sqrt(var)``.

    Parameters
    ----------
    var : np.ndarray
        Variance of each parameter estimate divided by the error variance.
        Infinite for parameters that are not estimable.
    df : float or np.ndarray
        Degrees of freedom of the t-test, one value or one per parameter.
        May be non-integer, as for Kenward-Roger.
    effect_size : float or np.ndarray
        Anticipated coefficient divided by the error standard deviation,
        one value for all parameters or one per parameter.
    alpha : float
        Significance level of the test.

    Returns
    -------
    power : np.ndarray
        Power for each parameter; zero for parameters that are not estimable.
    """
    assert 0 < alpha < 1, f"alpha must be between 0 and 1 (got {alpha})"
    var = np.asarray(var, dtype=np.float64)
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), var.shape)
    assert np.all(df > 0), "The degrees of freedom must be positive"
    ncp = np.abs(np.broadcast_to(np.asarray(effect_size, dtype=np.float64), var.shape)) / np.sqrt(var)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    pw = stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return np.where(np.isinf(var), 0.0, pw)
