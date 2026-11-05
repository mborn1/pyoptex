"""
Module for quasi-random and uniform space-filling design generation.

Thin wrappers around :py:mod:`scipy.stats.qmc`, scaled from [0, 1] to each
factor's [min, max] range. The engines are seeded from numpy's global random
generator, so :py:func:`set_seed <pyoptex._seed.set_seed>` makes them
reproducible.
"""

from scipy.stats import qmc

from .utils import _qmc_engine, _space_filling_dataframe, _validate_space_filling_factors


def sobol(factors, n, scramble=True):
    """
    Generate a Sobol' sequence design.

    A low-discrepancy (quasi-random) sequence that fills the factor space
    more evenly than random sampling.  Its balance properties require *n*
    to be a power of 2; scipy warns otherwise.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    n : int
        Number of runs, preferably a power of 2.
    scramble : bool
        Whether to use Owen scrambling (default), which avoids the
        unscrambled sequence starting at the ``min`` corner.

    Returns
    -------
    Y : pd.DataFrame
        The Sobol' design (denormalized).
    """
    _validate_space_filling_factors(factors, "a Sobol' design")
    assert n >= 1, f"n must be at least 1 (got {n})"
    unit = _qmc_engine(qmc.Sobol, len(factors), scramble=scramble).random(n)
    return _space_filling_dataframe(unit, factors)


def halton(factors, n, scramble=True):
    """
    Generate a Halton sequence design.

    A low-discrepancy (quasi-random) sequence that fills the factor space
    more evenly than random sampling, for any number of runs.  Without
    scrambling, the projections onto pairs of factors become strongly
    correlated for more than about 10 factors.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    n : int
        Number of runs.
    scramble : bool
        Whether to scramble the sequence (default).

    Returns
    -------
    Y : pd.DataFrame
        The Halton design (denormalized).
    """
    _validate_space_filling_factors(factors, "a Halton design")
    assert n >= 1, f"n must be at least 1 (got {n})"
    unit = _qmc_engine(qmc.Halton, len(factors), scramble=scramble).random(n)
    return _space_filling_dataframe(unit, factors)


def uniform(factors, n):
    """
    Generate a uniform design.

    A U-type design in the sense of Fang: every factor takes *n* equally
    spaced levels, each exactly once, at the centers of *n* equal strata
    of its range.  The levels are combined to minimize the centered L2
    discrepancy, using :py:class:`scipy.stats.qmc.LatinHypercube` with
    ``optimization='random-cd'`` (a heuristic search, so the result is
    not guaranteed to be the minimum-discrepancy design).

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors.
    n : int
        Number of runs, which is also the number of levels per factor.

    Returns
    -------
    Y : pd.DataFrame
        The uniform design (denormalized).
    """
    _validate_space_filling_factors(factors, "a uniform design")
    assert n >= 2, f"A uniform design requires at least 2 runs (got {n})"
    unit = _qmc_engine(qmc.LatinHypercube, len(factors), scramble=False, optimization="random-cd").random(n)
    return _space_filling_dataframe(unit, factors)
