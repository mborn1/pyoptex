"""
Shared helpers for classical design generation.
"""

from itertools import product

import numpy as np
import pandas as pd

from ...utils.factor import FactorMixin


def _validate_factors(factors):
    """Ensure factors is a non-empty list of FactorMixin instances."""
    assert len(factors) > 0, "At least one factor must be provided"
    for i, f in enumerate(factors):
        assert isinstance(f, FactorMixin), f"Factor {i} is not a valid Factor instance"


def _factor_encoding_params(factors):
    """
    Extract the encoding parameters from a list of factors.

    Returns the ``effect_types``, ``coords``, and ``colstart`` arrays
    required by the encoded design utilities (``full_factorial``,
    ``encode_design``, ``decode_design``).

    Parameters
    ----------
    factors : list(Factor)
        The list of factors.

    Returns
    -------
    effect_types : np.ndarray
        1 for continuous, ``len(levels)`` for categorical.
    coords : list(np.ndarray)
        Per-factor encoded coordinates (from ``Factor.coords_``).
    colstart : np.ndarray
        Column-start indices in the encoded design matrix.
    """
    effect_types = np.array(
        [1 if f.is_continuous else len(f.levels) for f in factors],
        dtype=np.int64,
    )
    coords = [f.coords_ for f in factors]
    colstart = np.concatenate(
        ([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))),
        dtype=np.int64,
    )
    return effect_types, coords, colstart


def _to_dataframe(design, factors):
    """
    Convert a normalized design matrix to a denormalized DataFrame.

    Parameters
    ----------
    design : np.ndarray
        Normalized design matrix (nruns x nfactors). Continuous values
        in [-1, 1], categorical values as integer level indices.
    factors : list(Factor)
        The factor definitions.

    Returns
    -------
    Y : pd.DataFrame
        Denormalized, decoded design.
    """
    col_names = [str(f.name) for f in factors]
    Y = pd.DataFrame(design, columns=col_names)
    for f in factors:
        Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    return Y


def _two_level_normalized(design, factors):
    """
    Map a +/-1 coded two-level design to the normalized values expected
    by :py:meth:`denormalize <pyoptex.utils.factor.FactorMixin.denormalize>`.

    * Continuous factors take their two coded levels: the normalized
      ``levels`` when explicitly specified (so e.g. ``levels=[10, 20]``
      is honored even with the default ``min``/``max``), otherwise the
      coded extremes ``-1``/``+1`` (which denormalize to ``min``/``max``).
    * Categorical factors take their level indices ``{0, 1}``.

    Parameters
    ----------
    design : np.ndarray
        The +/-1 coded design matrix (nruns x nfactors).
    factors : list(Factor)
        The two-level factor definitions.

    Returns
    -------
    normalized : np.ndarray
        Normalized design matrix ready for ``_to_dataframe``.
    """
    normalized = np.empty_like(design, dtype=np.float64)
    for j, f in enumerate(factors):
        normalized[:, j] = _levels(f, 2)[(design[:, j] > 0).astype(np.int64)]
    return normalized


def _levels(f, n_levels):
    """
    Normalized level set of a factor for a classical design.

    * Categorical factors use their level indices ``0, ..., L - 1``.
    * Continuous factors with explicit ``levels`` use those levels
      (normalized), regardless of *n_levels*.
    * Other continuous factors use *n_levels* equally spaced points
      in ``[-1, 1]``, the design's own default.

    Classical designs use this instead of ``Factor.coords_``, whose
    three-level default is the candidate set for coordinate exchange.

    Parameters
    ----------
    f : Factor
        The factor.
    n_levels : int
        Default number of levels for continuous factors without
        explicit ``levels``.

    Returns
    -------
    levels : np.ndarray
        1-D array of normalized levels.
    """
    if not f.is_continuous:
        return np.arange(len(f.levels), dtype=np.float64)
    if f.levels is not None:
        return f.normalize(np.array(f.levels, dtype=np.float64))
    return np.linspace(-1, 1, n_levels)


def _center_points(factors, n_levels, n_center):
    """
    Normalized center runs for a factorial-type design.

    Continuous factors are set to the middle of their level range.
    The *n_center* center runs are repeated for every combination of
    categorical levels, so a 3-level categorical factor with
    ``n_center=2`` gives 6 center runs.

    Parameters
    ----------
    factors : list(Factor)
        The factors.
    n_levels : int
        Default number of levels for continuous factors, see :py:func:`_levels`.
    n_center : int
        Number of center runs per categorical level combination.

    Returns
    -------
    center : np.ndarray
        Normalized center runs (n_center * prod(categorical levels) x nfactors).
    """
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"
    if n_center == 0:
        return np.empty((0, len(factors)), dtype=np.float64)
    assert any(f.is_continuous for f in factors), "Center points require at least one continuous factor"

    grid = []
    for f in factors:
        lv = _levels(f, n_levels)
        grid.append([(lv.min() + lv.max()) / 2] if f.is_continuous else lv)
    grid = np.array(list(product(*grid)), dtype=np.float64)
    return np.repeat(grid, n_center, axis=0)


def _assert_no_categorical_center(factors, n_center, design_name):
    """Center points with categorical factors are only defined for full factorials."""
    if n_center > 0:
        for f in factors:
            assert f.is_continuous, (
                f"Center points are not defined for categorical factor {f.name} in a {design_name}; "
                f"use n_center=0 and add center runs manually"
            )


def _validate_space_filling_factors(factors, design_name):
    """Space-filling designs require continuous, non-mixture factors."""
    _validate_factors(factors)
    for f in factors:
        assert f.is_continuous and not f.is_mixture, f"Factor {f.name} must be continuous for {design_name}"


def _qmc_engine(engine, d, **kwargs):
    """
    Create a :py:mod:`scipy.stats.qmc` engine seeded from numpy's global
    random generator, so :py:func:`set_seed <pyoptex._seed.set_seed>` applies.
    """
    seed = np.random.randint(2**31 - 1)
    try:
        return engine(d, rng=seed, **kwargs)
    except TypeError:
        # scipy < 1.15 names the argument seed
        return engine(d, seed=seed, **kwargs)


def _space_filling_dataframe(unit, factors):
    """Scale a design in [0, 1]^k to each factor's [min, max] range."""
    return _to_dataframe(2.0 * unit - 1.0, factors)


def randomize(Y):
    """
    Randomize the run order of a design.

    Classical designs are returned in standard order; experiments should
    be run in a random order. The rows are shuffled with numpy's global
    random generator, so :py:func:`set_seed <pyoptex._seed.set_seed>`
    makes the order reproducible.

    Parameters
    ----------
    Y : pd.DataFrame
        The design.

    Returns
    -------
    Y : pd.DataFrame
        The same runs in random order. The index is kept, so it still
        gives the standard order of each run.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a DataFrame"
    return Y.iloc[np.random.permutation(len(Y))]
