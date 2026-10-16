"""
Shared helpers for classical design generation.
"""

from itertools import combinations, product
from math import comb

import numpy as np
import pandas as pd

from ...utils.factor import FactorMixin

# Name of the column with the block number (1, 2, ...) in blocked designs
BLOCK_COLUMN = "Block"


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


def _gf2_subspace_bases(p, q):
    """
    Every q-dimensional subspace of GF(2)^p exactly once, as a basis of q
    bitmasks in reduced row echelon form (the pivot is the lowest set bit,
    and no other basis vector has that bit set).
    """
    for pivots in combinations(range(p), q):
        free = [[j for j in range(pivots[i] + 1, p) if j not in pivots] for i in range(q)]
        n_free = sum(len(fr) for fr in free)
        for fill in range(2**n_free):
            basis, bit = [], 0
            for i in range(q):
                word = 1 << pivots[i]
                for j in free[i]:
                    if fill >> bit & 1:
                        word |= 1 << j
                    bit += 1
                basis.append(word)
            yield basis


def _shortest_effects(X, max_effects=200000):
    """
    Map every column that is a product of at most j_max columns of the +/-1
    coded design X (up to sign) to the shortest such product, where j_max is
    the largest order with at most *max_effects* effects in total.

    Returns the lookup ``{column bytes: factor indices}`` and j_max.
    """
    k = X.shape[1]
    lookup = {}
    total, j_max = 0, 0
    for j in range(1, k + 1):
        if j > 1 and total + comb(k, j) > max_effects:
            break
        total += comb(k, j)
        j_max = j
        for effect in combinations(range(k), j):
            col = np.prod(X[:, effect], axis=1)
            lookup.setdefault((col * col[0]).astype(np.int8).tobytes(), effect)
    return lookup, j_max


def _block_scheme(X, base, n_blocks):
    """
    Minimum aberration blocking of a two-level design.

    The blocks are defined by q = log2(n_blocks) independent products of
    base columns; all 2^q - 1 products of these are confounded with blocks.
    Every such product is identified with the shortest effect it is aliased
    with, and the scheme minimizes the number of confounded main effects
    (which must be zero), then two-factor interactions, then three-factor
    interactions, and so on, by exhaustive search over all schemes.

    Parameters
    ----------
    X : np.ndarray
        The +/-1 coded two-level design (nruns x nfactors).
    base : list(int)
        Indices of the base factors: every column of X is a product of
        base columns.
    n_blocks : int
        Number of blocks, a power of 2.

    Returns
    -------
    labels : np.ndarray
        Block index (0 to n_blocks - 1) of every run.
    confounded : list(tuple(int))
        The effects confounded with blocks, as factor indices, shortest first.
    """
    q = int(n_blocks).bit_length() - 1
    assert n_blocks >= 2 and 2**q == n_blocks, f"n_blocks must be a power of 2 (got {n_blocks})"
    p = len(base)
    assert q < p, f"{n_blocks} blocks of {X.shape[0] // n_blocks} runs are too small for {X.shape[1]} factors"

    lookup, j_max = _shortest_effects(X)

    # Shortest aliased effect and its order for every nonzero product of base columns
    words = np.arange(2**p)
    cols = np.ones((2**p, X.shape[0]), dtype=np.int8)
    for i in range(p):
        cols[(words >> i & 1) == 1] *= X[:, base[i]].astype(np.int8)
    effects = [()] + [
        lookup.get((cols[w] * cols[w][0]).tobytes(), tuple(base[i] for i in range(p) if w >> i & 1))
        for w in range(1, 2**p)
    ]
    orders = np.array([len(e) if len(e) <= j_max else j_max + 1 for e in effects])

    best = None
    for basis in _gf2_subspace_bases(p, q):
        span = [0]
        for b in basis:
            span += [s ^ b for s in span]
        counts = tuple(np.bincount(orders[span[1:]], minlength=j_max + 2)[1:])
        if best is None or counts < best[0]:
            best = (counts, basis, span[1:])

    counts, basis, span = best
    assert counts[0] == 0, f"{n_blocks} blocks would confound a main effect with blocks; use fewer blocks or more runs"
    labels = sum((cols[b][:] > 0).astype(np.int64) << i for i, b in enumerate(basis))
    confounded = sorted((effects[w] for w in span), key=lambda e: (len(e), e))
    return np.asarray(labels), confounded


def _blocked_dataframe(design, labels, center, factors, confounded=None):
    """
    Denormalized design with a block column: the runs grouped by block (in
    their original order within a block), each block followed by the same
    *center* runs. The effects confounded with blocks, if given, are stored
    as lists of factor names in ``Y.attrs['confounded_with_blocks']``.
    """
    assert all(str(f.name) != BLOCK_COLUMN for f in factors), (
        f"A factor cannot be named '{BLOCK_COLUMN}' in a blocked design"
    )
    parts, blocks = [], []
    for b in range(int(labels.max()) + 1):
        runs = np.vstack([design[labels == b], center])
        parts.append(runs)
        blocks.append(np.full(len(runs), b + 1, dtype=np.int64))
    Y = _to_dataframe(np.vstack(parts), factors)
    Y[BLOCK_COLUMN] = np.concatenate(blocks)
    if confounded is not None:
        Y.attrs["confounded_with_blocks"] = [[str(factors[i].name) for i in e] for e in confounded]
    return Y


def randomize(Y, block_column=BLOCK_COLUMN, shuffle_blocks=False):
    """
    Randomize the run order of a design.

    Classical designs are returned in standard order; experiments should
    be run in a random order. The rows are shuffled with numpy's global
    random generator, so :py:func:`set_seed <pyoptex._seed.set_seed>`
    makes the order reproducible.

    In a blocked design (with a *block_column*), the runs are shuffled
    within each block and every block stays contiguous, since all runs of
    a block must be done under the same conditions (e.g. on the same
    day).  By default the blocks keep their order; with
    ``shuffle_blocks=True`` the order of the blocks is randomized too.

    *block_column* can also be a list of columns, whose combinations of
    values define the groups.  For a classical split-plot experiment,
    ``randomize(Y, block_column=[hard-to-change factors],
    shuffle_blocks=True)`` keeps the runs with the same settings of the
    hard-to-change factors together as whole plots, in random order, and
    randomizes the runs within each whole plot.  The design is then a
    split-plot design and should be analysed with the corresponding
    (GLS) model; :py:mod:`pyoptex.doe.fixed_structure.splitk_plot`
    creates optimal split-plot designs and evaluates them with that
    model.

    Parameters
    ----------
    Y : pd.DataFrame
        The design.
    block_column : str, list(str) or None
        Name of the block column (``'Block'`` for blocked classical
        designs), or a list of columns whose combinations of values
        define the groups.  A single column that the design does not have
        is ignored; ``None`` shuffles all runs regardless of blocks.
    shuffle_blocks : bool
        Whether to randomize the order of the blocks as well (default
        False).

    Returns
    -------
    Y : pd.DataFrame
        The same runs in random order. The index is kept, so it still
        gives the standard order of each run.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a DataFrame"
    if isinstance(block_column, str):
        block_column = [block_column] if block_column in Y.columns else None
    if block_column is None:
        return Y.iloc[np.random.permutation(len(Y))]

    block_column = list(block_column)
    assert len(block_column) > 0, "block_column must name at least one column"
    for c in block_column:
        assert c in Y.columns, f"Column {c} in block_column is not a column of the design"

    # Groups in order of first appearance, each shuffled internally
    groups = Y.groupby(block_column, sort=False, dropna=False).ngroup().to_numpy()
    order = [np.random.permutation(np.flatnonzero(groups == g)) for g in range(groups.max() + 1)]
    if shuffle_blocks:
        order = [order[g] for g in np.random.permutation(len(order))]
    return Y.iloc[np.concatenate(order)]
