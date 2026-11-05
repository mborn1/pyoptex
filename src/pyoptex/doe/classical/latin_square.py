"""
Module for Latin, Graeco-Latin and hyper-Graeco-Latin square designs.

References: Montgomery, "Design and Analysis of Experiments" (Latin and
Graeco-Latin squares); Fisher & Yates, "Statistical Tables for Biological,
Agricultural and Medical Research" (randomization of Latin squares).
"""

import numpy as np

from ._paley import _galois_field, _prime_power
from .utils import _to_dataframe, _validate_factors


def latin_square(factors):
    """
    Generate a Latin square design, or a (hyper-)Graeco-Latin square.

    The first factor defines the rows, the second the columns, and every
    further factor is a "symbol" factor whose levels form a Latin square:
    each level appears exactly once in every row and every column.  With
    3 factors this is a Latin square, with 4 a Graeco-Latin square, and
    with 5 to t + 1 a hyper-Graeco-Latin square, in which every pair of
    factors moreover contains each combination of levels exactly once.
    The design has t^2 runs for factors with t levels.  For t = 3, 4 and 5
    the complete sets of t + 1 factors are equivalent to the Taguchi L9,
    L16b and L25 (the same arrays up to the order of runs, columns and
    levels).

    Construction, for row i and column j (0 to t - 1):

    * one symbol factor: ``(i + j) mod t``, for every t >= 2;
    * two or more symbol factors: symbol factor a = 1, ..., t - 1 gets
      ``i + a j`` in the finite field GF(t), which requires t to be a
      prime power.  For t = 2 and t = 6 no two orthogonal Latin squares
      exist; other orders that are not prime powers (10, 12, ...) are
      not supported.

    There are at most t - 1 mutually orthogonal Latin squares, so at most
    t + 1 factors.  The design is deterministic; in practice the rows,
    columns and symbol labels are randomly permuted before running the
    experiment (Fisher & Yates), whereas
    :py:func:`randomize <pyoptex.doe.classical.randomize>` only shuffles
    the run order.  Incomplete block designs (e.g. balanced incomplete
    block designs) are not included; for those an optimal design with a
    block structure is the route.

    Parameters
    ----------
    factors : list(Factor)
        The row factor, the column factor and one or more symbol
        factors, all categorical with the same number of levels t.

    Returns
    -------
    Y : pd.DataFrame
        The design (denormalized), with t^2 runs.
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 3, f"A Latin square needs a row, a column and at least one symbol factor (got {k} factors)"
    for f in factors:
        assert not f.is_continuous, f"Factor {f.name} must be categorical for a Latin square"
    t = len(factors[0].levels)
    for f in factors:
        assert len(f.levels) == t, (
            f"All factors must have the same number of levels ({t}), but {f.name} has {len(f.levels)}"
        )
    assert t >= 2, f"The factors must have at least 2 levels (got {t})"
    assert k <= t + 1, (
        f"At most {t + 1} factors are possible with {t} levels (at most {t - 1} mutually orthogonal Latin squares), "
        f"got {k}"
    )

    i, j = np.divmod(np.arange(t * t), t)
    columns = [i, j]
    if k == 3:
        columns.append((i + j) % t)
    else:
        assert t != 6, "No two orthogonal Latin squares of order 6 exist"
        pe = _prime_power(t)
        assert pe is not None, f"Orthogonal Latin squares of order {t} are only supported when {t} is a prime power"
        elems, mul = _galois_field(*pe)
        p = pe[0]
        index = {e: n for n, e in enumerate(elems)}
        for a in range(1, k - 1):
            symbol = []
            for ii, jj in zip(i, j, strict=True):
                aj = mul(elems[a], elems[jj])
                symbol.append(index[tuple((x + y) % p for x, y in zip(elems[ii], aj, strict=True))])
            columns.append(np.array(symbol))

    return _to_dataframe(np.column_stack(columns).astype(np.float64), factors)
