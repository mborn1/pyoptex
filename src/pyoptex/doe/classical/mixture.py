"""
Module for classical mixture design generation (simplex lattice and
simplex centroid designs, Scheffé 1958, 1963).

Following the pyoptex convention for mixtures, a mixture of q components
is specified with q - 1 mixture factors; the last component is implied as
one minus the sum of the others and is not a column of the design.
"""

from itertools import combinations, product

import numpy as np

from .utils import _to_dataframe, _validate_factors


def _validate_mixture_factors(factors, design_name):
    """Every factor must be an unconstrained mixture component (proportions from 0 to 1)."""
    _validate_factors(factors)
    for f in factors:
        assert f.is_mixture, f"Factor {f.name} must be a mixture factor for a {design_name}"
        assert min(f.levels) == 0 and max(f.levels) == 1, (
            f"Mixture factor {f.name} must range from 0 to 1 for a {design_name} (has levels from "
            f"{min(f.levels)} to {max(f.levels)}); constrained mixtures need an optimal design"
        )


def simplex_lattice(factors, m):
    """
    Generate a {q, m} simplex lattice design.

    Every blend whose component proportions are multiples of ``1 / m``
    (0, 1/m, ..., 1) and sum to 1: ``C(q + m - 1, m)`` runs for q
    components.  It supports a Scheffé polynomial of degree *m*.

    Parameters
    ----------
    factors : list(Factor)
        The first q - 1 mixture components (``type='mixture'``), each
        ranging from 0 to 1.  The last component is implied as one minus
        the sum of the others and is not part of the design.
    m : int
        Lattice degree: the number of equal parts each proportion is
        divided into.

    Returns
    -------
    Y : pd.DataFrame
        The simplex lattice design with q - 1 columns (denormalized).
    """
    _validate_mixture_factors(factors, "simplex lattice design")
    assert m >= 1, f"m must be at least 1 (got {m})"
    q = len(factors) + 1

    parts = [p for p in product(range(m + 1), repeat=q) if sum(p) == m]
    design = np.array(sorted(parts, reverse=True), dtype=np.float64) / m
    return _to_dataframe(design[:, :-1], factors)


def simplex_centroid(factors):
    """
    Generate a simplex centroid design.

    Every equal-proportion blend of 1 to q components: the q pure
    components, the ``C(q, 2)`` binary 50/50 blends, and so on up to the
    overall centroid, ``2^q - 1`` runs in total.  It supports Scheffé's
    special polynomial with all products of distinct components (e.g.
    the special cubic model for q = 3), which has exactly ``2^q - 1``
    terms.

    Parameters
    ----------
    factors : list(Factor)
        The first q - 1 mixture components (``type='mixture'``), each
        ranging from 0 to 1.  The last component is implied as one minus
        the sum of the others and is not part of the design.

    Returns
    -------
    Y : pd.DataFrame
        The simplex centroid design with q - 1 columns (denormalized).
    """
    _validate_mixture_factors(factors, "simplex centroid design")
    q = len(factors) + 1

    rows = []
    for size in range(1, q + 1):
        for subset in combinations(range(q), size):
            row = np.zeros(q, dtype=np.float64)
            row[list(subset)] = 1.0 / size
            rows.append(row)
    design = np.array(rows, dtype=np.float64)
    return _to_dataframe(design[:, :-1], factors)
