"""
Module for classical mixture design generation (simplex lattice and
simplex centroid designs, Scheffé 1958, 1963), with lower bounds through
L-pseudocomponents and optional augmentation with check blends (Cornell,
J. A. (2002), "Experiments with Mixtures", 3rd ed., Wiley).

Following the pyoptex convention for mixtures, a mixture of q components
is specified with q - 1 mixture factors; the last component is implied as
one minus the sum of the others and is not a column of the design.
"""

from itertools import combinations, product

import numpy as np

from .utils import _to_dataframe, _validate_factors


def _pseudocomponent_bounds(factors, last_min, design_name):
    """
    Lower bounds L of all q components and the range R = 1 - sum(L) of the
    L-pseudocomponents, after checking that the constrained region is a simplex.
    """
    _validate_factors(factors)
    for f in factors:
        assert f.is_mixture, f"Factor {f.name} must be a mixture factor for a {design_name}"
    lower = np.array([min(f.levels) for f in factors] + [last_min], dtype=np.float64)
    assert np.all(lower >= 0), f"The lower bounds of the components must be non-negative (got {lower.tolist()})"
    R = 1 - lower.sum()
    assert R > 0, f"The lower bounds of the components must sum to less than 1 (they sum to {lower.sum():g})"

    for f, lo in zip(factors, lower, strict=False):
        implied_max = lo + R
        assert max(f.levels) >= implied_max - 1e-12, (
            f"Mixture factor {f.name} has maximum {max(f.levels):g}, below its implied maximum {implied_max:g} "
            f"(lower bound {lo:g} + 1 - sum of lower bounds {lower.sum():g}), so the region is not a simplex; "
            f"constrained mixtures need an optimal design"
        )
    return lower, R


def _augment(z, q):
    """Add the overall centroid (if absent) and the q axial check blends to a design in pseudocomponents."""
    centroid = np.full((1, q), 1 / q)
    rows = [z]
    if not np.any(np.all(np.isclose(z, centroid), axis=1)):
        rows.append(centroid)
    axial = np.full((q, q), 1 / (2 * q))
    np.fill_diagonal(axial, (q + 1) / (2 * q))
    rows.append(axial)
    return np.vstack(rows)


def _to_mixture_dataframe(z, factors, last_min, augment, design_name):
    """Transform a design in pseudocomponents z to proportions x = L + R z (q - 1 columns)."""
    lower, R = _pseudocomponent_bounds(factors, last_min, design_name)
    if augment:
        z = _augment(z, len(factors) + 1)
    design = lower + R * z
    return _to_dataframe(design[:, :-1], factors)


def simplex_lattice(factors, m, last_min=0.0, augment=False):
    """
    Generate a {q, m} simplex lattice design.

    Every blend whose component proportions are multiples of ``1 / m``
    (0, 1/m, ..., 1) and sum to 1: ``C(q + m - 1, m)`` runs for q
    components.  It supports a Scheffé polynomial of degree *m*.

    With lower bounds L_i on the components (the minimum of each
    factor's levels, and *last_min* for the implied last component), the
    design is built in L-pseudocomponents and transformed to
    ``x_i = L_i + (1 - sum(L)) z_i``, so that each component ranges from
    ``L_i`` to ``L_i + 1 - sum(L)``.  The region is then still a simplex,
    provided no factor has a maximum below that implied maximum; other
    constrained regions need an optimal design.

    Parameters
    ----------
    factors : list(Factor)
        The first q - 1 mixture components (``type='mixture'``).  The
        last component is implied as one minus the sum of the others and
        is not part of the design.
    m : int
        Lattice degree: the number of equal parts each proportion is
        divided into.
    last_min : float
        Lower bound of the implied last component (default 0).
    augment : bool
        Whether to add the overall centroid (if it is not already a run)
        and the q axial check blends, with one component at
        ``(q + 1) / (2q)`` and the others at ``1 / (2q)`` in
        pseudocomponents, to check the fit of the model inside the
        simplex (Cornell, 2002).  Default False.

    Returns
    -------
    Y : pd.DataFrame
        The simplex lattice design with q - 1 columns (denormalized).
    """
    assert m >= 1, f"m must be at least 1 (got {m})"
    q = len(factors) + 1

    parts = [p for p in product(range(m + 1), repeat=q) if sum(p) == m]
    z = np.array(sorted(parts, reverse=True), dtype=np.float64) / m
    return _to_mixture_dataframe(z, factors, last_min, augment, "simplex lattice design")


def simplex_centroid(factors, last_min=0.0, augment=False):
    """
    Generate a simplex centroid design.

    Every equal-proportion blend of 1 to q components: the q pure
    components, the ``C(q, 2)`` binary 50/50 blends, and so on up to the
    overall centroid, ``2^q - 1`` runs in total.  It supports Scheffé's
    special polynomial with all products of distinct components (e.g.
    the special cubic model for q = 3), which has exactly ``2^q - 1``
    terms.

    With lower bounds L_i on the components (the minimum of each
    factor's levels, and *last_min* for the implied last component), the
    design is built in L-pseudocomponents and transformed to
    ``x_i = L_i + (1 - sum(L)) z_i``, so that each component ranges from
    ``L_i`` to ``L_i + 1 - sum(L)``.  The region is then still a simplex,
    provided no factor has a maximum below that implied maximum; other
    constrained regions need an optimal design.

    Parameters
    ----------
    factors : list(Factor)
        The first q - 1 mixture components (``type='mixture'``).  The
        last component is implied as one minus the sum of the others and
        is not part of the design.
    last_min : float
        Lower bound of the implied last component (default 0).
    augment : bool
        Whether to add the q axial check blends, with one component at
        ``(q + 1) / (2q)`` and the others at ``1 / (2q)`` in
        pseudocomponents, to check the fit of the model inside the
        simplex (Cornell, 2002), giving ``2^q - 1 + q`` runs.  Default
        False.

    Returns
    -------
    Y : pd.DataFrame
        The simplex centroid design with q - 1 columns (denormalized).
    """
    q = len(factors) + 1

    rows = []
    for size in range(1, q + 1):
        for subset in combinations(range(q), size):
            row = np.zeros(q, dtype=np.float64)
            row[list(subset)] = 1.0 / size
            rows.append(row)
    z = np.array(rows, dtype=np.float64)
    return _to_mixture_dataframe(z, factors, last_min, augment, "simplex centroid design")
