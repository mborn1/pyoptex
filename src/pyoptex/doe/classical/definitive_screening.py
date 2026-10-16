"""
Module for definitive screening design (DSD) generation.

Design from Jones & Nachtsheim (2011), "A class of three-level designs for
definitive screening in the presence of second-order effects", Journal of
Quality Technology 43(1), pp. 1-15, with the fake factors (extra runs)
recommended by Jones, B. & Nachtsheim, C. J. (2017), "Effective design-based
model selection for definitive screening designs", Technometrics 59(3),
pp. 319-329, doi:10.1080/00401706.2016.1234979.
"""

import numpy as np

from ._paley import _paley_conference, _prime_power
from .utils import _to_dataframe, _validate_factors


def _conference(n):
    """
    Conference matrix C of order n (zero diagonal, +/-1 elsewhere,
    C'C = (n - 1) I), or None when no construction is implemented.

    Uses the Paley construction when n - 1 is a power of an odd prime, and
    otherwise doubles an antisymmetric conference matrix S of order n / 2 to
    ``[[S, S + I], [S - I, -S]]``, which is again antisymmetric.
    """
    if n >= 4 and _prime_power(n - 1) is not None:
        return _paley_conference(n - 1)
    if n % 4 == 0 and n >= 8:
        S = _conference(n // 2)
        if S is not None and np.array_equal(S, -S.T):
            eye = np.eye(n // 2)
            return np.block([[S, S + eye], [S - eye, -S]])
    return None


def definitive_screening(factors, n_fake=0, n_center=1):
    """
    Generate a definitive screening design (DSD).

    The design is ``[C; -C; 0]`` for a conference matrix ``C`` of even
    order m: 2m + 1 runs at three levels, in which

    * main effects are orthogonal to each other,
    * main effects are orthogonal to all two-factor interactions and
      quadratic effects (because of the fold-over),
    * no two-factor interactions are fully aliased with each other.

    With an even number of factors k the order is m = k, with an odd
    number m = k + 1 and the last column is dropped.  Conference
    matrices are built from the Paley construction (m - 1 a power of an
    odd prime, e.g. 27 for order 28) or by doubling a smaller
    antisymmetric one, which covers every order up to 50 except 22 and
    34 (no conference matrix exists) and 36 and 46 (not implemented).
    For those the next available order is used, which adds runs.

    Every factor takes the fixed levels ``min``, mid-range and ``max``;
    explicit factor ``levels`` are not used.  Categorical factors are not
    supported; for those an optimal design is the route.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 3).
    n_fake : int
        Number of fake factors: the design is built for ``k + n_fake``
        factors and the extra columns are dropped, which adds runs.
        Jones & Nachtsheim (2017) recommend using 2 fake factors
        routinely (4 extra runs, e.g. 17 instead of 13 runs for 6
        factors), which improves model selection for the second-order
        effects.  Default 0, the original design of Jones & Nachtsheim
        (2011).
    n_center : int
        Number of center runs (default 1, the single center run of the
        original design).

    Returns
    -------
    Y : pd.DataFrame
        The definitive screening design (denormalized).
    """
    _validate_factors(factors)
    k = len(factors)
    assert k >= 3, f"Definitive screening designs require at least 3 factors (got {k})"
    assert n_fake >= 0, f"n_fake must be non-negative (got {n_fake})"
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"

    for f in factors:
        assert f.is_continuous, f"Factor {f.name} must be continuous for a definitive screening design"

    m = k + n_fake
    m += m % 2
    C = _conference(m)
    while C is None:
        m += 2
        C = _conference(m)

    C = C[:, :k]
    design = np.vstack([C, -C, np.zeros((n_center, k), dtype=np.float64)])
    return _to_dataframe(design, factors)
