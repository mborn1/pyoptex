"""
Module for definitive screening design (DSD) generation.

Design from Jones & Nachtsheim (2011), "A class of three-level designs for
definitive screening in the presence of second-order effects", Journal of
Quality Technology 43(1), pp. 1-15, with the fake factors of Jones &
Nachtsheim (2013), "Definitive screening designs with added two-level
categorical factors", Journal of Quality Technology 45(2), pp. 121-129.
"""

import numpy as np

from .utils import _to_dataframe, _validate_factors


def _prime_power(q):
    """Return (p, e) with q = p^e for an odd prime p and e in (1, 2), or None."""
    for e in (1, 2):
        p = round(q ** (1 / e))
        if p**e == q and p > 2 and all(p % d for d in range(2, int(p**0.5) + 1)):
            return p, e
    return None


def _paley_conference(q):
    """
    Paley conference matrix of order q + 1 over GF(q), for q an odd prime or
    the square of one. Symmetric when q = 1 (mod 4), antisymmetric when
    q = 3 (mod 4).

    GF(p^2) is represented as pairs (a, b) = a + b*t with t^2 = r, for r a
    quadratic non-residue modulo p.
    """
    p, e = _prime_power(q)
    if e == 1:
        r = 0
        elems = [(a, 0) for a in range(p)]
    else:
        r = next(x for x in range(2, p) if pow(x, (p - 1) // 2, p) == p - 1)
        elems = [(a, b) for a in range(p) for b in range(p)]

    squares = {((a * a + r * b * b) % p, (2 * a * b) % p) for a, b in elems if (a, b) != (0, 0)}

    def chi(x):
        """Quadratic character of x in GF(q)."""
        if x == (0, 0):
            return 0
        return 1 if x in squares else -1

    C = np.zeros((q + 1, q + 1), dtype=np.float64)
    C[0, 1:] = 1
    C[1:, 0] = 1 if q % 4 == 1 else -1
    C[1:, 1:] = [[chi(((x[0] - y[0]) % p, (x[1] - y[1]) % p)) for y in elems] for x in elems]
    return C


def _conference(n):
    """
    Conference matrix C of order n (zero diagonal, +/-1 elsewhere,
    C'C = (n - 1) I), or None when no construction is implemented.

    Uses the Paley construction when n - 1 is an odd prime or the square of
    one, and otherwise doubles an antisymmetric conference matrix S of order
    n / 2 to ``[[S, S + I], [S - I, -S]]``, which is again antisymmetric.
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
    matrices are built from the Paley construction (m - 1 an odd prime
    or the square of one) or by doubling a smaller antisymmetric one,
    which covers every order up to 50 except 22 and 34 (no conference
    matrix exists) and 28, 36 and 46 (not implemented).  For those the
    next available order is used, which adds runs.

    Every factor takes the fixed levels ``min``, mid-range and ``max``;
    explicit factor ``levels`` are not used.

    Parameters
    ----------
    factors : list(Factor)
        Continuous factors (at least 3).
    n_fake : int
        Number of fake factors: the design is built for ``k + n_fake``
        factors and the extra columns are dropped.  Jones & Nachtsheim
        (2013) recommend this to add runs, which increases the power to
        detect second-order effects.  Default 0, the original design.
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
