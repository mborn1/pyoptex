"""
Paley conference and Hadamard matrices over finite fields GF(q), q = p^e for
an odd prime p.

Paley, R. E. A. C. (1933), "On orthogonal matrices", Journal of Mathematics
and Physics 12, pp. 311-320.
"""

from functools import cache
from itertools import product

import numpy as np


def _prime_power(q):
    """Return (p, e) with q = p^e for a prime p and e >= 1, or None."""
    for p in range(2, q + 1):
        if q % p == 0:  # the smallest divisor > 1 is prime
            e = 0
            while q % p == 0:
                q //= p
                e += 1
            return (p, e) if q == 1 else None
    return None


def _poly_mulmod(a, b, low, p):
    """Product of two elements of GF(p)[x] / (x^e + low(x)), as coefficient tuples (constant first)."""
    e = len(low)
    prod = [0] * (2 * e - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                prod[i + j] += ai * bj
    for d in range(2 * e - 2, e - 1, -1):
        c = prod[d] % p
        if c:
            for i, fi in enumerate(low):
                prod[d - e + i] -= c * fi
    return tuple(c % p for c in prod[:e])


@cache
def _galois_field(p, e):
    """
    Elements and multiplication of GF(p^e).

    Elements are coefficient tuples (constant term first), in the order of
    ``itertools.product(range(p), repeat=e)``. Multiplication is modulo the
    first monic polynomial x^e + f_{e-1} x^(e-1) + ... + f_0 that gives a
    field, searching the coefficients (f_{e-1}, ..., f_0) in lexicographic
    order (e.g. x^2 + 1 for GF(9) and x^3 + 2x + 1 for GF(27)). The quotient
    ring is a field exactly when it has no zero divisors.
    """
    elems = list(product(range(p), repeat=e))
    if e == 1:
        return elems, lambda a, b: ((a[0] * b[0]) % p,)

    zero = (0,) * e
    nonzero = elems[1:]
    for high_first in product(range(p), repeat=e):
        low = tuple(reversed(high_first))
        if low[0] == 0:  # x divides the polynomial
            continue
        if all(_poly_mulmod(a, b, low, p) != zero for a in nonzero for b in nonzero):
            return elems, lambda a, b, low=low: _poly_mulmod(a, b, low, p)
    raise ValueError(f"No irreducible polynomial of degree {e} over GF({p})")


def _paley_conference(q):
    """
    Paley conference matrix of order q + 1 over GF(q), for q a power of an odd
    prime: symmetric when q = 1 (mod 4), antisymmetric when q = 3 (mod 4).
    """
    p, e = _prime_power(q)
    assert p > 2, f"The Paley construction needs a power of an odd prime (got q = {q})"
    elems, mul = _galois_field(p, e)
    squares = {mul(x, x) for x in elems[1:]}

    def chi(x):
        """Quadratic character of x in GF(q)."""
        if not any(x):
            return 0
        return 1 if x in squares else -1

    C = np.zeros((q + 1, q + 1), dtype=np.float64)
    C[0, 1:] = 1
    C[1:, 0] = 1 if q % 4 == 1 else -1
    C[1:, 1:] = [[chi(tuple((xi - yi) % p for xi, yi in zip(x, y, strict=True))) for y in elems] for x in elems]
    return C


def _paley_hadamard(q):
    """
    Paley Hadamard matrix for q a power of an odd prime: of order q + 1 when
    q = 3 (mod 4) (Paley I, ``I + S`` with S the antisymmetric conference
    matrix), and of order 2(q + 1) when q = 1 (mod 4) (Paley II,
    ``[[C + I, C - I], [C - I, -C - I]]`` with C the symmetric one).
    """
    C = _paley_conference(q)
    eye = np.eye(q + 1)
    if q % 4 == 3:
        return C + eye
    return np.block([[C + eye, C - eye], [C - eye, -C - eye]])
