"""
Module for fractional factorial design generation (L^(k-p)).
"""

from itertools import product

import numpy as np

from ._min_aberration import _MA_GENERATORS
from .utils import (
    _assert_no_categorical_center,
    _block_scheme,
    _blocked_dataframe,
    _center_points,
    _levels,
    _to_dataframe,
    _validate_factors,
)


def _is_prime(n):
    """Whether *n* is a prime number."""
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def _parse_generator(gen_name, defining):
    """
    Convert a generator definition to a dict of integer coefficients.

    The list form is shorthand for coefficients equal to 1, where a
    leading ``"-"`` gives coefficient -1: ``["-A", "B"]`` is ``{"A": -1, "B": 1}``.
    """
    assert len(defining) > 0, f"Generator for '{gen_name}' must reference at least one base factor"
    if isinstance(defining, dict):
        coefs = {}
        for name, c in defining.items():
            assert isinstance(c, (int, np.integer)), (
                f"Coefficient of '{name}' in generator '{gen_name}' must be an integer (got {c!r})"
            )
            coefs[str(name)] = int(c)
        return coefs

    coefs = {}
    for name in defining:
        name = str(name)
        sign = -1 if name.startswith("-") else 1
        name = name.lstrip("-")
        assert name not in coefs, f"Factor '{name}' appears more than once in generator '{gen_name}'"
        coefs[name] = sign
    return coefs


def _alias_key(coefs, L):
    """
    Key identifying the column a generator produces, up to relabeling of
    its levels. Two generators with the same key give perfectly aliased columns.

    For two-level factors the sign does not matter (``D = ABC`` and
    ``D = -ABC`` are aliased). For prime ``L > 2``, coefficient vectors
    that are multiples of each other (mod L) are aliased, so the vector
    is scaled to make its first coefficient 1.
    """
    names = sorted(coefs)
    if L == 2:
        return (L, tuple(names))
    inv = pow(coefs[names[0]] % L, -1, L)
    return (L, tuple((n, coefs[n] * inv % L) for n in names))


def min_aberration_generators(factors, n_runs=None, resolution=None):
    """
    Generators of the minimum aberration two-level fractional factorial.

    Looks up the design in the tables of minimum aberration designs of
    Mee (2009), "A Comprehensive Guide to Factorial Two-Level
    Experimentation", Appendix G, for 8 to 128 runs and up to 31 factors
    (plus the 4-run half fraction C = AB).  Among designs with the same
    number of runs, minimum aberration is the standard criterion: it has
    the highest resolution and, within that, aliases the fewest short
    words (e.g. ``E = ABCD`` for 5 factors in 16 runs).

    Specify either the number of runs, or the minimum resolution, in
    which case the smallest design with at least that resolution is
    returned (like ``fracfact_by_res`` in pyDOE).  The full factorial
    (no generators) counts as a design of unlimited resolution when it
    has at most 128 runs.

    The first ``log2(n_runs)`` factors are the base factors, so put the
    factors in the order you want them assigned.  The result can be
    passed to :py:func:`fractional_factorial` or as the ``cube`` of
    :py:func:`central_composite <pyoptex.doe.classical.central_composite>`.

    Parameters
    ----------
    factors : list(Factor)
        Two-level factors.
    n_runs : int or None
        Number of runs (a power of 2).
    resolution : int or None
        Minimum resolution (at least 3).

    Returns
    -------
    generators : dict(str, list(str))
        Maps each generated factor name to its defining base factors.
    """
    _validate_factors(factors)
    k = len(factors)
    names = [str(f.name) for f in factors]
    for f in factors:
        nlev = len(_levels(f, 2))
        assert nlev == 2, (
            f"Factor {f.name} must have exactly 2 levels for a two-level fractional factorial (has {nlev})"
        )
    assert (n_runs is None) != (resolution is None), "Specify exactly one of n_runs and resolution"

    # Available designs for k factors: (n_runs, resolution, generators), plus the
    # full factorial when it is within the catalogue's range of up to 128 runs
    available = [(n, res, gens) for (kk, n), (res, gens) in _MA_GENERATORS.items() if kk == k]
    if 2**k <= 128:
        available.append((2**k, np.inf, []))
    available.sort(key=lambda d: d[0])

    if n_runs is not None:
        match = [d for d in available if d[0] == n_runs]
        assert len(match) > 0, (
            f"No catalogued fractional factorial for {k} factors in {n_runs} runs; "
            f"available run sizes: {[d[0] for d in available]}"
        )
    else:
        assert resolution >= 3, f"resolution must be at least 3 (got {resolution})"
        match = [d for d in available if d[1] >= resolution]
        assert len(match) > 0, (
            f"No catalogued fractional factorial for {k} factors with resolution {resolution} in at most 128 runs"
        )

    n, _, gens = match[0]

    p = int(np.log2(n))
    return {names[p + i]: [names[j] for j in gen] for i, gen in enumerate(gens)}


def fractional_factorial(factors, generators, n_levels=2, n_center=0, n_blocks=1):
    """
    Generate a regular fractional factorial design.

    The *base* factors (those **not** named as keys in *generators*)
    form a full factorial.  Each generated factor's column is computed
    from the level indices of its defining base factors.

    **Two-level factors.**  The result matches the classical generator
    notation exactly: the generated column equals the element-wise
    **product** of the +/-1 coded defining columns (e.g. ``D = ABC``
    means ``D = A * B * C``).  A negative coefficient flips the sign,
    which selects the other half of the fraction (the fold-over):
    ``{"D": ["-A", "B", "C"]}`` or ``{"D": {"A": -1, "B": 1, "C": 1}}``
    means ``D = -ABC``.

    **Factors with L > 2 levels.**  The generated level index is the
    weighted sum of the defining level indices modulo ``L``, the standard
    construction for regular ``L^(k-p)`` designs (e.g. Wu & Hamada).
    ``{"D": {"A": 1, "B": 2}}`` means ``D = A + 2B (mod L)``, written
    ``AB^2`` in the textbook notation.  Coefficients are taken modulo
    ``L``, so -1 is equivalent to ``L - 1``.  The construction is only
    valid when ``L`` is prime (2, 3, 5, 7, ...), so generated factors
    with a non-prime number of levels are rejected.

    Continuous factors use their explicit ``levels`` when specified,
    otherwise *n_levels* equally spaced levels (default 2).
    Categorical factors use ``len(levels)`` levels.

    Parameters
    ----------
    factors : list(Factor)
        The factors.  Every base factor referenced in a generator must
        have the same number of levels as the generated factor.
    generators : dict(str, list(str) or dict(str, int))
        Maps each generated factor name to its definition: either a
        dict of base-factor names to integer coefficients, or a list
        of base-factor names as shorthand for all coefficients equal
        to 1 (a leading ``"-"`` on a name gives coefficient -1).
        E.g. the L9 orthogonal array is the 3^(4-2) design
        ``{"C": ["A", "B"], "D": {"A": 2, "B": 1}}`` with
        ``n_levels=3``.
    n_levels : int
        Number of levels for continuous factors without explicit
        ``levels``. Use 3 for a 3^(k-p) design.
    n_center : int
        Number of center points to append (default 0, as for a
        screening design), with each factor at the middle of its level
        range; per block in a blocked design.  Not allowed with
        categorical factors, since there is no established convention
        for fractional designs; add such center runs manually.
    n_blocks : int
        Number of blocks (a power of 2), for two-level designs. Among
        the effects that can define the blocks, the scheme confounds the
        highest-order ones possible with the block effect (minimum
        aberration blocking, judged by the shortest effect in each alias
        set). The design gets a ``'Block'`` column numbered from 1, and
        the confounded effects (the shortest one of each alias set) are
        listed in ``Y.attrs['confounded_with_blocks']``. Default 1 (no
        blocks).

    Returns
    -------
    Y : pd.DataFrame
        The fractional factorial design (denormalized).
    """
    _validate_factors(factors)
    name2idx = {str(f.name): i for i, f in enumerate(factors)}
    k = len(factors)

    assert n_levels >= 2, f"n_levels must be at least 2 (got {n_levels})"
    _assert_no_categorical_center(factors, n_center, "fractional factorial")
    levels = [_levels(f, n_levels) for f in factors]
    nlevels = np.array([len(lv) for lv in levels], dtype=np.int64)

    for gen_name in generators:
        assert gen_name in name2idx, f"Generator factor '{gen_name}' is not in the factor list"

    base_names = [str(f.name) for f in factors if str(f.name) not in generators]
    assert len(base_names) > 0, "At least one base factor is required"

    base_ranges = [range(int(nlevels[name2idx[n]])) for n in base_names]
    base_design = np.array(list(product(*base_ranges)), dtype=np.int64)
    nruns = base_design.shape[0]

    design_idx = np.zeros((nruns, k), dtype=np.int64)
    for j, bname in enumerate(base_names):
        design_idx[:, name2idx[bname]] = base_design[:, j]

    alias_keys = {}
    for gen_name, defining in generators.items():
        gen_col = name2idx[gen_name]
        L = int(nlevels[gen_col])
        assert _is_prime(L), (
            f"Generated factor '{gen_name}' has {L} levels; generators require a prime number of levels "
            f"(2, 3, 5, 7, ...) because the mod-L construction is only valid for prime L"
        )

        coefs = _parse_generator(gen_name, defining)
        for dname, c in coefs.items():
            assert dname in name2idx, f"Defining factor '{dname}' not found in factor list"
            assert dname in base_names, f"Defining factor '{dname}' must be a base factor (not a generator)"
            d_idx = name2idx[dname]
            assert nlevels[d_idx] == L, (
                f"Base factor '{dname}' has {nlevels[d_idx]} levels but generated "
                f"factor '{gen_name}' has {L} levels; they must match"
            )
            if L == 2:
                assert c in (1, -1), (
                    f"Coefficient of '{dname}' in generator '{gen_name}' must be 1 or -1 "
                    f"for two-level factors (got {c})"
                )
            else:
                assert c % L != 0, (
                    f"Coefficient of '{dname}' in generator '{gen_name}' is a multiple of {L}, "
                    f"which removes the factor from the generator"
                )

        key = _alias_key(coefs, L)
        assert key not in alias_keys, (
            f"Generators '{alias_keys[key]}' and '{gen_name}' define the same column up to relabeling "
            f"({defining}); this produces perfectly aliased duplicate columns"
        )
        alias_keys[key] = gen_name

        if L == 2:
            # The classical generator (e.g. C = AB) is the element-wise product of
            # the +/-1 coded columns. In level-index space {0, 1} the product
            # corresponds to the index sum shifted by (m - 1), which restores the
            # correct sign for an even number of defining factors. Each negative
            # coefficient flips the sign once more.
            col = sum(design_idx[:, name2idx[d]] for d in coefs)
            col = (col + len(coefs) - 1) % 2
            if sum(c < 0 for c in coefs.values()) % 2 == 1:
                col = 1 - col
        else:
            col = sum(c * design_idx[:, name2idx[d]] for d, c in coefs.items()) % L
        design_idx[:, gen_col] = col

    design = np.zeros((nruns, k), dtype=np.float64)
    for j in range(k):
        design[:, j] = levels[j][design_idx[:, j]]
    center = _center_points(factors, n_levels, n_center)

    assert n_blocks >= 1, f"n_blocks must be at least 1 (got {n_blocks})"
    if n_blocks == 1:
        return _to_dataframe(np.vstack([design, center]), factors)

    for f, nlev in zip(factors, nlevels, strict=True):
        assert nlev == 2, f"Blocking requires two-level factors, but factor {f.name} has {nlev} levels"
    coded = (2 * design_idx - 1).astype(np.int8)
    labels, confounded = _block_scheme(coded, [name2idx[b] for b in base_names], n_blocks)
    return _blocked_dataframe(design, labels, center, factors, confounded)
