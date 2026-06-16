"""
Module for fractional factorial design generation (L^(k-p)).
"""

from itertools import product

import numpy as np

from .utils import _to_dataframe, _validate_factors


def fractional_factorial(factors, generators, n_center=0):
    """
    Generate a fractional factorial design.

    The *base* factors (those **not** named as keys in *generators*)
    form a full factorial.  Each generated factor's column is computed
    from its defining base-factor level indices.

    For two-level factors the result matches the classical generator
    notation exactly: the generated column equals the element-wise
    **product** of the +/-1 coded defining columns (e.g. ``D = ABC``
    means ``D = A * B * C``).

    For factors with ``L > 2`` levels the generated column is the sum of
    the defining factors' level indices modulo ``L`` (the standard
    construction for regular multi-level fractional factorials).

    The number of levels per factor is determined by ``Factor.coords_``
    (default 3 for continuous factors without explicit ``levels``,
    ``len(levels)`` otherwise).

    Parameters
    ----------
    factors : list(Factor)
        The factors.  Every base factor referenced in a generator must
        have the same number of levels as the generated factor.
    generators : dict(str, list(str))
        Maps each generated factor name to the list of base-factor
        names whose level indices are summed (mod L) to define it.
        E.g. ``{"D": ["A", "B", "C"]}`` means
        ``D_index = (A_index + B_index + C_index) % L``.
    n_center : int
        Number of center points to append (continuous factors at
        mid-range, categorical factors at first level).

    Returns
    -------
    Y : pd.DataFrame
        The fractional factorial design (denormalized).
    """
    _validate_factors(factors)
    name2idx = {str(f.name): i for i, f in enumerate(factors)}
    k = len(factors)

    nlevels = np.array([len(f.coords_) for f in factors], dtype=np.int64)

    for gen_name in generators:
        assert gen_name in name2idx, f"Generator factor '{gen_name}' is not in the factor list"

    base_names = [str(f.name) for f in factors if str(f.name) not in generators]
    assert len(base_names) > 0, "At least one base factor is required"

    defining_sets = {}
    for gen_name, defining in generators.items():
        assert len(defining) > 0, f"Generator for '{gen_name}' must reference at least one base factor"
        key = tuple(sorted(defining))
        assert key not in defining_sets, (
            f"Generators '{defining_sets[key]}' and '{gen_name}' share the same defining factors "
            f"{list(defining)}; this produces perfectly aliased duplicate columns"
        )
        defining_sets[key] = gen_name

    base_ranges = [range(int(nlevels[name2idx[n]])) for n in base_names]
    base_design = np.array(list(product(*base_ranges)), dtype=np.int64)
    nruns = base_design.shape[0]

    design_idx = np.zeros((nruns, k), dtype=np.int64)
    for j, bname in enumerate(base_names):
        design_idx[:, name2idx[bname]] = base_design[:, j]

    for gen_name, defining in generators.items():
        gen_col = name2idx[gen_name]
        L = int(nlevels[gen_col])
        col = np.zeros(nruns, dtype=np.int64)
        for dname in defining:
            assert dname in name2idx, f"Defining factor '{dname}' not found in factor list"
            assert dname in base_names, f"Defining factor '{dname}' must be a base factor (not a generator)"
            d_idx = name2idx[dname]
            assert nlevels[d_idx] == L, (
                f"Base factor '{dname}' has {nlevels[d_idx]} levels but generated "
                f"factor '{gen_name}' has {L} levels; they must match"
            )
            col = col + design_idx[:, d_idx]
        # For two-level factors the classical generator (e.g. C = AB) is the
        # element-wise product of the +/-1 coded columns. In level-index space
        # {0, 1} the product corresponds to the index sum shifted by (m - 1),
        # which restores the correct sign for an even number of defining factors.
        # For L > 2 levels the standard construction is the index sum modulo L.
        col = (col + len(defining) - 1) % 2 if L == 2 else col % L
        design_idx[:, gen_col] = col

    design = np.zeros((nruns, k), dtype=np.float64)
    for j, f in enumerate(factors):
        if f.is_continuous:
            design[:, j] = f.coords_[design_idx[:, j], 0]
        else:
            design[:, j] = design_idx[:, j].astype(np.float64)

    if n_center > 0:
        center = np.zeros((n_center, k), dtype=np.float64)
        design = np.vstack([design, center])

    return _to_dataframe(design, factors)
