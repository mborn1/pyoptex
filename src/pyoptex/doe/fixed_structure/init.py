"""
Module for all init functions of fixed structure.
"""

from math import prod

import numpy as np

from ..._profile import profile
from ...utils.design import encode_design
from ..utils.init import full_factorial, init_single_unconstrained
from ._init_cy import __correct_constraints, __init_unconstrained


def check_candidate_runs(colstart, coords, constraints, Y2X, prior=None, max_full=200_000, n_random=50_000):
    """
    Checks, once per design and before any random start, that the constraints
    leave candidate runs, and that those runs can estimate the model. Used by
    the fixed-structure and split-plot routes, whose initializers sample runs
    per random start and would otherwise retry forever.

    The candidate runs are all level combinations of the coordinates when there
    are at most `max_full`, otherwise `n_random` random level combinations
    (drawn with a separate generator, so numpy's global random state, and thus
    the designs, are unchanged). The rank of the model is only checked when all
    level combinations were checked.

    Parameters
    ----------
    colstart : np.array(1d)
        The starting column of each factor in the encoded design matrix.
    coords : list(np.array(2d))
        The encoded coordinates (candidate levels) of each factor.
    constraints : func(Y)
        The constraints on the encoded design matrix, returning True for
        the runs that are excluded.
    Y2X : func(Y)
        The function converting the encoded design matrix to the model matrix.
    prior : np.array(2d) or None
        The encoded prior runs, included in the rank check.
    max_full : int
        The maximum number of level combinations to check them all.
    n_random : int
        The number of random level combinations to check otherwise.

    Raises
    ------
    ValueError
        If no run satisfies the constraints, or if the runs allowed by the
        constraints (with the prior) cannot estimate all model terms.
    """
    # Candidate runs
    exhaustive = prod(c.shape[0] for c in coords) <= max_full
    if exhaustive:
        Y = full_factorial(colstart, coords)
    else:
        rng = np.random.default_rng(0)
        Y = np.empty((n_random, colstart[-1]), dtype=np.float64)
        for i, c in enumerate(coords):
            Y[:, colstart[i] : colstart[i + 1]] = c[rng.integers(c.shape[0], size=n_random)]

    # Drop the runs the constraints exclude
    allowed = Y[~np.asarray(constraints(Y), dtype=np.bool_)]
    if hasattr(constraints, "clear"):
        constraints.clear()  # Clear to permit pickling for multiprocessing
    if len(allowed) == 0:
        raise ValueError(
            f"No run satisfies the constraints (checked {len(Y)} level combinations). "
            "Check the constraints and the factor levels."
        )

    # Check that the allowed runs (with the prior) can estimate the model
    if exhaustive:
        if prior is not None and len(prior) > 0:
            allowed = np.concatenate((prior, allowed), axis=0)
        X = Y2X(allowed)

        # The rank of X[:, :i] equals the rank of R[:, :i] of its QR decomposition (small)
        R = np.linalg.qr(X, mode="r")
        s = np.linalg.svd(R, compute_uv=False)
        tol = s.max(initial=0) * max(X.shape) * np.finfo(np.float64).eps
        if np.sum(s > tol) < X.shape[1]:
            # The first column (counted from 1) that depends on the earlier ones
            i = next(i for i in range(1, X.shape[1] + 1) if np.linalg.matrix_rank(R[:, :i], tol=tol) < i)
            raise ValueError(
                f"The runs allowed by the constraints cannot estimate all model terms (model column {i} depends on "
                "earlier columns). Relax the constraints or simplify the model."
            )


@profile
def initialize_feasible(params, complete=False, max_tries=1000):
    """
    Generates a random initial design for a generic design.
    `grps` specifies at each level which level-groups should be
    initialized. This is useful when augmenting an existing design.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.
    max_tries : int
        The maximum number of tries to generate a feasible design.

    Returns
    -------
    Y : np.array(2d)
        The generated design.
    enc : tuple(np.array(2d), np.array(2d))
        The categorical factor encoded Y and X respectively.
    """
    # Compute design sizes
    ncol = params.effect_types.shape[0]

    # Initiate design matrix
    Y = params.prior
    if Y is None:
        Y = np.zeros((params.nruns, ncol), dtype=np.float64)

    feasible = False
    tries = 0
    while not feasible:
        # Add one try
        tries += 1

        # Initialize unconstrained
        Y = __init_unconstrained(
            params.effect_types, params.effect_levels, params.grps, params.coords, params.Zs, Y, complete
        )

        # Constraint corrections
        Y = __correct_constraints(
            params.effect_types,
            params.effect_levels,
            params.grps,
            params.coords,
            params.fn.constraintso,
            params.Zs,
            Y,
            complete,
        )

        # Encode the design
        Yenc = encode_design(Y, params.effect_types, params.coords)

        # Make sure it's feasible
        Xenc = params.fn.Y2X(Yenc)
        feasible = np.linalg.matrix_rank(Xenc) >= Xenc.shape[1]

        # Check if not in infinite loop
        if tries >= max_tries and not feasible:
            # Determine which column causes rank deficiency
            for i in range(1, Xenc.shape[1] + 1):
                if np.linalg.matrix_rank(Xenc[:, :i]) < i:
                    break

            # pylint: disable=line-too-long
            raise ValueError(
                f"Unable to find a feasible design due to the model: component {i} causes rank collinearity with all prior components (note that these are categorically encoded)"
            )

    return Y, (Yenc, Xenc)


def init_random(params, n=1, complete=False):
    """
    Initialize a design with `n` randomly sampled runs. They must
    be within the constraints.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    n : int
        The number of runs
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.

    Returns
    -------
    design : np.array(2d)
        The resulting design.
    """
    # Initialize
    run = np.zeros((n, params.colstart[-1]), dtype=np.float64)
    invalid = np.ones(n, dtype=np.bool_)

    # Adjust for completeness
    if complete:
        # Must still use the correct categorical encoding
        coords = [None if f.is_continuous else params.coords[i] for i, f in enumerate(params.factors)]
    else:
        coords = params.coords

    # Loop until all are valid
    while np.any(invalid):
        run[invalid] = init_single_unconstrained(params.colstart, coords, run[invalid], params.effect_types)
        invalid[invalid] = params.fn.constraints(run[invalid])

    return run
