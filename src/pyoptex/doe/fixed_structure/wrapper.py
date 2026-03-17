"""
Module for the interface to run the generic coordinate-exchange algorithm
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...utils.design import decode_design, obs_var_from_Zs
from ..constraints import mixture_constraints, no_constraints
from .init import initialize_feasible
from .optimize import optimize
from .utils import Factor, FunctionSet, Parameters, RandomEffect, State


def default_fn(factors, metric, Y2X, constraints=None, init=initialize_feasible):
    """
    Create a functionset with the default operators. Each
    operator can be manually overriden by providing the parameter.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.fixed_stucture.utils.Factor>`)
        The factors of the experiment.
    metric : :py:class:`Metric <pyoptex.doe.fixed_structure.metric.Metric>`
        The metric object.
    Y2X : func
        The function converting from the design matrix to the
        model matrix.
    constraints : func
        The constraints function,
        :py:func:`no_constraints <pyoptex.doe.constraints.no_constraints>`
        by default.
    init : func
        The initialization function,
        :py:func:`initialize_feasible <pyoptex.doe.fixed_structure.init.initialize_feasible>`
        by default.

    Returns
    -------
    fn : :py:class:`FunctionSet <pyoptex.doe.fixed_structure.utils.FunctionSet>`
        The function set.
    """

    # Check if factors contain mixtures
    if any(f.is_mixture for f in factors):
        # Create the mixture constraints
        mix_constr = mixture_constraints([str(f.name) for f in factors if f.is_mixture], factors)

        # Add the mixture constraints
        constraints = mix_constr if constraints is None else (constraints | mix_constr)

    # Default to no constraints
    if constraints is None:
        constraints = no_constraints

    return FunctionSet(metric, Y2X, constraints.encode(), constraints.func(), init)


def create_parameters(factors, fn, nruns, block_effects=(), prior=None, grps=None):
    """
    Creates the parameters object by preprocessing the inputs.
    This is a utility function to transform each variable
    to its correct representation.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.doe.fixed_structure.utils.Factor>`)
        The list of factors.
    fn : :py:class:`FunctionSet <pyoptex.doe.fixed_structure.utils.FunctionSet>`
        A set of operators for the algorithm.
    nruns : int
        Total number of runs (prior + new when augmenting).
    block_effects : list(:py:class:`RandomEffect <pyoptex.doe.fixed_structure.utils.RandomEffect>`)
        Any additional blocking effects, not assigned to a factor.
    prior : None or pd.DataFrame
        Optional prior design for augmentation. Must be denormalized and decoded;
        rows correspond to the first ``len(prior)`` rows of the new design.
    grps : None or list(np.ndarray or None)
        Optional list of additional group indices to optimize per factor (on top of
        automatically determined new groups when prior is given).

    Returns
    -------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.
    """
    # Assertions
    assert len(factors) > 0, "At least one factor must be provided"
    for i, f in enumerate(factors):
        assert isinstance(f, Factor), f"Factor {i} is not of type Factor"
        assert f.re is None or isinstance(f.re, RandomEffect), (
            f"Factor {i} with name {f.name} does not have a RandomEffect as random effect"
        )
        if f.re is not None:
            assert len(f.re.Z) == nruns, f"Factor {i} with name {f.name} does not have enough runs as random effect"
    if prior is not None:
        assert isinstance(prior, pd.DataFrame), f"The prior must be specified as a dataframe but is a {type(prior)}"
    for i, be in enumerate(block_effects):
        assert len(be.Z) == nruns, (
            f"Blocking effect {i} does not have the correct length: {len(be.Z)}. Should be the number of runs {nruns}"
        )

    nblocks = len(block_effects)

    # Extract the random effects
    re = []
    for f in factors:
        if f.re is not None and f.re not in re:
            re.append(f.re)

    # Extract the plot sizes
    ratios = []
    for r in re + list(block_effects):
        # Extract ratios
        r = np.sort(r.ratio) if isinstance(r.ratio, (tuple, list, np.ndarray)) else [r.ratio]

        # Append the ratios
        ratios.append(r)

    # Align ratios
    if len(ratios) > 0:
        nratios = max([len(r) for r in ratios])
        assert all(len(r) == 1 or len(r) == nratios for r in ratios), (
            "All ratios must be either a single number or and array of the same size"
        )
        ratios = np.array(
            [np.repeat(ratio, nratios) if len(ratio) == 1 else ratio for ratio in ratios], dtype=np.float64
        ).T

        # Split regular and blocking ratios
        if nblocks == 0:
            be_ratios = np.empty_like(ratios, shape=(0, ratios.shape[1]))
        else:
            be_ratios = ratios[:, -len(block_effects) :]
            ratios = ratios[:, : len(block_effects)]
    else:
        # No blocking ratios
        be_ratios = []

    # Extract parameter arrays
    col_names = [str(f.name) for f in factors]
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors], dtype=np.int64)
    effect_levels = np.array([re.index(f.re) + 1 if f.re is not None else 0 for f in factors], dtype=np.int64)
    coords = [f.coords_ for f in factors]

    # Encode the coordinates
    colstart = np.concatenate(
        ([0], np.cumsum(np.where(effect_types == 1, effect_types, effect_types - 1))), dtype=np.int64
    )

    # Compute Zs and Vinv
    if len(re) > 0:
        # Zs = np.array([np.unique(np.array(r.Z, dtype=np.int64), return_inverse=True)[1] for r in re])
        Zs = np.array([np.array(r.Z, dtype=np.int64) for r in re])
        V = np.array([obs_var_from_Zs(Zs, N=nruns, ratios=r) for r in ratios], dtype=np.float64)
    else:
        Zs = np.empty((0, 0), dtype=np.int64)
        V = np.expand_dims(np.eye(nruns, dtype=np.float64), 0)

    # Augment V with the random blocking effects
    if len(block_effects) > 0:
        # beZs = np.array([np.unique(np.array(be.Z, dtype=np.int64), return_inverse=True)[1] for be in block_effects])
        beZs = np.array([np.array(be.Z, dtype=np.int64) for be in block_effects])
        V += np.array(
            [obs_var_from_Zs(beZs, N=nruns, ratios=r, include_error=False) for r in be_ratios], dtype=np.float64
        )

    # Invert V
    Vinv = np.linalg.inv(V)

    # Handle prior design for augmentation
    n_old = 0
    if prior is not None:
        # Extract the old prior number of runs
        n_old = len(prior)

        # Normalize and convert to numpy
        for f in factors:
            prior[str(f.name)] = f.normalize(prior[str(f.name)])
        prior = prior[col_names].to_numpy()

        # Don't encode the design
        # prior = encode_design(prior, effect_types, coords=coords)

        # Validate HTC structure: same Z-group must have same factor value in prior
        re_vals = [None] * len(factors)
        if len(re) > 0:
            for i in range(len(factors)):
                # Retrieve the random effect values for this factor
                level = effect_levels[i]

                # Check if HTC
                if level > 0:
                    # Retrieve the unique Z-groups for this factor
                    Z = Zs[level - 1][: len(prior)]
                    prior_groups = np.unique(Z)

                    # Initialize the random effect values for this factor (needed later)
                    re_vals[i] = [None] * len(prior_groups)

                    # Check the HTC structure: same Z-group must have same factor value in prior
                    for grp in prior_groups:
                        # Find the runs in the prior that belong to this Z-group
                        mask = Z == grp
                        col_vals = prior[mask, i]

                        # Assert they are all the same
                        assert np.all(col_vals == col_vals[0]), (
                            f"Prior is not consistent for factor {f.name}: same group must have same value"
                        )

                        # Store the random effect value for this Z-group and factor combination
                        re_vals[i][grp] = col_vals[0]

        # Validate constraints
        assert not np.any(fn.constraintso(prior)), "Prior contains constraint-violating runs"
        fn.constraintso.clear()  # Clear to permit pickling for multiprocessing

        # Build full design: prior runs first, then zeros for new runs
        prior = np.concatenate((prior, np.zeros((nruns - len(prior), len(factors)), dtype=np.float64)), axis=0)

        # Propagate HTC values to new runs that share a Z-group with prior runs
        if len(re) > 0:
            for i in range(len(factors)):
                # Retrieve the random effect values for this factor
                level = effect_levels[i]

                # Check if HTC
                if level > 0:
                    # Retrieve the Z-groups for this factor
                    Z = Zs[level - 1]

                    # Loop over the new runs and assign the random effect value
                    for r in range(n_old, nruns):
                        if Z[r] < len(re_vals[i]):
                            prior[r, i] = re_vals[i][Z[r]]

        # Make the prior contiguous for cython
        prior = np.ascontiguousarray(prior)

    # Define which groups to optimize in case no other groups are provided
    if prior is not None:
        # New groups only: level 0 = new run indices; level > 0 = groups that appear only in new runs
        lgrps = [np.arange(n_old, nruns, dtype=np.int64)]
        for k in range(Zs.shape[0]):
            # Retrieve the unique Z-groups for this factor and detect which don't appear in the prior
            old_groups = np.unique(Zs[k][:n_old])
            new_groups = np.setdiff1d(np.unique(Zs[k][n_old:]), old_groups)
            lgrps.append(new_groups.astype(np.int64))
    else:
        lgrps = [np.arange(nruns, dtype=np.int64)] + [np.arange(int(np.max(Z)) + 1, dtype=np.int64) for Z in Zs]

    # Define which groups are finally optimized
    if grps is None:
        grps = [lgrps[lvl] for lvl in effect_levels]
    else:
        grps = [
            np.concatenate((np.asarray(grps[i], dtype=np.int64), lgrps[effect_levels[i]]), dtype=np.int64)
            if grps[i] is not None
            else lgrps[effect_levels[i]]
            for i in range(len(effect_levels))
        ]

    # Precompute run indices for each (factor, group) pair
    grp_runs = [None] * len(effect_levels)
    for i in range(len(effect_levels)):
        level = effect_levels[i]
        grp_runs[i] = [None] * len(grps[i])
        for j in range(len(grps[i])):
            if level == 0:
                grp_runs[i][j] = np.array([grps[i][j]], dtype=np.int64)
            else:
                grp_runs[i][j] = np.flatnonzero(Zs[level - 1] == grps[i][j])

    # Create the parameters
    params = Parameters(
        fn, factors, nruns, effect_types, effect_levels, grps, grp_runs, ratios, coords, prior, colstart, Zs, Vinv
    )

    return params


def create_fixed_structure_design(params, n_tries=10, max_it=10000, validate=False):
    """
    Creates an optimal design for the specified factors, using the parameters.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`)
        The simulation parameters.
    n_tries : int
        The number of random start repetitions. Must be larger than zero.
    max_it : int
        The maximum number of iterations per random initialization for the
        coordinate-exchange algorithm. Prevents infinite loop scenario.
    validate : bool
        Whether to validate each state.

    Returns
    -------
    Y : pd.DataFrame
        A pandas dataframe with the best found design. The
        design is decoded and denormalized.
    best_state : :py:class:`State <pyoptex.doe.fixed_structure.utils.State>`
        The state corresponding to the returned design.
        Contains the encoded design, model matrix, metric, etc.
    """
    assert n_tries > 0, "Must specify at least one random initialization (n_tries > 0)"
    assert max_it > 0, "Must specify at least one iteration of the coordinate-exchange per random initialization"

    # Pre initialize metric
    params.fn.metric.preinit(params)

    # Main loop
    best_metric = -np.inf
    best_state = None
    try:
        for _ in tqdm(range(n_tries)):
            # Optimize the design
            Y, state = optimize(params, max_it, validate=validate)

            # Store the results
            if state.metric > best_metric:
                best_metric = state.metric
                best_state = State(np.copy(state.Y), np.copy(state.X), state.metric)
    except KeyboardInterrupt:
        print("Interrupted: returning current results...")

    # Decode the design
    if best_state is not None:
        Y = decode_design(best_state.Y, params.effect_types, coords=params.coords)
        Y = pd.DataFrame(Y, columns=[str(f.name) for f in params.factors])
        for f in params.factors:
            Y[str(f.name)] = f.denormalize(Y[str(f.name)])
    else:
        Y = None

    # Return the design and the final state
    return Y, best_state
