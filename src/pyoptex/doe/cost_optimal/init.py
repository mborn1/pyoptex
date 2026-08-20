"""
Module for all init functions of the cost optimal designs
"""
import warnings
from itertools import combinations

import numpy as np
from tqdm import tqdm

from ...utils.design import encode_design
from ..utils.init import full_factorial, init_single_unconstrained


def greedy_cost_minimization(Y, params):
    """
    Greedily minimizes the cost of the design Y.

    Parameters
    ----------
    Y : np.array(2d)
        The design to cost minimize.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    Y : np.array(2d)
        The greedily cost minimized design.
    """
    # Initialization (force prior)
    nprior = len(params.prior)
    Yn = np.zeros_like(Y)
    Yn[:nprior] = params.prior
    chosen = np.zeros(len(Y), dtype=np.bool_)
    chosen[:nprior] = True

    # Iteratively use greedy cost minimization
    for i in range(nprior, len(Y)):
        # Find parameters that are not chosen
        non_chosen = np.where(~chosen)[0]

        # # Initialize all costs
        costs = [None] * non_chosen.size

        # Compute all costs
        for k in range(non_chosen.size):
            Yn[i] = Y[non_chosen[k]]
            costs[k] = params.fn.cost(Yn[: i + 1], params)

        # Compute the total cost of each operation
        costs_Y = np.array([np.sum([np.sum(c) / m * c.size / (i + 1) for c, m, _ in cost]) for cost in costs])
        min_cost_idx = non_chosen[np.argmin(costs_Y)]

        # Chose the index
        Yn[i] = Y[min_cost_idx]
        chosen[min_cost_idx] = True

    return Yn


################################################


def init(params, n=1, complete=False):
    """
    Initialize a design with `n` randomly sampled runs. They must
    be within the constraints.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    n : int
        The number of runs to initialize
    complete : bool
        False means use the coordinates and prior specified in params,
        otherwise, no coords or prior are used.
        Can be used to perform a complete sample of the design space.

    Returns
    -------
    run : np.array(2d)
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
        nprior = len(params.prior)
        run[:nprior] = params.prior
        invalid[:nprior] = False
        coords = params.coords

    # Loop until all are valid
    while np.any(invalid):
        run[invalid] = init_single_unconstrained(params.colstart, coords, run[invalid], params.effect_types)
        invalid[invalid] = params.fn.constraints(run[invalid])

    return run


def init_feasible(params, max_tries=3, max_size=None, force_cost_feasible=True):
    """
    Generate a random initial and feasible design. From a random
    permutation of a full factorial design, the runs are dropped one-by-one
    as long as they still provide a feasible design. Finally, the design
    is greedily reordered for minimal cost.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    max_tries : int
        The maximum number of random tries. If all random tries fail, a
        final non-randomized design is created. If this also fails, a ValueError is thrown.
    max_size : int
        The maximum number of runs before iteratively removing them.
    force_cost_feasible : bool
        Force a final cost feasibility check.

    Returns
    -------
    Y : np.array(2d)
        The initial design.
    """
    # Initialize the tries for randomization
    tries = -1
    reverse = False

    # Check if prior is estimeable
    Xprior = params.fn.Y2X(params.prior)
    if Xprior.shape[0] != 0 and np.linalg.matrix_rank(Xprior) >= Xprior.shape[1]:
        return params.prior
    nprior = len(Xprior)

    feasible = False
    while not feasible:
        # Add one try
        tries += 1

        # Create a full factorial design
        Y = full_factorial(params.colstart, params.coords)

        # Permute to randomize
        if tries < max_tries:
            Y = np.random.permutation(Y)

        # Drop impossible combinations
        Y = Y[~params.fn.constraints(Y)]

        # Define a maximum size (for feasibility)
        if max_size is not None:
            Y = Y[:max_size]

        # Compute X
        X = params.fn.Y2X(Y)

        # Augmentation
        Y = np.concatenate((params.prior, Y), axis=0)
        X = np.concatenate((Xprior, X), axis=0)

        # Drop runs
        keep = np.ones(len(Y), dtype=np.bool_)
        keep[:nprior] = True
        r = range(nprior, len(Y)) if not reverse else range(len(Y) - 1, nprior - 1, -1)
        for i in tqdm(r):
            keep[i] = False
            Xk = X[keep]
            if np.linalg.matrix_rank(Xk) < X.shape[1]:
                keep[i] = True
        Y = Y[keep]

        # Reorder for cost optimization (greedy)
        if tries < max_tries:
            Y = greedy_cost_minimization(Y, params)

        # Fill it up
        X = params.fn.Y2X(Y)
        costs = params.fn.cost(Y, params)
        cost_Y = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])
        feasible = (np.linalg.matrix_rank(X) >= X.shape[1]) and (np.all(cost_Y <= max_cost) or not force_cost_feasible)

        # Raise an error if no feasible design can be found
        if tries >= max_tries and not feasible:
            if reverse:
                # Check if within budget
                if np.all(cost_Y <= max_cost):
                    # Determine which column causes rank deficiency
                    for i in range(1, X.shape[1] + 1):
                        if np.linalg.matrix_rank(X[:, :i]) < i:
                            break

                    # pylint: disable=line-too-long
                    raise ValueError(
                        f"Unable to find a feasible design due to the model: component {i} causes rank collinearity with all prior components (note that these are categorically encoded)"
                    )

                # pylint: disable=line-too-long
                raise ValueError(
                    f"Unable to find a feasible design due to the budget: maximum costs are {max_cost}, design costs are {cost_Y}"
                )
            else:
                reverse = True

    return Y


def init_feasible_(params, max_tries=3, minimal=True, max_size=None, force_cost_feasible=True):
    """
    Generate a random initial and feasible design. From a random
    permutation of a full factorial design, the runs are dropped one-by-one
    as long as they still provide a feasible design. Finally, the design
    is greedily reordered for minimal cost.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    max_tries : int
        The maximum number of random tries. If all random tries fail, a
        final non-randomized design is created. If this also fails, a ValueError is thrown.
    minimal : bool
        Whether to remove runs for a minimal initial design, or only until
        the cost is satisfied.
    max_size : int
        The maximum number of runs before iteratively removing them. This parameter
        can be set when the computational time is too long for the initialization
        in large problems with many factors and factor levels.
    force_cost_feasible : bool
        Permit a design whose cost function is too high if no feasible
        design is found.

    Returns
    -------
    Y : np.array(2d)
        The initial design.
    """
    # Initialize the tries for randomization
    tries = -1

    # Check if prior is estimeable
    Xprior = params.fn.Y2X(params.prior)
    if minimal and Xprior.shape[0] != 0 and np.linalg.matrix_rank(Xprior) >= Xprior.shape[1]:
        return params.prior
    nprior = len(Xprior)

    feasible = False
    while not feasible:
        # Add one try
        tries += 1

        # Create a full factorial design
        Y = full_factorial(params.colstart, params.coords)

        # Permute to randomize
        if tries < max_tries:
            Y = np.random.permutation(Y)

        # Drop impossible combinations
        Y = Y[~params.fn.constraints(Y)]

        # Define a maximum size (for computational feasibility)
        if max_size is not None:
            Y = Y[:max_size]

        # Compute X
        X = params.fn.Y2X(Y)

        # Augmentation
        Y = np.concatenate((params.prior, Y), axis=0)
        X = np.concatenate((Xprior, X), axis=0)

        # Initialize the array of which runs to keep
        keep = np.ones(len(Y), dtype=np.bool_)

        # Check for a minimal or maximal design
        if minimal:
            # Keep dropping terms until no more are droppable
            for i in tqdm(range(nprior, len(Y))):
                # Drop term
                keep[i] = False

                # Test for feasibility
                if np.linalg.matrix_rank(X[keep]) < X.shape[1]:
                    # Keep the term
                    keep[i] = True
            Y = Y[keep]

        else:
            # Keep dropping terms until within the cost constraints
            for i in tqdm(range(nprior, len(Y))):
                # Drop term
                keep[i] = False

                # Test for feasibility
                if np.linalg.matrix_rank(X[keep]) < X.shape[1]:
                    # Keep the term
                    keep[i] = True

                else:
                    # Check if within budget
                    costs = params.fn.cost(Y[keep], params)
                    cost_Y = np.array([np.sum(c) for c, _, _ in costs])
                    max_cost = np.array([m for _, m, _ in costs])
                    if np.all(cost_Y <= max_cost):
                        break
            Y = Y[keep]

        # Reorder for cost optimization (greedy)
        if tries < max_tries:
            Y = greedy_cost_minimization(Y, params)

        # Fill it up
        X = params.fn.Y2X(Y)
        costs = params.fn.cost(Y, params)
        cost_Y = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])
        feasible = (np.linalg.matrix_rank(X) >= X.shape[1]) and (np.all(cost_Y <= max_cost) or not force_cost_feasible)

        # Raise an error if no feasible design can be found
        if tries >= max_tries and not feasible:
            # Check if within budget
            if np.all(cost_Y <= max_cost) or not force_cost_feasible:
                # Determine which column causes rank deficiency
                for i in range(1, X.shape[1] + 1):
                    if np.linalg.matrix_rank(X[:, :i]) < i:
                        break

                # pylint: disable=line-too-long
                raise ValueError(
                    f"Unable to find a feasible design due to the model: component {i} causes rank collinearity with all prior components (note that these are categorically encoded)"
                )

            # pylint: disable=line-too-long
            raise ValueError(
                f"Unable to find a feasible design due to the budget: maximum costs are {max_cost}, design costs are {cost_Y}"
            )

    return Y

def init_warm_feasible(params, df, group_col=None, n_keep=None, max_tries=5000):
    """
    Seed the optimizer from an existing design by keeping a random subset of intact groups.

    Sampling by group (rather than by individual rows) can ensure that hard-to-change factor
    settings remain together. However, because feasibility and estimability are not guaranteed
    by a random draw, the function resamples until the retained subset yields a full-rank 
    model matrix and satisfies all cost constraints. If the maximum number of attempts is 
    reached without finding a fully feasible set, it returns the closest rank-sufficient 
    attempt, letting the caller decide how to proceed.

    Parameters
    ----------
    df : pandas.DataFrame
        The design DataFrame containing the denormalized factor levels, 
        optionally with a grouping column
    group_col : str, optional
        The column name to group rows by (e.g., a batch or day identifier). If None, 
        defaults to the run index.
    n_keep : int, optional
        The number of groups to retain in the warm start subset. If None, defaults 
        to half of the unique groups.
    max_tries : int, default=5000
        The maximum number of random draws to attempt before yielding the best 
        infeasible (but full-rank) subset.

    Returns
    -------
    Y : np.array(2d)
        The warm-started initial design subset.

    Example
    -------
    >>> init_fn = lambda params: init_warm_feasible(params, df, group_col="day", n_keep=3)
    >>> fn = default_fn(..., init=init_fn)
    """
    Ydf = df.copy()
    for f in params.factors:
        Ydf[f.name] = f.normalize(Ydf[f.name])
    factor_names = [f.name for f in params.factors]
    Ydec = Ydf[factor_names].to_numpy()
    Yenc = encode_design(Ydec, params.effect_types, coords=params.coords)

    if group_col is None:
        group = list(Ydf.index)
    else:
        group = Ydf[group_col].to_numpy()

    groups = np.unique(group)
    k = n_keep if n_keep is not None else max(1, len(groups) // 2)
    if k >= len(groups):
        return Yenc                            # nothing to subsample

    pick = list(combinations(groups, n_keep))
    pick = np.random.permutation(pick)

    tries = min(max_tries, pick.shape[0])

    best, best_cost = None, None
    for i in tqdm(range(tries)):
        mask = np.isin(group, pick[i])
        Yk = Yenc[mask]

        costs = params.fn.cost(Yk, params)
        cost_Yk = np.array([np.sum(c) for c, _, _ in costs])
        max_cost = np.array([m for _, m, _ in costs])

        Xk = params.fn.Y2X(Yk)
        rank = np.linalg.matrix_rank(Xk)
        feasible = (rank >= Xk.shape[1]) \
                and (np.all(cost_Yk <= max_cost))

        if feasible:
            return Yk                            # estimable: done
        
        elif rank >= Xk.shape[1] and (best is None or np.all(cost_Yk[cost_Yk > max_cost] < best_cost)):
            best, best_cost = Yk, cost_Yk[cost_Yk > max_cost]


    # FINAL CHECKS
    if best is None:
        raise ValueError(
            f"No {k}-group subset achieved full rank in {max_tries} draws "
            f"(need {Xk.shape[1]} params). Cannot warm start. Try a larger n_keep."
        )
    
    warnings.warn(
        f"no {k}-group subset reached full rank in {max_tries} draws "
        f"(need {Xk.shape[1]} params); returning largest attempt ({best.shape[0]} runs). "
        f"Try a larger n_keep.")
    return best