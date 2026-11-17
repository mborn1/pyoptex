"""
Module containing all the cost functions of the cost optimal designs
"""

from collections.abc import Callable, Sequence
from functools import partial, wraps

import numba
import numpy as np
import pandas as pd

from ...utils.design import decode_design
from .utils import Factor


def _fn_no_params(f, Y, params):
    return f(Y)


def _fn_decoded(f, Y, params):
    Y = decode_design(Y, params.effect_types, coords=params.coords)
    return f(Y, params)


def _fn_as_dataframe(f, Y, params):
    Y = pd.DataFrame(Y, columns=[str(factor.name) for factor in params.factors])
    return f(Y, params)


def _fn_denormalized(f, Y, params):
    for factor in params.factors:
        Y[str(factor.name)] = factor.denormalize(Y[str(factor.name)])
    return f(Y, params)


def __cost_fn(f, denormalize=True, decoded=True, contains_params=False, dataframe=None):
    """
    Cost function decorator code.

    Parameters
    ----------
    f : func(Y, params) or func(Y)
        The cost function to be wrapped.
    denormalize : bool
        Whether to denormalize (and decode) the data before passing it to `f`.
    decoded : bool
        Whether to only decode, but not denormalize the data before passing it to `f`.
    contains_params : bool
        Whether the cost function requires the CODEX
        :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`.
        This prevents numba compilation.
    dataframe : bool or None, default None
        Whether to convert the design to a pandas DataFrame before passing it to `f`.
        If None, this defaults to True when `denormalize=True` or `decoded=True`
        and False otherwise. Ignored when denormalize=True: denormalized inputs
        are always passed as a DataFrame.

    Returns
    -------
    fn : func(Y, params)
        The decorated function
    """
    # Denormalized inputs are always a DataFrame
    if denormalize and dataframe is False:
        dataframe = True

    if dataframe is None:
        dataframe = denormalize or decoded

    # Encoded categoricals span several columns, so column names cannot be assigned
    if dataframe and not (denormalize or decoded):
        raise ValueError(
            "dataframe=True requires decoded=True or denormalize=True: "
            "encoded designs have no one-column-per-factor layout."
        )

    # Check if parameters are required (prevents direct use of numba.njit compilation)
    if not contains_params:
        f = wraps(f)(partial(_fn_no_params, f))

    # The last wrapper added runs first: decode, then DataFrame, then denormalize
    if denormalize:
        f = wraps(f)(partial(_fn_denormalized, f))

    if dataframe:
        f = wraps(f)(partial(_fn_as_dataframe, f))

    if denormalize or decoded:
        f = wraps(f)(partial(_fn_decoded, f))

    if denormalize:
        NOTE = "This cost function works on denormalized inputs"

    elif decoded:
        NOTE = "This cost function works on decoded categorical inputs"

    else:
        NOTE = "This cost function works on normalized (encoded) inputs"

    if f.__doc__ is not None:
        params_pos = f.__doc__.find("    Parameters\n    ---------")
        f.__doc__ = f.__doc__[:params_pos] + f"\n    .. note::\n        {NOTE}\n\n" + f.__doc__[params_pos:]

    return f


def cost_fn(*args, **kwargs):
    """
    Parameters
    ----------
    f : func(Y, params) or func(Y)
        The cost function to be wrapped.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        A list of factors for the design.
    denormalize : bool
        Whether to denormalize (and decode) the data before passing it to `f`.
    decoded : bool
        Whether to only decode, but not denormalize the data before passing it to `f`.
    contains_params : bool
        Whether the cost function requires the
        :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`.
        This prevents numba compilation.
    dataframe : bool or None, default None
        Whether to convert the design to a pandas DataFrame before passing it to `f`.
        If None, this defaults to True when `denormalize=True` or `decoded=True`
        and False otherwise. Ignored when denormalize=True: denormalized inputs
        are always passed as a DataFrame.

    Returns
    -------
    fn : func(Y, params)
        The decorated function
    """
    if len(args) > 0 and callable(args[0]):
        return __cost_fn(args[0], *args[1:], **kwargs)
    else:

        def wrapper(f):
            return __cost_fn(f, *args, **kwargs)

        return wrapper


############################################################


def _combined(Y, params, costs):
    return [c for cf in costs for c in cf(Y, params)]


def combine_costs(costs):
    """
    Combine multiple cost functions together.

    Parameters
    ----------
    costs : iterable(func)
        An iterable of cost functions to concatenate

    Returns
    -------
    cost_fn : func(Y, params)
        The combined cost function for the simulation algorithm.
    """
    # A module-level function with a partial (not a closure) can be sent to worker processes
    costs = tuple(costs)
    _cost = wraps(_combined)(partial(_combined, costs=costs))

    # pylint: disable=line-too-long
    _cost.__doc__ = "This is a combined cost function of:\n* " + "\n* ".join(cf.__name__ for cf in costs)

    return _cost


def discount_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = max(c1, c2, ..., base).
    This means the total transition cost is determined by the most-hard-to-change factor.

    Parameters
    ----------
    costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    base_cost : float
        The base cost when no factors are changed, i.e., when a run
        is repeated.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Expand the costs to categorically encoded
    costs = np.array(
        [
            c
            for f in factors
            for c in ([costs[str(f.name)]] if f.is_continuous else [costs[str(f.name)]] * (len(f.levels) - 1))
        ]
    )

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        """Internal cost function according to
        :py:function:`discount_cost <pyoptex.doe.cost_optimal.cost.discount_cost>`"""
        # Initialize costs
        cc = np.zeros(len(Y))
        cc[0] = base_cost

        # Loop for each cost
        for i in range(1, len(Y)):
            # Extract runs
            old_run = Y[i - 1]
            new_run = Y[i]

            # Detect change in runs
            c = 0
            for j in range(old_run.size):
                if old_run[j] != new_run[j] and costs[j] > c:
                    c = costs[j]

            # Set base cost
            c = max(c, base_cost)

            # Set the cost
            cc[i] = c

        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)


def parallel_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    multiple workers can work on the transition between
    two consecutive runs in parallel. The total transition
    cost is determined by the most-hard-to-change factor.

    Parameters
    ----------
    transition_costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return discount_cost(
        {k: v + execution_cost for k, v in transition_costs.items()}, factors, max_cost, execution_cost
    )


def additive_cost(costs, factors, max_cost, base_cost=1):
    """
    Create a transition cost function according to the formula C = c1 + c2 + ... + base.
    This means that every factor is independently, and sequentially changed.

    Parameters
    ----------
    costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    base_cost : float
        The base cost when no factors are changed, i.e., when a run
        is repeated.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    costs = np.array([costs[str(f.name)] for f in factors])

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = base_cost

        for i in range(1, len(Y)):
            # Base cost of a run
            tc = base_cost

            # Define the old / new run for transition
            old_run = Y[i - 1]
            new_run = Y[i]

            # Additive costs
            for j in range(colstart.size - 1):
                if np.any(old_run[colstart[j] : colstart[j + 1]] != new_run[colstart[j] : colstart[j + 1]]):
                    tc += costs[j]

            cc[i] = tc

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)


def single_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition
    cost is determined by the sum of all transition costs.

    Parameters
    ----------
    transition_costs : dict(str, float)
        A dictionary mapping the factor name to the transition cost.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return additive_cost(
        {k: v + execution_cost for k, v in transition_costs.items()}, factors, max_cost, execution_cost
    )


def scaled_parallel_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    multiple workers can work on the transition between
    two consecutive runs in parallel. The total transition
    cost is determined by the most-hard-to-change factor.
    The transition cost is determined by scaling the
    transition cost between start and stop with a base cost.
    See the parameters for more information.

    Parameters
    ----------
    transition_costs : dict(str, tuple(float, float, float, float) or float)
        A dictionary mapping the factor name to the transition cost.
        The cost is a tuple with as first element the base cost of any
        positive transition (-1 to +1), as second element the base cost
        of any negative transition (+1 to -1), as third element the
        additional cost to positively scale between min (-1) and max (+1),
        and as third element the additional cost to negatively scale between
        max (+1) and min (-1).
        Categorical factors should have only a float indicating the base cost
        of any transition.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Validate the categorical factors
    assert all(isinstance(transition_costs[str(f.name)], (int, float)) for f in factors if f.is_categorical), (
        "Categorical factors must only have a single float or integer representing the base transition cost of any transition"
    )
    # Validate continuous factors
    assert all(len(transition_costs[str(f.name)]) == 4 for f in factors if f.is_continuous), (
        "Continuous variables must specify (base positive, base negative, scaling positive, scaling negative)"
    )

    # Restructure transition costs
    transition_costs = {
        str(f.name): (
            transition_costs[str(f.name)]
            if f.is_continuous
            else (transition_costs[str(f.name)], transition_costs[str(f.name)], 0, 0)
        )
        for f in factors
    }

    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    is_continuous = np.array([f.is_continuous for f in factors])

    # Expand the costs to arrays
    base_costs = np.array([[transition_costs[str(f.name)][0], transition_costs[str(f.name)][1]] for f in factors])
    scale_costs = np.array(
        [[transition_costs[str(f.name)][2] / 2, transition_costs[str(f.name)][3] / 2] for f in factors]
    )

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = execution_cost

        for i in range(1, len(Y)):
            # Define the old / new run for transition
            old_run = Y[i - 1]
            new_run = Y[i]

            # Additive costs
            cc_ = np.zeros(colstart.size - 1)
            for j in range(colstart.size - 1):
                # Check for a transition
                if np.any(old_run[colstart[j] : colstart[j + 1]] != new_run[colstart[j] : colstart[j + 1]]):
                    # Check if continuous or categorical factor
                    if is_continuous[j]:
                        diff = new_run[colstart[j]] - old_run[colstart[j]]
                        if diff > 0:
                            # Positive transition
                            cc_[j] = base_costs[j][0] + scale_costs[j][0] * diff
                        else:
                            # Negative transition
                            cc_[j] = base_costs[j][1] - scale_costs[j][1] * diff
                    else:
                        # Categorical base cost
                        cc_[j] = base_costs[j][0]

            # Take the maximum as most-hard-to-change
            cc[i] = np.max(cc_) + execution_cost

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)


def scaled_single_worker_cost(transition_costs, factors, max_cost, execution_cost=1):
    """
    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition
    cost is determined by the sum of all transition costs.
    The transition cost is determined by scaling the
    transition cost between start and stop with a base cost.
    See the parameters for more information.

    Parameters
    ----------
    transition_costs : dict(str, tuple(float, float, float, float) or float)
        A dictionary mapping the factor name to the transition cost.
        The cost is a tuple with as first element the base cost of any
        positive transition (-1 to +1), as second element the base cost
        of any negative transition (+1 to -1), as third element the
        additional cost to positively scale between min (-1) and max (+1),
        and as third element the additional cost to negatively scale between
        max (+1) and min (-1).
        Categorical factors should have only a float indicating the base cost
        of any transition.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        the execution cost of a run.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Validate the categorical factors
    assert all(isinstance(transition_costs[str(f.name)], (int, float)) for f in factors if f.is_categorical), (
        "Categorical factors must only have a single float or integer representing the base transition cost of any transition"
    )
    # Validate continuous factors
    assert all(len(transition_costs[str(f.name)]) == 4 for f in factors if f.is_continuous), (
        "Continuous variables must specify (base positive, base negative, scaling positive, scaling negative)"
    )

    # Restructure transition costs
    transition_costs = {
        str(f.name): (
            transition_costs[str(f.name)]
            if f.is_continuous
            else (transition_costs[str(f.name)], transition_costs[str(f.name)], 0, 0)
        )
        for f in factors
    }

    # Compute the column starts
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))
    is_continuous = np.array([f.is_continuous for f in factors])

    # Expand the costs to arrays
    base_costs = np.array([[transition_costs[str(f.name)][0], transition_costs[str(f.name)][1]] for f in factors])
    scale_costs = np.array(
        [[transition_costs[str(f.name)][2] / 2, transition_costs[str(f.name)][3] / 2] for f in factors]
    )

    # Define the transition costs
    @numba.njit
    def _cost(Y):
        # Initialize the costs
        cc = np.zeros(len(Y))
        cc[0] = execution_cost

        for i in range(1, len(Y)):
            # Define the old / new run for transition
            old_run = Y[i - 1]
            new_run = Y[i]

            # Additive costs
            cc_ = np.zeros(colstart.size - 1)
            for j in range(colstart.size - 1):
                # Check for a transition
                if np.any(old_run[colstart[j] : colstart[j + 1]] != new_run[colstart[j] : colstart[j + 1]]):
                    # Check if continuous or categorical factor
                    if is_continuous[j]:
                        diff = new_run[colstart[j]] - old_run[colstart[j]]
                        if diff > 0:
                            # Positive transition
                            cc_[j] = base_costs[j][0] + scale_costs[j][0] * diff
                        else:
                            # Negative transition
                            cc_[j] = base_costs[j][1] - scale_costs[j][1] * diff
                    else:
                        # Categorical base cost
                        cc_[j] = base_costs[j][0]

            # Take the maximum as most-hard-to-change
            cc[i] = np.sum(cc_) + execution_cost

        # Return the costs
        return [(cc, max_cost, np.arange(len(Y)))]

    return cost_fn(_cost, denormalize=False, decoded=False, contains_params=False)


def _level_cost(
    transition_costs: dict[str, float | np.ndarray],
    factors: Sequence[Factor],
    max_cost: float,
    execution_cost: float,
    level_costs: dict[str, np.ndarray] | None,
    parallel: bool,
) -> Callable:
    """
    Create a transition cost function from tables with a cost for every
    pair of levels, per factor. The cost of a run is

        C = execution_cost + l1 + l2 + ... + max(t1, t2, ...)   (parallel)
        C = execution_cost + l1 + l2 + ... + t1 + t2 + ...      (single worker)

    with lf the level cost of factor f at its level in this run, and tf
    the table's cost of changing factor f from its level in the previous
    run to its level in this run (zero for the first run).
    Shared by :py:func:`level_parallel_worker_cost <pyoptex.doe.cost_optimal.cost.level_parallel_worker_cost>`
    and :py:func:`level_single_worker_cost <pyoptex.doe.cost_optimal.cost.level_single_worker_cost>`;
    see those for the other parameters.

    Parameters
    ----------
    parallel : bool
        Whether the change between two runs costs the most expensive
        factor change (parallel workers) or the sum of the factor
        changes (a single worker).

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    level_costs = {} if level_costs is None else level_costs
    names = [str(f.name) for f in factors]
    for argument, costs in (("transition_costs", transition_costs), ("level_costs", level_costs)):
        unknown = [name for name in costs if name not in names]
        assert len(unknown) == 0, f"The {argument} contain factors {unknown} that are not among the factors {names}"

    # The levels of each factor: categorical factors by index, continuous factors by their normalized coordinates
    nlevels = np.array([len(f.levels) if f.is_categorical else len(f.coords_) for f in factors], dtype=np.int64)
    max_levels = int(np.max(nlevels))
    is_continuous = np.array([f.is_continuous for f in factors])
    coords = np.zeros((len(factors), max_levels), dtype=np.float64)
    transitions = np.zeros((len(factors), max_levels, max_levels), dtype=np.float64)
    run_costs = np.zeros((len(factors), max_levels), dtype=np.float64)
    for i, (f, name, k) in enumerate(zip(factors, names, nlevels, strict=True)):
        if f.is_continuous:
            coords[i, :k] = np.asarray(f.coords_)[:, 0]

        # The cost of changing from level a (row) to level b (column), without the diagonal
        if name in transition_costs:
            t = transition_costs[name]
            t = np.full((k, k), t, dtype=np.float64) if np.ndim(t) == 0 else np.array(t, dtype=np.float64)
            assert t.shape == (k, k), (
                f"The transition costs of factor {name} must be a ({k}, {k}) matrix, one row and column per level, "
                f"but have shape {t.shape}"
            )
            np.fill_diagonal(t, 0)
            assert np.all(np.isfinite(t) & (t >= 0)), (
                f"The transition costs of factor {name} must be non-negative and finite, but are {t.tolist()}"
            )
            transitions[i, :k, :k] = t

        # The extra execution cost of a run at each level
        if name in level_costs:
            c = np.array(level_costs[name], dtype=np.float64)
            assert c.shape == (k,), f"The level costs of factor {name} must have one cost per level ({k}), but are {c}"
            assert np.all(np.isfinite(c) & (c >= 0)), (
                f"The level costs of factor {name} must be non-negative and finite, but are {c.tolist()}"
            )
            run_costs[i, :k] = c

    @numba.njit
    def _cost(Y):
        # The level of every run and factor: the index of a categorical factor, the nearest level of a continuous one
        levels = np.zeros(Y.shape, dtype=np.int64)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if is_continuous[j]:
                    levels[i, j] = np.argmin(np.abs(coords[j, : nlevels[j]] - Y[i, j]))
                else:
                    levels[i, j] = int(Y[i, j])

        # The execution cost and level costs of each run, plus the change from the previous run
        cc = np.full(len(Y), float(execution_cost))
        for i in range(len(Y)):
            change = 0.0
            for j in range(Y.shape[1]):
                cc[i] += run_costs[j, levels[i, j]]
                if i > 0:
                    c = transitions[j, levels[i - 1, j], levels[i, j]]
                    change = max(change, c) if parallel else change + c
            cc[i] += change

        return [(cc, max_cost, np.arange(len(Y)))]

    fn: Callable = cost_fn(_cost, denormalize=False, decoded=True, contains_params=False, dataframe=False)
    return fn


def level_parallel_worker_cost(
    transition_costs: dict[str, float | np.ndarray],
    factors: Sequence[Factor],
    max_cost: float,
    execution_cost: float = 1,
    level_costs: dict[str, np.ndarray] | None = None,
) -> Callable:
    """
    Create a transition cost function for a problem where
    multiple workers can work on the transition between
    two consecutive runs in parallel. The total transition
    cost is the most expensive of the factors' transition costs.

    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition cost is the
    sum of the factors' transition costs.

    Each factor's transition cost is read from a table with a
    cost for every pair of levels (from row to column), so the
    cost of a change can follow any pattern: it need not grow
    with the size of the change, can differ per direction, and
    can differ per pair of categorical levels. Optionally, each
    level also adds its own cost to the execution of a run.
    For a fixed cost per change, see
    :py:func:`single_worker_cost <pyoptex.doe.cost_optimal.cost.single_worker_cost>`;
    for a cost that grows linearly with the size of the change, see
    :py:func:`scaled_single_worker_cost <pyoptex.doe.cost_optimal.cost.scaled_single_worker_cost>`.

    Parameters
    ----------
    transition_costs : dict(str, np.array(2d) or float)
        A dictionary mapping the factor name to its transition costs:
        a (k, k) table for a factor with k levels, where element
        (i, j) is the cost of changing from level i to level j, in
        the order of the factor's levels; the diagonal is ignored.
        A float gives the same cost for every change. Factors that
        are left out cost nothing to change.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        The (base) execution cost of a run.
    level_costs : None or dict(str, np.array(1d))
        A dictionary mapping the factor name to the extra execution
        cost of a run at each of its levels, in the order of the
        factor's levels.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return _level_cost(transition_costs, factors, max_cost, execution_cost, level_costs, parallel=True)


def level_single_worker_cost(
    transition_costs: dict[str, float | np.ndarray],
    factors: Sequence[Factor],
    max_cost: float,
    execution_cost: float = 1,
    level_costs: dict[str, np.ndarray] | None = None,
) -> Callable:
    """
    Create a transition cost function for a problem where
    only a single worker can work on the transition between
    two consecutive runs. The total transition cost is the
    sum of the factors' transition costs.

    Each factor's transition cost is read from a table with a
    cost for every pair of levels (from row to column), so the
    cost of a change can follow any pattern: it need not grow
    with the size of the change, can differ per direction, and
    can differ per pair of categorical levels. Optionally, each
    level also adds its own cost to the execution of a run.
    For a fixed cost per change, see
    :py:func:`single_worker_cost <pyoptex.doe.cost_optimal.cost.single_worker_cost>`;
    for a cost that grows linearly with the size of the change, see
    :py:func:`scaled_single_worker_cost <pyoptex.doe.cost_optimal.cost.scaled_single_worker_cost>`.

    Parameters
    ----------
    transition_costs : dict(str, np.array(2d) or float)
        A dictionary mapping the factor name to its transition costs:
        a (k, k) table for a factor with k levels, where element
        (i, j) is the cost of changing from level i to level j, in
        the order of the factor's levels; the diagonal is ignored.
        A float gives the same cost for every change. Factors that
        are left out cost nothing to change.
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_cost : float
        The budget available for this cost function.
    execution_cost : float
        The (base) execution cost of a run.
    level_costs : None or dict(str, np.array(1d))
        A dictionary mapping the factor name to the extra execution
        cost of a run at each of its levels, in the order of the
        factor's levels.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    return _level_cost(transition_costs, factors, max_cost, execution_cost, level_costs, parallel=False)


def _fixed_runs(Y, max_runs):
    return [(np.ones(len(Y)), max_runs, np.arange(len(Y)))]


def fixed_runs_cost(max_runs):
    """
    Cost function to deal with a fixed maximum number of experiments.
    The maximum cost is supposed to be the number of runs, and this cost function
    simply returns 1 for each run.

    Parameters
    ----------
    max_runs : int
        The maximum number of runs.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # A module-level function with a partial (not a closure) can be sent to worker processes
    _cost_fn = wraps(_fixed_runs)(partial(_fixed_runs, max_runs=max_runs))
    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)


def max_changes_cost(factor, factors, max_changes):
    """
    Cost function to deal with a fixed maximum number of changes in a specific factor.
    The maximum cost is supposed to be the number of changes, and this cost function
    simply returns 1 for each change.

    .. note::
        It does not account for the initial setup and final result

    Parameters
    ----------
    factor : str or int
        The name or index of the factor
    factors : list(:py:class:`Factor <pyoptex.doe.cost_optimal.utils.Factor>`)
        The factors for the design.
    max_changes : int
        The maximum number of changes in the specified factor.

    Returns
    -------
    cost_fn : func(Y, params)
        The cost function.
    """
    # Expand factor for categorical variables
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    colstart = np.concatenate(([0], np.cumsum(np.where(effect_types == 1, 1, effect_types - 1))))

    # Determine the columns of the factor
    if isinstance(factor, str):
        factor = [str(f.name) for f in factors].index(factor)
    factor = slice(colstart[factor], colstart[factor + 1])

    # Create cost function
    def _cost_fn(Y):
        changes = np.zeros(len(Y))
        changes[1:] = np.any(np.diff(Y[:, factor], axis=0), axis=1).astype(int)
        return [(changes, max_changes, np.arange(len(Y)))]

    return cost_fn(_cost_fn, denormalize=False, decoded=False, contains_params=False)
