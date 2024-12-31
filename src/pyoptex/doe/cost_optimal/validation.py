"""
Module for all validation functions of the cost optimal designs.
"""

import warnings

import numpy as np

from ...utils.design import obs_var_from_Zs
from .utils import obs_var_Zs


def validate_state(state, params, eps=1e-6):
    """
    Validates the state to see if it is still correct to within a precision of epsilon.
    Mostly used to validate intermediate steps of the algorithm and debugging purposes.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state from which to start.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    eps : float
        Epsilon for the numerical comparisons.
    """

    # Make sure all runs are possible
    constraints = params.fn.constraints(state.Y)
    assert not np.any(constraints), f'(validation) Constraints of Y are violated: {constraints}'

    # Make sure the prior is at the start
    assert np.all(state.Y[:len(params.prior)] == params.prior), f'(validation) Prior is no longer correct: {state.Y[:len(params.prior)] - params.prior}'

    # Validate X
    X = params.fn.Y2X(state.Y)
    assert np.all(state.X == X), f'(validation) X does not match Y: {state.X - X}'

    # Validate Zs
    Zs = obs_var_Zs(state.Y, params.colstart, params.grouped_cols)
    assert all((Zs[i] is None and state.Zs[i] is None) \
               or np.all(Zs[i] == state.Zs[i]) for i in range(len(Zs))), f'(validation) Grouping matrices Zs are wrong: {Zs}, {state.Zs}'

    # Make sure every set of ratios has a Vinv attached
    assert params.ratios.shape[0] == len(state.Vinv), f'(validation) Number of variance sets does not match the numbe of Vinv matrices: {params.ratios.shape[0]}, {len(state.Vinv)}'

    # Validate Vinv
    for i in range(len(state.Vinv)):
        vinv = np.linalg.inv(obs_var_from_Zs(state.Zs, len(state.Y), params.ratios[i]))
        assert np.all(np.abs(state.Vinv[i] - vinv) < eps), f'(validation) Vinv[{i}] does not match: {np.linalg.norm(state.Vinv[i] - vinv)}'

    # Validate costs
    costs = params.fn.cost(state.Y, params)

    # Validate cost_Y
    cost_Y = np.array([np.sum(c) for c, _, _ in costs])
    assert np.all(state.cost_Y == cost_Y), f'(validation) The total cost does not match: {state.cost_Y}, {cost_Y}'

    # Validate max costs
    max_cost = np.array([m for _, m, _ in costs])
    assert np.all(state.max_cost == max_cost), f'(validation) The maximum cost (budget) does not match: {state.max_cost}, {max_cost}'

    # Validate cost indices
    assert all(np.all(costs[i][2] == state.costs[i][2]) for i in range(len(costs))), f'(validation) The cost indices do not match: {[idx for _, _, idx in state.costs]}, {[idx for _, _, idx in costs]}'

    # Validate metric
    metric = params.fn.metric.call(state.Y, state.X, state.Zs, state.Vinv, state.costs)
    if (metric == 0 and state.metric == 0) \
        or (np.isnan(metric) and np.isnan(state.metric))\
        or (np.isinf(metric) and np.isinf(state.metric))\
        or (np.any(cost_Y > max_cost) and np.isinf(state.metric)):
        warnings.warn(f'Metric is {state.metric}')
    else:
        assert np.abs((state.metric - metric) / metric) < eps, f'(validation) The metric does not match: {state.metric}, {metric}'
