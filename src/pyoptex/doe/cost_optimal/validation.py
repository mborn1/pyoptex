import numpy as np
import warnings

from .utils import obs_var_Zs
from ..utils.design import obs_var_from_Zs


def validate_state(state, params, eps=1e-6):
    """
    Validates the state to see if it is still correct to within a precision of epsilon.
    Mostly used to validate intermediate steps of the algorithm and debugging purposes.
    """

    # Make sure all runs are possible
    assert not np.any(params.fn.constraints(state.Y))

    # Make sure the prior is at the start
    assert np.all(state.Y[:len(params.prior)] == params.prior)

    # Validate X
    X = params.Y2X(state.Y)
    assert np.all(state.X == X), f'{state.X - X}'

    # Validate Zs
    Zs = obs_var_Zs(state.Y, params.colstart, params.grouped_cols)
    assert all((Zs[i] is None and state.Zs[i] is None) or np.all(Zs[i] == state.Zs[i]) for i in range(len(Zs)))

    # Make sure every set of ratios has a Vinv attached
    assert params.ratios.shape[0] == len(state.Vinv)

    # Validate Vinv
    for i in range(len(state.Vinv)):
        vinv = np.linalg.inv(obs_var_from_Zs(state.Zs, len(state.Y), params.ratios[i]))
        assert np.all(np.abs(state.Vinv[i] - vinv) < eps), f'{np.linalg.norm(state.Vinv[i] - vinv)}'

    # Validate costs
    costs = params.fn.cost(state.Y)
    assert np.all(state.costs == costs), f'{state.costs - costs}'

    # Validate cost_Y
    cost_Y = np.sum(state.costs, axis=1)
    assert np.all(state.cost_Y == cost_Y), f'{state.cost_Y} -- {cost_Y}'

    # Validate metric
    metric = params.fn.metric.call(state.Y, state.X, state.Zs, state.Vinv, state.costs)
    if (metric == 0 and state.metric == 0) \
        or (np.isnan(metric) and np.isnan(state.metric))\
        or (np.isinf(metric) and np.isinf(state.metric)):
        warnings.warn(f'Metric is {state.metric}')
    else:
        assert np.abs((state.metric - metric) / metric) < eps, f'{state.metric} -- {metric}'
