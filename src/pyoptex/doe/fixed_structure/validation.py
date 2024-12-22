"""
Module for all validation functions of the generic coordinate-exchange algorithm
"""

import warnings

import numpy as np


def validate_state(state, params, eps=1e-6):
    """
    Validates that the provided state is correct.

    Parameters
    ----------
    state : :py:class:`State <pyoptex.doe.fixed_structure.utils.State>`
        The state to validate.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    eps : float
        The epsilon to use for floating point comparison.
    """
    # Validate X
    assert np.all(state.X == params.fn.Y2X(state.Y)), '(validation) X does not match Y2X(Y)'

    # Validate metric
    metric = params.fn.metric.call(state.Y, state.X, params)
    if (metric == 0 and state.metric == 0) \
        or (np.isnan(metric) and np.isnan(state.metric))\
        or (np.isinf(metric) and np.isinf(state.metric)):
        warnings.warn(f'Metric is {state.metric}')
    else:
        assert np.abs((state.metric - metric) / metric) < eps, f'(validation) The metric does not match: {state.metric}, {metric}'

    # Validate constraints
    constraints = params.fn.constraints(state.Y)
    assert not np.any(constraints), f'(validation) Constraints are violated: {constraints}'
