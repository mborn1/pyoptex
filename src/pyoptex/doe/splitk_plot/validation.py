"""
Module for all validation functions of the split^k-plot algorithm
"""

import warnings

import numpy as np


def validate_state(state, params, eps=1e-6):
    """
    Validates that the provided state is correct.

    Parameters
    ----------
    state : :py:class:`pyoptex.doe.splitk_plot.utils.State`
        The state to validate.
    params : :py:class:`pyoptex.doe.splitk_plot.utils.Parameters`
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

# def validate_UD(U, D, Xi_star, runs, unstable_state, params):
#     # Extract copies
#     Ynew = np.copy(unstable_state.Y)
#     X = np.copy(unstable_state.X)

#     # Compute V and its inverse
#     V = np.array([obs_var(params.plot_sizes, ratio) for ratio in params.ratios])
#     Vinv = np.linalg.inv(V)

#     # Create current information matrix
#     M = X.T @ Vinv @ X

#     # Set new X values
#     X[runs] = Xi_star

#     # Validate updates with new M
#     for i in range(len(M)):
#         Mstar = M[i] + U.T @ np.diag(D[i]) @ U
#         assert np.linalg.norm(Mstar - X.T @ Vinv[i] @ X) < 1e-6
