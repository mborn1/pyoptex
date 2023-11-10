import numpy as np

from .utils import obs_var, State

def validate_UD(U, D, Xi_star, runs, unstable_state, params):
    # Extract copies
    Ynew = np.copy(unstable_state.Y)
    X = np.copy(unstable_state.X)

    # Compute V and its inverse
    V = np.array([obs_var(params.plot_sizes, ratio) for ratio in params.ratios])
    Vinv = np.linalg.inv(V)

    # Create current information matrix
    M = X.T @ Vinv @ X

    # Set new X values
    X[runs] = Xi_star

    # Validate updates with new M
    for i in range(len(M)):
        Mstar = M[i] + U.T @ np.diag(D[i]) @ U
        assert np.linalg.norm(Mstar - X.T @ Vinv[i] @ X) < 1e-6
    
def validate_state(state, params):
    # Validate X
    assert np.all(state.X == params.Y2X(state.Y)), 'X does not match Y2X(Y)'

    # Validate metric
    new_metric = params.fn.metric.call(state.Y, state.X, params)
    assert (state.metric - new_metric) / new_metric < 1e-6, f'The metric does not match: {state.metric} -- {new_metric}'
