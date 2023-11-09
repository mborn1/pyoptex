import numpy as np

from .utils import obs_var
from .formulas import det_update_UD, inv_update_UD_no_P

# TODO: covariances
# TODO: multiple ratios (bayesian)
# TODO: remove first ratio
# TODO: priors and augmentation

# TODO: plot_sizes = lowest first: note in documentation!

# TODO: more metrics
# TODO: validation


class Dopt:
    """
    The D-optimality criterion.
    Computes the geometric mean in case multiple Vinv are provided.
    """
    def __init__(self):
        self.Y2X = None
        self.c = None
        self.Vinv = None
        self.Minv = None

    def init(self, params, Y, X):
        # Store link to the Y2X function
        self.Y2X = params.Y2X

        # Compute information matrix
        # self.Vinv = np.array([obs_var(params.plot_sizes, ratios=c) for c in params.c])
        self.Vinv = params.Vinv
        M = X.T @ params.Vinv @ X
        self.Minv = np.linalg.inv(M)

    def update(self, Y, X, update):
        # Compute change in determinant
        du = 1
        for i in range(len(self.Minv)):
            _du, _ = det_update(update.U[i], update.D[i], self.Minv[i])
            du *= _du

        if du > 1:
            # Update inv(M)
            for i in range(len(self.Minv)):
                self.Minv[i] -= inv_update_UD_no_P(U[i], D[i], self.Minv[i])
            return True
        
        # Return no update
        return False

    def call(self, Y, X):
        M = X.T @ self.Vinv @ X
        return np.power(
            np.product(np.maximum(np.linalg.det(M), 0)), 
            1/(X.shape[1] * len(self.Vinv))
        )
