import numpy as np
from sklearn.base import BaseEstimator

from .mixins.fit_mixin import RegressionMixin

class SimpleRegressor(RegressionMixin, BaseEstimator):

    def _fit(self, X, y):
        # Define the terms
        self.terms_ = np.arange(X.shape[1])

        # Fit the data
        fit = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = fit.params[:fit.k_fe]
        self.scale_ = fit.scale
        self.vcomp_ = fit.vcomp
