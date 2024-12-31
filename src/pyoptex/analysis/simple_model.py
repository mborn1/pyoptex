import numpy as np
from sklearn.base import BaseEstimator

from .mixins.fit_mixin import RegressionMixin
from .mixins.conditional_mixin import ConditionalRegressionMixin
from .utils import identity

class SimpleRegressor(ConditionalRegressionMixin, RegressionMixin, BaseEstimator):

    def __init__(self, factors=(), Y2X=identity, random_effects=(), conditional=False):
        super().__init__(
            factors=factors, Y2X=Y2X, random_effects=random_effects, 
            conditional=conditional
        )

    def _fit(self, X, y):
        # Define the terms
        self.terms_ = np.arange(X.shape[1])

        # Fit the data
        self.fit_ = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = self.fit_.params[:self.fit_.k_fe]
        self.scale_ = self.fit_.scale
        self.vcomp_ = self.fit_.vcomp
