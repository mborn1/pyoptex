import numpy as np
from sklearn.base import BaseEstimator

from ..mixins.fit_mixin import RegressionMixin
from ..mixins.conditional_mixin import ConditionalRegressionMixin
from ...utils.model import identityY2X

class SimpleRegressor(ConditionalRegressionMixin, RegressionMixin, BaseEstimator):
    """
    A simple linear regressor implementing the
    :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>` and
    :py:class:`ConditionalRegressionMixin <pyoptex.analysis.mixins.conditional_mixin.ConditionalRegressionMixin>`
    interfaces.

    Permits to fit a simple model provided in Y2X with optionally random effects.

    .. note::
        It includes all parameters and attributes from 
        :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>` and
        :py:class:`ConditionalRegressionMixin <pyoptex.analysis.mixins.conditional_mixin.ConditionalRegressionMixin>`
    """

    def __init__(self, factors=(), Y2X=identityY2X, random_effects=(), conditional=False):
        """
        Simple regression model.

        Parameters
        ----------
        factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
            A list of factors to be used during fitting. It contains
            the categorical encoding, continuous normalization, etc.
        Y2X : func(Y)
            The function to transform a design matrix Y to a model matrix X.
        random_effects : list(str)
            The names of any random effect columns. Every random effect
            is interpreted as a string column and encoded using 
            effect encoding.
        conditional : bool
            Whether to create a conditional model or not.
        """
        super().__init__(
            factors=factors, Y2X=Y2X, random_effects=random_effects, 
            conditional=conditional
        )

    def _fit(self, X, y):
        """
        Internal fit function for the simple regressor.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        # Define the terms
        self.terms_ = np.arange(X.shape[1])

        # Fit the data
        self.fit_ = self.fit_fn_(X, y, self.terms_)

        # Store the final results
        self.coef_ = self.fit_.params[:self.fit_.k_fe]
        self.scale_ = self.fit_.scale
        self.vcomp_ = self.fit_.vcomp
