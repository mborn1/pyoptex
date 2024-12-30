import numpy as np
import pandas as pd
from functools import cached_property
from numba.typed import List
from sklearn.utils.validation import (
    check_X_y, check_is_fitted
)
from sklearn.base import RegressorMixin as RegressorMixinSklearn

from ..utils import identity, fit_ols, fit_mixedlm
from ...doe.utils.design import encode_design, obs_var_from_Zs

class RegressionMixin(RegressorMixinSklearn):
    """
    
    Requires:
    * coef_ : The regression coefficients
    * terms_ : The indices of the terms
    * scale_ : The scale factor of the covariance matrix.
    """

    def __init__(self, factors=(), Y2X=identity, random_effects=()):
        # Store the parameters
        self.factors = factors
        self.Y2X = Y2X
        self.re = random_effects

    def _compute_derived(self):
        # Compute derived parameters from the inputs
        self.n_features_in_ = len(self.factors)
        self.features_names_in_ = [str(f.name) for f in self.factors]
        self.effect_types_ = np.array([
            1 if f.is_continuous else len(f.levels) 
            for f in self.factors
        ])
        self.coords_ = List([f.coords_ for f in self.factors])


    def _validate_X(self, X):
        assert isinstance(X, pd.DataFrame), f'X must be a dataframe'
        assert all(c in X.columns for c in self.features_names_in_), f'X does not have the correct features'
        # TODO: Validate the level of categorical factors

    def _preprocess_X(self, X):
        # Normalize the factors
        for f in self.factors:
            X[str(f.name)] = f.normalize(X[str(f.name)])

        # Select correct order + to numpy
        X = X[self.features_names_in_].to_numpy()

        # Encode
        X = encode_design(X, self.effect_types_, coords=self.coords_)

        # Transform
        X = self.Y2X(X)

        return X


    def _validate_fit(self, X, y):
        # Validate init parameters
        assert len(self.factors) > 0, f'Must have at least one factor'
        # TODO: validate factors
        # TODO: validate random effects

        # Validate inputs
        self._validate_X(X)
        q = 'Did you forget the random effects?' if X.shape[1] == self.n_features_in_ else ''
        assert X.shape[1] == self.n_features_in_ + len(self.re), f'X does not have the correct number of features: {self.n_features_in_ + len(self.re)} vs. {X.shape[1]}. {q}'
        assert all(c in X.columns for c in self.re), f'X does not have the correct random effects'

    def preprocess_fit(self, X, y):
        # Normalize y
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        assert self.y_std_ > 0, f'y is a constant vector, cannot do regression'
        y = (y - self.y_mean_) / (self.y_std_)
        y = np.asarray(y)

        # Define the fit function
        if len(self.re) == 0:
            # Define OLS fit
            self.fit_fn_ = lambda X, y, terms: fit_ols(X[:, terms], y)

        else:
            # Create list from the random effects
            re = list(self.re)

            # Convert them to indices
            for r in re:
                X[r] = X[r].map(
                    {lname: i for i, lname in enumerate(X[r].unique())}
                )

            # Extract and create mixedlm fit function
            self.Zs_ = X[re].to_numpy().T
            self.fit_fn_ = lambda X, y, terms: fit_mixedlm(X[:, terms], y, self.Zs_)
            X = X.drop(columns=re)
        
        # Preprocess X
        X = self._preprocess_X(X)

        return X, y

    def _fit(self, X, y):
        raise NotImplementedError('The fit function has not been implemented')

    def fit(self, X, y):
        # Compute derived parameters
        self._compute_derived()

        # Validate input X and y
        self._validate_fit(X, y)

        # Preprocess the fitting
        X, y = self.preprocess_fit(X, y)
        self.X_ = X
        self.y_ = y

        # Fit the data
        X, y = check_X_y(X, y, accept_sparse=True)
        self._fit(X, y)

        # Mark as fitted
        self.is_fitted_ = True

        return self


    def _validate_predict(self, X):
        # Validate X
        self._validate_X(X)
        assert X.shape[1] == self.n_features_in_, f'X does not have the correct number of features: {self.n_features_in_} vs. {X.shape[1]}'

    def preprocess_predict(self, X):
        # Preprocess X
        X = self._preprocess_X(X)

        return X

    def _predict(self, X):
        # Predict based on linear regression
        return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1) \
                    * self.y_std_ + self.y_mean_

    def predict(self, X):
        # Drop potential remaining random effects
        X = X.drop(columns=list(self.re), errors='ignore')

        # Validate this model has been fitted
        check_is_fitted(self, 'is_fitted_')
        self._validate_predict(X)

        # Preprocess the input
        X = self.preprocess_predict(X)

        # Predict
        return self._predict(X)

    ##################################################

    @cached_property
    def obs_cov(self):
        return obs_var_from_Zs(
            self.Zs_, len(self.X_), self.vcomp_ / self.scale_
        ) * self.scale_
    
    @property
    def V_(self):
        """
        Alias for 
        :py:func:`obs_cov <pyoptex.analysis.mixins.fit_mixin.obs_cov>`
        """
        return self.obs_cov

    @cached_property
    def inv_obs_cov(self):
        return np.linalg.inv(self.obs_cov)
    
    @property
    def Vinv_(self):
        """
        Alias for 
        :py:func:`inv_obs_cov <pyoptex.analysis.mixins.fit_mixin.inv_obs_cov>`
        """
        return self.inv_obs_cov
        
    @cached_property
    def information_matrix(self):
        # Compute observation covariance matrix
        if len(self.re) > 0:
            M = self.X_.T @ np.linalg.solve(self.obs_cov, self.X_)
        else:
            M = (self.X_.T @ self.X_) / self.scale

        return M

    @property
    def M_(self):
        """
        Alias for 
        :py:func:`information_matrix <pyoptex.analysis.mixins.fit_mixin.information_matrix>`
        """
        return self.information_matrix
    
    @cached_property
    def inv_information_matrix(self):
        return np.linalg.inv(self.information_matrix)
    
    @property
    def Minv_(self):
        """
        Alias for 
        :py:func:`inv_information_matrix <pyoptex.analysis.mixins.fit_mixin.inv_information_matrix>`
        """
        return self.inv_information_matrix
    
    @property
    def total_var(self):
        return self.scale_ + np.sum(self.vcomp_)

    def pred_var(self, X):
        # Compute base prediction variance
        pv = np.sum((X @ self.inv_information_matrix) * X, axis=1) # X @ Minv @ X.T

        # Additional variance from random error and random effects
        # during a new prediction
        pv += self.total_var

        # Account for y-scaling
        pv *= self.y_std_ * self.y_std_

        return pv
