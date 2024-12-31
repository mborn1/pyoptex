import numpy as np
import pandas as pd
from functools import cached_property
from numba.typed import List
from sklearn.utils.validation import (
    check_X_y, check_is_fitted
)
from sklearn.base import RegressorMixin as RegressorMixinSklearn

from ..utils import identity, fit_ols, fit_mixedlm
from ...utils.design import encode_design, obs_var_from_Zs
from ...utils.model import encode_model, model2encnames

class RegressionMixin(RegressorMixinSklearn):
    """
    
    Requires:
    * coef_ : The regression coefficients.
    * terms_ : The indices of the terms.
    * scale_ : The scale factor of the covariance matrix.
    * vcomp_ : The variance components of any random effects.
    """

    def __init__(self, factors=(), Y2X=identity, random_effects=()):
        # Store the parameters
        self.factors = factors
        self.Y2X = Y2X
        self.re = random_effects

    def _regr_params(self, X, y):
        self._factors = list(self.factors)
        self._re = list(self.re)
        self._Y2X = self.Y2X

    def _compute_derived(self):
        # Compute derived parameters from the inputs
        self.n_features_in_ = len(self._factors)
        self.features_names_in_ = [str(f.name) for f in self._factors]
        self.effect_types_ = np.array([
            1 if f.is_continuous else len(f.levels) 
            for f in self._factors
        ])
        self.coords_ = List([f.coords_ for f in self._factors])


    def _validate_X(self, X):
        assert isinstance(X, pd.DataFrame), f'X must be a dataframe'
        assert all(c in X.columns for c in self.features_names_in_), f'X does not have the correct features'
        for f in self._factors:
            if f.is_categorical:
                assert all(l in f.levels for l in X[str(f.name)].unique()), f'X contains a categorical level not specified in the factor, unable to encode'

    def _preprocess_X(self, X):
        # Normalize the factors
        for f in self._factors:
            X[str(f.name)] = f.normalize(X[str(f.name)])

        # Select correct order + to numpy
        X = X[self.features_names_in_].to_numpy()

        # Encode
        X = encode_design(X, self.effect_types_, coords=self.coords_)

        # Transform
        X = self._Y2X(X)

        return X


    def _validate_fit(self, X, y):
        # Validate init parameters
        assert len(self._factors) > 0, f'Must have at least one factor'

        # Validate inputs
        self._validate_X(X)
        q = 'Did you forget the random effects?' if X.shape[1] == self.n_features_in_ else ''
        assert X.shape[1] == self.n_features_in_ + len(self._re), f'X does not have the correct number of features: {self.n_features_in_ + len(self.re)} vs. {X.shape[1]}. {q}'
        assert all(c in X.columns for c in self._re), f'X does not have the correct random effects'

    def preprocess_fit(self, X, y):
        # Normalize y
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        assert self.y_std_ > 0, f'y is a constant vector, cannot do regression'
        y = (y - self.y_mean_) / (self.y_std_)
        y = np.asarray(y)

        # Define the fit function
        if len(self._re) == 0:
            # Define OLS fit
            self.fit_fn_ = lambda X, y, terms: fit_ols(X[:, terms], y)
            self.Zs_ = np.empty((0, len(X)), dtype=np.int_)

        else:
            # Create list from the random effects
            re = list(self._re)

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

        # Set the number of encoded features
        self.n_encoded_features_ = X.shape[1]

        return X, y

    def _fit(self, X, y):
        raise NotImplementedError('The fit function has not been implemented')

    def fit(self, X, y):
        # Adjust the regression parameters
        self._regr_params(X, y)

        # Compute derived parameters
        self._compute_derived()

        # Validate input X and y
        self._validate_fit(X, y)

        # Preprocess the fitting
        X, y = self.preprocess_fit(X, y)

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
        X = X.drop(columns=list(self._re), errors='ignore')

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
        """
        The observation covariance matrix :math:`V = var(Y)`.

        .. math::

            V = \sigma_{\epsilon}^2 I_N + \sum_{i=1}^k \sigma_{\gamma_i}^2 Z_i Z_i^T

        When no random effects are specified, this reduces to a scaled
        identity matrix.
        """
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
        """
        The inverse of the observation covariance matrix.
        See
        :py:func:`obs_cov <pyoptex.analysis.mixins.fit_mixin.obs_cov>`
        for more information.
        """
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
        """
        The information matrix of the fitted data.

        .. math::

            M = X^T V^{-1} X

        where :math:`X` is the normalized, encoded data, and 
        :math:`V` the observation covariance matrix
        (:py:func:`obs_cov <pyoptex.analysis.mixins.fit_mixin.obs_cov>`).
        When no random effects are specified, this reduces to
        :math:`M = X^T X`.
        """
        # Compute observation covariance matrix
        if len(self._re) > 0:
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
        """
        The inverse of the information matrix. See
        :py:func:`information_matrix <pyoptex.analysis.mixins.fit_mixin.information_matrix>`
        for more information.
        """
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
        """
        The total variance on the normalized y-values.
        Includes both the scale and the variance components of the
        random effects.
        """
        return self.scale_ + np.sum(self.vcomp_)

    ##################################################

    def _pred_var(self, X):
        """
        Prediction variances from the values specified in X
        """
        # Compute base prediction variance
        pv = np.sum((X @ self.inv_information_matrix) * X, axis=1) # X @ Minv @ X.T

        # Additional variance from random error and random effects
        # during a new prediction
        pv += self.total_var

        # Account for y-scaling
        pv *= self.y_std_ * self.y_std_

        return pv

    def pred_var(self, X):
        X = self.preprocess_predict(X)
        return self._pred_var(X)

    def model_formula(self, model):
        # Make sure model is a dataframe
        assert isinstance(model, pd.DataFrame), 'The specified model must be a dataframe'

        # Encode the labels
        labels = model2encnames(
            model, 
            np.array([1 if f.is_continuous else len(f.levels) for f in self._factors])
        )

        return self.formula(labels)

    def formula(self, labels=None):
        """
        Creates the prediction formula for the fit for the encoded and
        normalized data. The labels for each term are given by the 
        `labels` parameter.
        The number of labels must be the number of parameters from Y2X,
        i.e., len(labels) == Y2X(Y).shape[1].

        .. warning::
            This formula is the prediction formula of the encoded and
            normalized data. First apply factor normalization
            and then categorical encoding before applying this
            prediction formula.

            >>> # Imports
            >>> from numba.typed import List
            >>> from pyoptex.utils import Factor
            >>> from pyoptex.utils.design import encode_design
            >>> 
            >>> # Example factors
            >>> factors = [
            >>>     Factor('A'), 
            >>>     Factor('B'),
            >>>     Factor('C', type='categorical', levels=['L1', 'L2', 'L3'])
            >>> ]
            >>> 
            >>> # Compute derived parameters
            >>> effect_types = np.array([
            >>>     1 if f.is_continuous else len(f.levels)
            >>>     for f in factors
            >>> ])
            >>> coords = List([f.coords_ for f in factors])
            >>> 
            >>> # Normalize the factors
            >>> for f in factors:
            >>>     data[str(f.name)] = f.normalize(data[str(f.name)])
            >>> 
            >>> # Select correct order + to numpy
            >>> data = data[[str(f.name) for f in factors]].to_numpy()
            >>> 
            >>> # Encode
            >>> data = encode_design(data, effect_types, coords=coords)
            >>> 
            >>> # Transform according to the model
            >>> data = Y2X(data)

            
        .. note::
            If you created Y2X using
            :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`,
            use
            :py:func:`model_formula <pyoptex.analysis.mixins.fit_mixin.model_formula`.
            It will automatically assign the correct labels.

        Parameters
        ----------
        labels : list(str)
            The list of labels for each encoded, normalized term.

        Returns
        -------
        formula : str
            The prediction formula for encoded and normalized data.
        """
        
        if labels is None:
            # Specify default x features
            labels = [f'x{i}' for i in range(self.n_encoded_features_)]

        # Validate the labels
        assert len(labels) == self.n_encoded_features_, 'Must specify one label per encoded feature (= Y2X(Y).shape[1])'

        # Create the formula
        formula = ' + '.join(f'{c:.3f} * {labels[t]}' for c, t in zip(self.coef_, self.terms_)) 

        return formula   

    def summary(self):
        if hasattr(self, 'fit_'):
            return self.fit_.summary()
        else:
            raise AttributeError('Must have a fit_ object to print a fit summary')
