import numpy as np
import pandas as pd
from functools import cached_property
from numba.typed import List
from sklearn.utils.validation import check_X_y
from sklearn.base import RegressorMixin as RegressorMixinSklearn

from ..utils.fit import fit_ols, fit_mixedlm
from ...utils.design import encode_design, obs_var_from_Zs
from ...utils.model import model2encnames, identityY2X

class RegressionMixin(RegressorMixinSklearn):
    """
    Base mixin for all regressors. This mixin extends
    the regressor mixin from sklearn. To create your own
    regressor, do

    >>> class MyRegressor(RegressionMixin):
    >>>     def _fit(self, X, y):
    >>>         # Your fit code
    >>>         pass
    >>> 
    >>>     def _predict(self, X):
    >>>         # Optional, if you require a custom prediction
    >>>         # Defaults to
    >>>         return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1) \
    >>>                       * self.y_std_ + self.y_mean_

    One function should be implemented: the _fit
    function which fits your model based on the encoded
    and normalized X, and normalized y. It should set the
    parameters specified below. Inside the _fit function,
    you have access to the attributes specified below.

    Optionally, you can implement your own prediction
    function, however, when setting the coefficients and
    terms correctly, this should not be necessary. The
    _predict function receives a normalized and encoded
    X.

    Any attributes suffixed by `_` is only accessible after
    fitting.

    .. note::
        Regressor should be able to handle both OLS and
        mixed models, or raise an error otherwise. Use
        `fit_fn\_` attribute to fit a model given some
        terms and data. It automatically accounts for OLS
        vs. mixed model.

    .. note::
        If you require access to the attributes `factors`,
        `re` or `Y2X`, use the underscored versions `_factors`,
        `_re` and `_Y2X`. As sklearn does not permit to adapt
        these factors directly, they may be adapted during fitting.

    Parameters
    ----------
    terms\_ : np.array(1d)
        The indices of the terms (= columns in X)
        in the model.
    coef\_ : np.array(1d)
        An array of coefficients corresponding to
        the terms.
    scale\_ : float
        The scale (= variance of the fit).
    vcomp\_ : float
        The estimates of any presented variance components.
    fit\_ : optional
        The result of calling

        >>> fit_fn_(X, y, self.terms_)

        if applicable. If not specified,
        :py:func:`summary <pyoptex.analysis.mixins.fit_mixin.summary>`
        is unavailable.

    Attributes
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
    n_features_in\_ : int
        The number of features. Equals len(self._factors).
    features_names_in\_ : list(str)
        The names of the features.
    n_encoded_features\_ : int
        The number of encoded features. Is the result of Y2X(Y).shape[1].
    effect_types\_ : np.array(1d)
        An array indicating the type of each factor (effect). A
        1 indicates a continuous variable, anything higher indicates
        a categorical factor with that many levels. Can be
        used for internal package functions such as 
        :py:func:`encode_model <pyoptex.utils.model.encode_model>`.
    coords\_ : :py:class:`numba.typed.List`
        A list of 2d numpy arrays. Each element corresponds to
        the possible encodings of a factor. Retrieved using
        factor.coords\_ property.
    y_mean\_ : float
        The mean y-value, used in normalization.
    y_std\_ : float
        The standard deviation of the y-value, used in normalization.
    fit_fn\_ : func(X, y, terms)
        A fit function used to fit a model from data and the specified
        terms. When random effects are specified, this fits a 
        mixed model, otherwise an OLS is fitted.
    Zs\_ : np.array(2d)
        The groups of each random effect. Zs.shape[0] == len(self._re)
        and Zs.shape[1] == len(X). For example, if the first row is
        [0, 0, 1, 1], then the first two runs are in group 0 according
        to the first random effect, and the last two runs are in group 1.
    is_fitted\_ : bool
        Whether the regressor has been fitted.
    """

    def __init__(self, factors=(), Y2X=identityY2X, random_effects=()):
        """
        Creates the regressor

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
        """
        # Store the parameters
        self.factors = factors
        self.Y2X = Y2X
        self.re = random_effects

    def _regr_params(self, X, y):
        """
        Callback function to dynamically alter the inputted variables.
        By default simply references them.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with the original data.
        y : pd.Series or np.array(1d)
            The output variable.
        """
        self._factors = self.factors
        self._re = self.re
        self._Y2X = self.Y2X

    def _compute_derived(self):
        """
        Computes any necessary derived parameters from the
        inputs such as the number of features, feature names,
        effect types and coordinates for encoding the 
        categorical variables.
        """
        # Compute derived parameters from the inputs
        self.n_features_in_ = len(self._factors)
        self.features_names_in_ = [str(f.name) for f in self._factors]
        self.effect_types_ = np.array([
            1 if f.is_continuous else len(f.levels) 
            for f in self._factors
        ])
        self.coords_ = List([f.coords_ for f in self._factors])

    @property
    def is_fitted(self):
        """
        Checks whether the regressor has been fitted.

        Returns
        -------
        is_fitted : bool
            True when the regressor has been fitted.
        """
        return getattr(self, 'is_fitted_', False)

    def _validate_X(self, X):
        """
        Validates an inputted X (=data).

        Parameters
        ----------
        X : pd.DataFrame
            The data.
        """
        assert isinstance(X, pd.DataFrame), f'X must be a dataframe'
        assert all(c in X.columns for c in self.features_names_in_), f'X does not have the correct features'
        for f in self._factors:
            if f.is_categorical:
                assert all(l in f.levels for l in X[str(f.name)].unique()), f'X contains a categorical level not specified in the factor, unable to encode'

    def _preprocess_X(self, X):
        """
        Preprocesses X by normalizing, encoding 
        the categorical factors, and applying
        Y2X.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        X : np.array(2d)
            The normalized, encoded model matrix of the data.
        """
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
        """
        Validate the inputted parameters before fitting the model.

        Parameters
        ----------
        X : pd.DataFrame
            The data.
        y : pd.Series or np.array(1d)
            The output variable.
        """
        # Validate init parameters
        assert len(self._factors) > 0, f'Must have at least one factor'

        # Validate inputs
        self._validate_X(X)
        q = 'Did you forget the random effects?' if X.shape[1] == self.n_features_in_ else ''
        assert X.shape[1] == self.n_features_in_ + len(self._re), f'X does not have the correct number of features: {self.n_features_in_ + len(self.re)} vs. {X.shape[1]}. {q}'
        assert all(c in X.columns for c in self._re), f'X does not have the correct random effects'

    def preprocess_fit(self, X, y):
        """
        Preprocesses before fitting the data.
        Applies _preprocess_X and normalizes the
        output variable. Also creates
        the `fit_fn\_` attribute by analyzing the
        random effects.

        Parameters
        ----------
        X : pd.DataFrame
            The data.
        y : pd.Series or np.array(1d)
            The output variable.

        Returns
        -------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
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
        """
        To be implemented by the user. It should fit the data
        and create attributes terms\_, coef\_, scale\_, vcomp\_,
        and optionally fit\_.
        See :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin`
        for more information.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        y : np.array(1d)
            The normalized output variable.
        """
        raise NotImplementedError('The fit function has not been implemented')

    def fit(self, X, y):
        """
        Fits the data. After fitting, you can use
        the :py:func:`predict <pyoptex.analysis.mixins.fit_mixin.RegressionMixin.predict>`
        function.

        Make sure the input data X only has the necessary
        columns present.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series or np.array(1d)
            The output variable
        """
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
        """
        Validates the data for predictions. It should
        have the correct columns and only those columns
        present.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        """
        # Validate X
        self._validate_X(X)
        assert X.shape[1] == self.n_features_in_, f'X does not have the correct number of features: {self.n_features_in_} vs. {X.shape[1]}'

    def preprocess_predict(self, X):
        """
        Preprocessing the incoming data
        before prediction. It normalized, encodes
        and converts to a model matrix.

        Parameters
        ----------
        X : pd.DataFrame
            The data.

        Returns
        -------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.
        """
        # Preprocess X
        X = self._preprocess_X(X)

        return X

    def _predict(self, X):
        """
        Internal predict function based on the
        encoded, normalized model matrix of the data.
        It applies the simple linear regression formula.

        Parameters
        ----------
        X : np.array(2d)
            The encoded, normalized model matrix of the data.

        Returns
        -------
        pred : np.array(1d)
            The predictions.
        """
        # Predict based on linear regression
        return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1) \
                    * self.y_std_ + self.y_mean_

    def predict(self, X):
        """
        Predict on new data after fitting.
        Make sure the data only has the necessary
        columns used during fitting.

        Parameters
        ----------
        X : pd.DataFrame    
            The data.
        
        Returns
        -------
        pred : np.array(1d)
            The predictions.
        """
        # Drop potential remaining random effects
        X = X.drop(columns=list(self._re), errors='ignore')

        # Validate this model has been fitted
        assert self.is_fitted, 'You must fit the regressor before predicting'
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
        Prediction variances for the new values specified in X.
        It includes the additional variance from the random
        errors and random effects for a new prediction.

        Parameters
        ----------
        X : np.array(2d)
            The normalized, encoded model matrix of the data.

        Returns
        -------
        pred_var : np.array(1d)
            The prediction variance for each sample.
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
        """
        Prediction variances for the new values specified in X.
        It includes the additional variance from the random
        errors and random effects for a new prediction.

        Parameters
        ----------
        X : pd.DataFrame
            The data

        Returns
        -------
        pred_var : np.array(1d)
            The prediction variance for each sample.
        """
        X = self.preprocess_predict(X)
        return self._pred_var(X)

    def model_formula(self, model):
        """
        Creates the prediction formula of the fit for the encoded and
        normalized data. This function assumes the regressor was
        fitted with the result of :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`.
        In that case, the model can be provided to automatically
        generate the correct labels.

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
            If you did not create Y2X using
            :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`,
            use
            :py:func:`formula <pyoptex.analysis.mixins.fit_mixin.RegressionMixin.formula>`.
            You must manually specify the labels here.

        Parameters
        ----------
        model : pd.DataFrame
            The dataframe of the model used in
            :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`.

        Returns
        -------
        formula : str
            The prediction formula for encoded and normalized data.
        """
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
        Creates the prediction formula of the fit for the encoded and
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
            :py:func:`model_formula <pyoptex.analysis.mixins.fit_mixin.RegressionMixin.model_formula>`.
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
        """
        Generates a summary of the fit in case it was stored
        during training in the `fit\_` attribute. Use as

        >>> print(regr.summary())
        """
        if hasattr(self, 'fit_'):
            return self.fit_.summary()
        else:
            raise AttributeError('Must have a fit_ object to print a fit summary')

class MultiRegressionMixin(RegressionMixin):
    """
    Base mixin for all regressors which output multiple models
    during the model selection. This mixin extends
    :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`,
    which extends the regression mixin from sklearn. To create your own
    regressor, do

    >>> class MyMultiRegressor(MultiRegressionMixin):
    >>>     def _fit(self, X, y):
    >>>         # Your fit code
    >>>         pass
    >>> 
    >>>     def _predict(self, X):
    >>>         # Optional, if you require a custom prediction
    >>>         # Defaults to
    >>>         return np.sum(X[:, self.terms_] * np.expand_dims(self.coef_, 0), axis=1) \
    >>>                       * self.y_std_ + self.y_mean_

    One function should be implemented: the _fit
    function which fits your model based on the encoded
    and normalized X, and normalized y. It should set the
    parameters specified below. Inside the _fit function,
    you have access to the attributes specified below.

    Optionally, you can implement your own prediction
    function, however, when setting the coefficients and
    terms correctly, this should not be necessary. The
    _predict function receives a normalized and encoded
    X.

    Any attributes suffixed by `_` is only accessible after
    fitting.

    .. note::
        Contains the same attributes as
        :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`.

    .. note::
        Prediction happens based on the top model (is the first model 
        in `models\_`). To predict based on any other model, fit that
        specific model using
        :py:class:`SimpleRegressor <pyoptex.analysis.estimators.simple_model.SimpleRegressor>`.

        Assume it is based on a model (in a pandas dataframe) and you fitted
        a multi-regression model `multi_regr`:

        >>> model = ...
        >>> multi_regr = ...
        >>> 
        >>> terms = multi_regr.models_[1]
        >>> new_model = model.iloc[terms]
        >>> Y2X = model2Y2X(new_model, factors)
        >>> 
        >>> regr = SimpleRegressor(factors, Y2X, random_effects).fit(X, y)

    Parameters
    ----------
    models\_ : list(np.array(1d))
        The list of models, sorted by the selection_metrics\_ (highest metric first).
        Each model is an integer array specifying the selected terms.
    selection_metrics\_ : np.array(1d)
        The metric of each model, sorted highest first. The selection metric
        defines the order in which the models should be analyzed.
    metric_name\_ : str
        The name of the selection metric.
    """

    def __init__(self, factors=(), Y2X=identityY2X, random_effects=()):
        """
        Creates the regressor

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
        """
        super().__init__(factors, Y2X, random_effects)

    def fit(self, X, y):
        """
        Fits the data. After fitting, you can use
        the :py:func:`predict <pyoptex.analysis.mixins.fit_mixin.RegressionMixin.predict>`
        and :py:func:`plot_selection <pyoptex.analysis.mixins.fit_mixin.MultiRegressionMixin.plot_selection>`
        functions.

        Make sure the input data X only has the necessary
        columns present.

        Parameters
        ----------
        X : pd.DataFrame
            The data
        y : pd.Series or np.array(1d)
            The output variable
        """
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

        # Add additional parameters required for RegressionMixin
        self.terms_ = self.models_[0]
        self.fit_ = self.fit_fn_(X, y, self.terms_)
        self.coef_ = self.fit_.params[:self.fit_.k_fe]
        self.scale_ = self.fit_.scale
        self.vcomp_ = self.fit_.vcomp

        # Mark as fitted
        self.is_fitted_ = True

        return self

    def plot_selection(self, ntop=5):
        """
        Creates a selection plot to visually display how the top performing
        models were selected and ordered.

        .. note::
            There are no restrictions on how the plot should look.
            This depends entirely on the model selection algorithm.

        Parameters
        ----------
        ntop : int
            The number of top models to indicate in the selection plot.

        Returns
        -------
        fig : :py:class:`plotly.graph_objects.Figure`        
        """
        raise NotImplementedError('No selection plot was implemented')
