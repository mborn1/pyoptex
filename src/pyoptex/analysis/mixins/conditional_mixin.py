import numpy as np
import pandas as pd

from ...doe.fixed_structure import Factor
from ...utils.model import encode_model, x2fx, model2encnames

class ConditionalRegressionMixin:
    """
    Mixin to provide additional capabilities of
    fitting a conditional model or not. Should be used
    as 

    >>> class MyRegressor(ConditionalRegressionMixin, RegressionMixin):
    >>>     ...

    See :py:class:`RegressionMixin <pyoptex.analysis.mixins.fit_mixin.RegressionMixin>`
    for more information.

    The conditional model removes any random effects and models them
    as categorical fixed effects. These categorical effects
    are effect encoded.

    Attributes
    ----------
    conditional : bool
        Whether to fit a conditional model or not.
    """

    def __init__(self, *args, conditional=False, **kwargs):
        """
        Initializes the mixin.

        Parameters
        ----------
        conditional : bool
            Whether to create a conditional model or not.
        """
        super().__init__(*args, **kwargs)
        self.conditional = conditional

    def _regr_params(self, X, y):
        """
        Alters the regression parameters by removing the
        random effects and adding them as fixed effects. Also
        upgrade the Y2X function to model the main effects
        of the conditional effects.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe with the original data.
        y : pd.Series or np.array(1d)
            The output variable.
        """
        # Set initial values
        super()._regr_params(X, y)

        # Update those values
        if self.conditional and len(self._re) > 0:
            # Validate all present
            assert all(col in X.columns for col in self._re), 'Not all random effects are present in the dataframe'
            
            # Create conditional factors
            self._conditional_factors = [
                Factor(re, type='categorical', levels=X[re].unique().tolist())
                for re in self._re
            ]
            assert all(len(f.levels) > 1 for f in self._conditional_factors), 'Conditional random effects must have more than 1 level'
            effect_types = np.array([len(f.levels) for f in self._conditional_factors])
            n_conditional_cols = np.sum([len(f.levels) - 1 for f in self._conditional_factors])

            # Extend the factors
            self._factors = list(self._factors) # Copy to prevent altering the original
            self._factors.extend(self._conditional_factors)

            # Create the conditional model
            self._conditional_model = pd.DataFrame(
                np.eye(len(self._conditional_factors) ,dtype=np.int_), 
                columns=[str(f.name) for f in self._conditional_factors]
            )

            # Encode the conditional model
            conditional_model_enc = encode_model(
                self._conditional_model.to_numpy(), 
                effect_types
            )

            # Add additional random effects in the Y2X function
            self._Y2X = lambda Y: np.concatenate((
                self.Y2X(Y[:, :-n_conditional_cols]),
                x2fx(Y[:, -n_conditional_cols:], conditional_model_enc)
            ), axis=1)

            # Clear the random effects
            self._re = ()

        else:
            # Empty dataframe as nothing was added
            self._conditional_model_enc = pd.DataFrame()

    def model_formula(self, model):
        """
        Creates the prediction formula of the fit for the encoded and
        normalized data. This function assumes the regressor was
        fitted with the result of :py:func:`model2Y2X <pyoptex.utils.model.model2Y2X>`.
        In that case, the model can be provided to automatically
        generate the correct labels.

        It also includes the conditional effects automatically.

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
            :py:func:`formula <pyoptex.analysis.mixins.fit_mixin.formula`.
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
        if self.conditional:
            # Make sure model is a dataframe
            assert isinstance(model, pd.DataFrame), 'The specified model must be a dataframe'

            # Create the conditional model
            model = pd.concat((
                model.assign(**{c: 0 for c in self._conditional_model.columns}),
                self._conditional_model.assign(**{c: 0 for c in model.columns})
            ), axis=0, ignore_index=True)

        return super().model_formula(model)

    def formula(self, labels=None):
        """
        Creates the prediction formula of the fit for the encoded and
        normalized data. The labels for each term are given by the 
        `labels` parameter.
        The number of labels must be the number of parameters from Y2X,
        i.e., len(labels) == Y2X(Y).shape[1].

        It also includes the conditional effects automatically.

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
        if labels is not None \
                and self.conditional\
                and len(labels) != self.n_encoded_features_:
            # Add the conditional labels
            effect_types = np.array([len(f.levels) for f in self._conditional_factors])
            labels = [*labels, *model2encnames(self._conditional_model, effect_types)]
            

        return super().formula(labels)
