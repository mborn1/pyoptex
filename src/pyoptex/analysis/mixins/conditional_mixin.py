import numpy as np
import pandas as pd

from ...doe.fixed_structure import Factor
from ...utils.model import encode_model, x2fx, model2encnames

class ConditionalRegressionMixin:

    def __init__(self, *args, conditional=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = conditional

    def _regr_params(self, X, y):
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
        if labels is not None \
                and self.conditional\
                and len(labels) != self.n_encoded_features_:
            # Add the conditional labels
            effect_types = np.array([len(f.levels) for f in self._conditional_factors])
            labels = [*labels, *model2encnames(self._conditional_model, effect_types)]
            

        return super().formula(labels)
