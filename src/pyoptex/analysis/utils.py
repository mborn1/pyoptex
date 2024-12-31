from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as spstats
import statsmodels.api as sm
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from statsmodels.regression.mixed_linear_model import VCSpec

from ..utils.design import obs_var_from_Zs
from ..utils.model import encode_model


def identity(Y):
    """
    The identity function.

    Parameters
    ----------
    Y : np.array
        The input

    Returns
    -------
    Y : np.array
        The input returned
    """
    return Y

def order_dependencies(model, factors):
    """
    Create a dependency matrix from a model where
    interactions and higher order effects depend
    on their components and lower order effects.

    For example:
    * :math:`x_0`: depends only on the intercept.
    * :math:`x_0^2`: depends on :math:`x_0`, which in turn depends on the intercept.
    * :math:`x_0 x_1`: depends on both :math:`x_0` and :math:`x_1`, which both depend on the intercept.
    * :math:`x_0^2 x_1` : depends on both :math:`x_0^2` and :math:`x_1`, which depend on :math:`x_0` and the intercept.

    Parameters
    ----------
    model : pd.DataFrame
        The model
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.

    Returns
    -------
    dep : np.array(2d)
        The dependency matrix of size (N, N) with N the number
        of terms in the encoded model. Term i depends on term j
        if dep(i, j) = true.
    """
    # Validation
    assert isinstance(model, pd.DataFrame), 'Model must be a dataframe'
    assert np.all(model >= 0), 'All powers must be larger than zero'

    col_names = [str(f.name) for f in factors]
    assert all(col in col_names for col in model.columns), 'Not all model parameters are factors'
    assert all(col in model.columns for col in col_names), 'Not all factors are in the model'

    # Extract factor parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])

    # Detect model in correct order
    model = model[col_names].to_numpy()

    # Encode model
    modelenc = encode_model(model, effect_types)

    # Compute the possible dependencies
    eye = np.expand_dims(np.eye(modelenc.shape[1]), 1)
    model = np.expand_dims(modelenc, 0)
    all_dep = model - eye # all_dep[:, i] are all possible dependencies for term i

    # Valid dependencies
    all_dep_valid = np.where(np.all(all_dep >= 0, axis=2))
    from_terms = all_dep_valid[1]

    # Extract the dependent terms
    to_terms = np.argmax(np.all(
        np.expand_dims(modelenc, 0) == np.expand_dims(all_dep[all_dep_valid], 1), 
        axis=2
    ), axis=1)

    # Compute dependencies
    dep = np.zeros((modelenc.shape[0], modelenc.shape[0]), dtype=np.bool_)
    dep[from_terms, to_terms] = True

    return dep

def model2strong(model, dep):
    # Create a mask
    strong = np.zeros(dep.shape[0], dtype=np.bool_)
    strong[model] = True
    nterms_old = 0
    nterms = np.sum(strong)

    # Loop until no new terms are added
    while nterms_old < nterms:
        # Add dependencies
        strong[np.any(dep[strong], axis=0)] = True

        # Update number of terms
        nterms_old = nterms
        nterms = np.sum(strong)

    return np.flatnonzero(strong)

def r2adj(fit):
    if fit.k_fe < len(fit.params):
        # Extract Xnumber of observations
        nobs = len(fit.model.exog)

        # Compute P
        P = np.eye(nobs) - np.ones((nobs, nobs)) / nobs

        # Fit intercept model
        fit0 = sm.MixedLM(
            fit.model.endog, np.ones((nobs, 1)), fit.model.groups,
            fit.model.exog_re, fit.model.exog_vc, fit.model.use_sqrt
        ).fit()

        # Extract the groups
        vc_mats = fit.model.exog_vc.mats
        Zs = np.stack([np.argmax(vc_mats[i][0], axis=1) for i in range(len(vc_mats))]).T

        # Compute intercept semi-variance
        V0 = obs_var_from_Zs(Zs, nobs, fit0.params[fit.k_fe:]) * fit0.scale
        rss0 = np.sum(V0 * P.T) # = np.trace(V0 @ P)

        # Compute model semi-variance
        V1 = obs_var_from_Zs(Zs, nobs, fit.params[fit.k_fe:]) * fit.scale
        rss = np.sum(V1 * P.T)

        # Compute adjusted R2
        r2a = 1 - rss / rss0
    else:

        # Attribute already exists for OLS
        r2a = fit.rsquared_adj

    return r2a

def fit_ols(X, y):
    fit = sm.OLS(y, X).fit()
    fit.k_fe = len(fit.params)
    fit.vcomp = np.array([], dtype=np.float64)
    fit.converged = True
    return fit

def fit_mixedlm(X, y, groups):
    # Retrieve dummy encoding for each group
    dummies = [pd.get_dummies(group) for group in groups]

    # Create the mixed lm spec
    exog_vc = VCSpec(
        [f'g{i}' for i in range(len(groups))],
        [[[f'g{i}[{col}]' for col in dummy.columns]] for i, dummy in enumerate(dummies)],
        [[dummy.to_numpy()] for dummy in dummies]
    )

    # Fit the model
    fit = sm.MixedLM(y, X, np.ones(len(X)), exog_vc=exog_vc).fit()

    # Add additional values
    fit.rsquared = cached_property(
        lambda self: r2_score(self.model.endog, self.predict(self.model.exog))
    )
    fit.rsquared_adj = cached_property(
        lambda self: r2adj(self)
    )

    return fit

def plot_res_diagnostics(df, y_true='y', y_pred='pred', textcols=(), color=None):
    # Define the colors
    if color is not None:
        unique_colors = df[color].unique()
        npcolor = df[color].to_numpy()
    else:
        unique_colors = [0]
        npcolor = np.zeros(len(df))

    # Compute the error
    y_true = df[y_true].to_numpy()
    y_pred = df[y_pred].to_numpy()
    error = y_pred - y_true

    # Compute the theoretical normal quantiles
    ppf = np.linspace(0, 1, len(error) + 2)[1:-1]
    theoretical_quant = spstats.norm.ppf(ppf)

    # Retrieve the true quantiles
    quant_idx = np.argsort(error)
    quant_idx_inv = np.argsort(quant_idx)
    true_quant = error[quant_idx]
    true_quant = (true_quant - np.nanmean(true_quant)) / np.nanstd(true_quant)

    # Compute figure ranges
    pred_range = np.array([
        min(np.nanmin(y_true), np.nanmin(y_pred)), 
        max(np.nanmax(y_true), np.nanmax(y_pred)),
    ])
    quant_range = np.array([theoretical_quant[0], theoretical_quant[-1]])

    # Create the figure
    fig = make_subplots(2, 2)

    # Loop over all colors
    for i, uc in enumerate(unique_colors):
        # Create subsets
        c = np.flatnonzero(npcolor == uc)
        tt = dict(
            hovertemplate=f'x: %{{x}}<br>y: %{{y}}<br>color: {uc}<br>' \
                    + '<br>'.join(f'{col}: %{{customdata[{j}]}}' for j, col in enumerate(textcols)),
            customdata=df.iloc[c][list(textcols)].to_numpy()
        )

        # Quantile subsets
        cquant = quant_idx_inv[c]

        # Prediction figure
        fig.add_trace(go.Scatter(
            x=y_pred[c], y=y_true[c], mode='markers', 
            marker_color=DEFAULT_PLOTLY_COLORS[i],
            name=str(uc), **tt
        ), row=1, col=1)

        # Error figure 1
        fig.add_trace(go.Scatter(
            x=y_pred[c], y=error[c], mode='markers', 
            marker_color=DEFAULT_PLOTLY_COLORS[i],
            showlegend=False, **tt
        ), row=1, col=2)

        # Error figure 2
        fig.add_trace(go.Scatter(
            x=c, y=error[c], mode='markers', 
            marker_color=DEFAULT_PLOTLY_COLORS[i],
            showlegend=False, **tt
        ), row=2, col=2)

        # QQ-plot
        fig.add_trace(go.Scatter(
            x=theoretical_quant[cquant], y=true_quant[cquant], 
            mode='markers', marker_color=DEFAULT_PLOTLY_COLORS[i],
            showlegend=False, **tt
        ), row=2, col=1)

    # Draw diagonals
    fig.add_trace(go.Scatter(
        x=pred_range, y=pred_range, marker_size=0.01, showlegend=False, line_color='black',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=quant_range, y=quant_range, marker_size=0.01, showlegend=False, line_color='black',
    ), row=2, col=1)

    # Update ax titles
    fig.update_xaxes(title='Predicted', row=1, col=1)
    fig.update_yaxes(title='Real', row=1, col=1)
    fig.update_xaxes(title='Predicted', row=1, col=2)
    fig.update_yaxes(title='Error (prediction - real)', row=1, col=2)
    fig.update_xaxes(title='Run', row=2, col=2)
    fig.update_yaxes(title='Error (prediction - real)', row=2, col=2)
    fig.update_xaxes(title='Theoretical quantile', row=2, col=1)
    fig.update_yaxes(title='Sample quantile', row=2, col=1)

    # Define legend
    fig.update_layout(
        showlegend=(len(unique_colors) > 1),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0, 
            title=color
        )
    )
    

    return fig
