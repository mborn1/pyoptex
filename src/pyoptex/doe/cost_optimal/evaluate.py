"""
Module containing all the evaluation functions, specific to the CODEX algorithm
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from ...utils.design import encode_design, obs_var_from_Zs
from ...utils.model import model2encnames
from .metric import Iopt
from .utils import obs_var_Zs


def evaluate_metrics(Y, params, metrics):
    """
    Evaluate the design on a set of metrics.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    metrics : list(:py:class:`Metric <pyoptex.doe.cost_optimal.metric.Metric>`)
        The list of metrics to evaluate.
    
    Returns
    -------
    metrics : list(float)
        The resulting evaluations of the metrics on the design.
    """
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([
        np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) 
        for ratios in params.ratios
    ])
    costs = params.fn.cost(Y, params)

    # Initialize the metrics
    for metric in metrics:
        metric.init(params)

    # Compute the metrics
    return [metric.call(Y, X, Zs, Vinv, costs) for metric in metrics]

def fraction_of_design_space(Y, params, N=10000):
    """
    Computes the fraction of the design space. It returns an array of relative
    prediction variances corresponding to the quantiles of np.linspace(0, 1, `N`).

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    N : int
        The number of samples to evaluate.

    Returns
    -------
    pred_var : np.array(2d)
        The array of relative prediction variances for each of the a-priori variance
        ratio sets provided.
    """
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([
        np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) 
        for ratios in params.ratios
    ])
    costs = params.fn.cost(Y, params)

    # Initialize Iopt
    iopt = Iopt(n=N, cov=params.fn.metric.cov)
    iopt.init(params)

    # Compute information matrix
    if iopt.cov is not None:
        _, X, _, Vinv = iopt.cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute prediction variances
    pred_var = np.sum(
        iopt.samples.T 
        * np.linalg.solve(
            M,
            np.broadcast_to(
                iopt.samples.T, 
                (M.shape[0], *iopt.samples.T.shape)
            )
        ), axis=-2
    )
    pred_var = np.sort(pred_var)

    return pred_var

def plot_fraction_of_design_space(Y, params, N=10000):
    """
    Plots the fraction of the design space. One is plotted
    for each set of a-prior variance components.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    N : int
        The number of samples to evaluate.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure with the fraction of design space plot.
    """
    # Compute prediction variances
    pred_var = fraction_of_design_space(Y, params, N=N)

    # Create the figure
    fig = go.Figure()
    for i, pv in enumerate(pred_var):
        color = DEFAULT_PLOTLY_COLORS[i]
        name = ', '.join([
            f'{str(f.name)}={r:.3f}' 
            for f, r in zip(params.factors, params.ratios[i]) 
            if f.grouped
        ])
        fig.add_trace(go.Scatter(
            x=np.linspace(0, 1, len(pv)), y=pv, 
            marker_color=color, name=name
        ))
        fig.add_hline(
            y=np.mean(pv), annotation_text=f'{np.mean(pv):.3f}', 
            annotation_font_color=color, 
            line_dash='dash', line_width=1, 
            line_color=color, annotation_position='bottom right'
        )

    # Set axis
    fig.update_layout(
        xaxis_title='Fraction of design space',
        yaxis_title='Relative prediction variance',
        legend_title_text='A-priori variance ratios',
        title='Fraction of design space plot',
        title_x=0.5
    )

    return fig

def estimation_variance_matrix(Y, params):
    """
    Computes the parameter estimation covariance matrix.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    est_var : np.array(3d)
        The mutiple parameter estimation covariance matrices for each of the
        a-priori variance ratio sets.
    """
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([
        np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) 
        for ratios in params.ratios
    ])
    costs = params.fn.cost(Y, params)

    # Compute information matrix
    if params.fn.metric.cov is not None:
        _, X, _, Vinv = params.fn.metric.cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    return Minv

def plot_estimation_variance_matrix(Y, params, model=None):
    """
    Plots the parameter estimation covariance matrix. One is plotted
    for each set of a-prior variance components.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    model : None or pd.DataFrame
        The model dataframe corresponding to the Y2X function in order
        to extract the parameter names.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure of a heatmap of the parameter estimation covariance matrix.
    """
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, params)

    # Determine the encoded column names
    if model is None:
        encoded_colnames = np.arange(Minv.shape[-1])
    else:
        col_names = [str(f.name) for f in params.factors]
        encoded_colnames = model2encnames(model[col_names], params.effect_types)
        if len(encoded_colnames) < Minv.shape[-1]:
            encoded_colnames.extend([f'cov_{i}' for i in range(Minv.shape[-1] - len(encoded_colnames))])

    # Create the figure
    fig = make_subplots(rows=len(Minv), cols=1, row_heights=list(np.ones(len(Minv))/len(Minv)), 
        vertical_spacing=0.07,
        subplot_titles=[
            'A-priori variance ratios: ' + ', '.join([f'{str(f.name)}={r:.3f}' for f, r in zip(params.factors, params.ratios[i]) if f.grouped])
            for i in range(len(Minv))
        ]
    )
    for i in range(len(Minv)):
        fig.add_trace(go.Heatmap(
            z=np.flipud(Minv[i]), x=encoded_colnames, y=encoded_colnames[::-1], colorbar_len=0.75/len(Minv),
            colorbar_x=1, colorbar_y=1-(i+0.75/2+0.05*i)/len(Minv)
        ), row=i+1, col=1)
    fig.update_layout(
        title='Estimation covariance plot',
        title_x=0.5
    )

    # Return the plot
    return fig

def estimation_variance(Y, params):
    """
    Computes the variances of the parameter estimations. This is the diagonal
    of :py:func:`estimation_variance_matrix <pyoptex.doe.cost_optimal.evaluate.estimation_variance_matrix>`.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    est_var : np.array(2d)
        The parameter estimation variances. This is the diagonal of
        :py:func:`estimation_variance_matrix <pyoptex.doe.cost_optimal.evaluate.estimation_variance_matrix>`
    """
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, params)
    return np.stack([np.diag(Minv[i]) for i in range(len(Minv))])
