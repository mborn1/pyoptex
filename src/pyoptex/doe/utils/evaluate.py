"""
Module containing all the generic evaluation functions
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numba.typed import List

from ...utils.design import encode_design
from ...utils.model import model2encnames


def design_heatmap(Y, factors):
    """
    Plots the design as a heatmap. Each factor is normalized
    to between -1 and 1. The categorical levels are indicated
    on the design.

    Parameters
    ----------
    Y : pd.DataFrame
        A decoded and denormalized design.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The heatmap of the design as a Plotly figure.
    """
    # Subselect the factors
    Y = Y.copy()
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names]

    # Extract original data
    customdata = Y.to_numpy()

    # Normalize the data
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Create the text labels
    def create_text(y, factor):
        if factor.is_categorical:
            bkps = np.concatenate(([0], np.flatnonzero(np.diff(y))+1, [len(y)]))
            bkps = np.floor(np.diff(bkps) / 2).astype(int) + bkps[:-1]

            max_len = max(len(l) for l in factor.levels)
            txt = np.full(len(y), '', dtype=f'U{max_len}')
            txt[bkps] = factor.denormalize(y[bkps])
        else:
            txt = np.full(len(y), '')
        return txt
    txt = np.stack([create_text(y, f) for y, f in zip(Y.to_numpy().T, factors)], axis=1)

    # Normalize categorical factors for same scale
    for f in factors:
        if f.is_categorical:
                Y[str(f.name)] = Y[str(f.name)] / (len(f.levels) - 1) * 2 - 1

    # Convert to numpy for plotting
    Y = Y.to_numpy()

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=np.flipud(Y), x=col_names, y=np.arange(len(Y))[::-1].astype(str), text=np.flipud(txt), texttemplate="%{text}",
        hovertemplate='<b>Factor</b>: %{x}<br><b>Run</b>: %{y}<br><b>Level</b>: %{customdata}<br><b>Normalized level</b>: %{z}',
        customdata=np.flipud(customdata)
    ))
    fig.update_layout(
        title=f'Design: {len(Y)} runs',
        title_x=0.5
    )

    # Top-down plotting
    return fig

def correlation_map(Y, factors, Y2X, model=None, method='pearson'):
    """
    Computes the map of correlations for the provided design.

    Parameters
    ----------
    Y : pd.DataFrame
        The decoded and denormalized design.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.
    Y2X : func(Y)
        The function to convert the design matrix Y to a
        model matrix X.
    model : pd.DataFrame or None
        The model, used to extract the parameter names.
    method : 'pearson', 'kendall', or 'spearman'
        The correlation method to use.

    Returns
    -------
    corr : pd.DataFrame
        The dataframe of correlations.
    """
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()
    
    # Create the design parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    coords = List([f.coords_ for f in factors])

    # Normalize Y
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, effect_types, coords)

    # Define the metric inputs
    X = Y2X(Y)

    # Determine the encoded column names
    if model is None:
        encoded_colnames = np.arange(Minv.shape[-1])
    else:
        col_names = [str(f.name) for f in factors]
        encoded_colnames = model2encnames(model[col_names], effect_types)

    # Compute the correlations
    corr = pd.DataFrame(X, columns=encoded_colnames).corr(method=method)
    return corr

def plot_correlation_map(Y, factors, Y2X, model=None, method='pearson', drop_nans=True):
    """
    Plots the map of correlations for the provided design.

    Parameters
    ----------
    Y : pd.DataFrame
        The decoded and denormalized design.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.Factor>`)
        The list of factors in the design.
    Y2X : func(Y)
        The function to convert the design matrix Y to a
        model matrix X.
    model : pd.DataFrame or None
        The model, used to extract the parameter names.
    method : 'pearson', 'kendall', or 'spearman'
        The correlation method to use.
    drop_nans : bool
        Whether to drop rows and columns that are completely nan.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The figure of the map of correlations.
    """
    # Compute correlation map
    corr = correlation_map(Y, factors, Y2X, model, method)
    
    # Iteratively drop entire rows and columns of nans
    if drop_nans:
        bad = np.all(np.isnan(corr), axis=1)
        while np.any(bad):
            if isinstance(corr, pd.DataFrame):
                bad = bad.to_numpy()
                corr = corr.iloc[~bad, ~bad]
            else: 
                corr = corr[~bad][:, ~bad]
            bad = np.all(np.isnan(corr), axis=1)

    fig = px.imshow(corr)
    fig.update_layout(
        title='Correlation map',
        title_x=0.5
    )

    return fig
