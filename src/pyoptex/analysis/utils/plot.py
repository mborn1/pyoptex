"""
Module for analysis plotting utilities.
"""

import numpy as np 
import scipy.stats as spstats
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots


def plot_res_diagnostics(df, y_true='y', y_pred='pred', textcols=(), color=None):
    """
    Plots the residual diagnostics of the fit. This plot contains
    four subplots in a 2-by-2 grid.

    * The upper left is the predicted vs. real plot. The black diagonal indicates
      the perfect fit.
    * The upper right is the predicted vs. error plot. This can indicate
      if there is any trend or correlation between the predictions and the 
      random error (they should be uncorrelated).
    * The lower left is the quantile-quantile plot for a normal distribution
      of the errors. The black diagonal line indicates the perfect normal
      distribution of the errors.
    * The lower right is the run vs. error plot. For example, if the runs
      are ordered in time, this plot indicates if effects are missing.
      A trend indicates a time related component, or something which changed with
      time. An offset for certain blocks of consecutive runs may indicate
      a missing effect if using design of experiments with hard-to-change factors.
      Ideally, there are no trends or correlations.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with the data, output, and predictions.
    y_true : str    
        The name of the output column.
    y_pred : str    
        The name of the prediction column.
    textcols : list(str) or tuple(str)
        Any columns which should be added as text upon hover over the graph.
    color : str
        The column to group by with colors in the plot. Can be used to identify
        missing effects for the easy-to-change variables. Note that you
        any continuous variable should be binned.
    
    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure with the residual diagnostics.
    """
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
