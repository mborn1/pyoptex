"""
Module containing all the evaluation functions, specific to the CODEX algorithm
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from ...utils.design import encode_design, obs_var_from_Zs
from ...utils.model import model2encnames
from ..utils.evaluate import _plot_prediction_variance, _prediction_variance
from .metric import Iopt
from .utils import Parameters, obs_var_Zs


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
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].astype(float).to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) for ratios in params.ratios])
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
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].astype(float).to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) for ratios in params.ratios])
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
        iopt.samples.T * np.linalg.solve(M, np.broadcast_to(iopt.samples.T, (M.shape[0], *iopt.samples.T.shape))),
        axis=-2,
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
        name = ", ".join(
            [f"{f.name!s}={r:.3f}" for f, r in zip(params.factors, params.ratios[i], strict=True) if f.grouped]
        )
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(pv)), y=pv, marker_color=color, name=name))
        fig.add_hline(
            y=np.mean(pv),
            annotation_text=f"{np.mean(pv):.3f}",
            annotation_font_color=color,
            line_dash="dash",
            line_width=1,
            line_color=color,
            annotation_position="bottom right",
        )

    # Set axis
    fig.update_layout(
        xaxis_title="Fraction of design space",
        yaxis_title="Relative prediction variance",
        legend_title_text="A-priori variance ratios",
        title="Fraction of design space plot",
        title_x=0.5,
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
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
    Y = Y.copy()

    # Normalize Y
    for f in params.factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in params.factors]
    Y = Y[col_names].astype(float).to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    Zs = obs_var_Zs(Y, params.colstart, grouped_cols=params.grouped_cols)
    Vinv = np.array([np.linalg.inv(obs_var_from_Zs(Zs, len(Y), ratios)) for ratios in params.ratios])
    costs = params.fn.cost(Y, params)

    # Compute information matrix
    if params.fn.metric.cov is not None:
        _, X, _, Vinv = params.fn.metric.cov(Y, X, Zs, Vinv, costs)
    M = X.T @ Vinv @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    return Minv


def plot_estimation_variance_matrix(Y, params, model=None, abs=False):
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
    abs : bool
        Whether to plot the actual estimation variances, or the absolute values
        of them.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure of a heatmap of the parameter estimation covariance matrix.
    """
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, params)

    # Optionally plot the absolute values
    if abs:
        Minv = np.abs(Minv)

    # Determine the encoded column names
    if model is None:
        encoded_colnames = np.arange(Minv.shape[-1])
    else:
        col_names = [str(f.name) for f in params.factors]
        encoded_colnames = model2encnames(model[col_names], params.effect_types)
        if len(encoded_colnames) < Minv.shape[-1]:
            encoded_colnames.extend([f"cov_{i}" for i in range(Minv.shape[-1] - len(encoded_colnames))])

    # Create the figure
    fig = make_subplots(
        rows=len(Minv),
        cols=1,
        row_heights=list(np.ones(len(Minv)) / len(Minv)),
        vertical_spacing=0.07,
        subplot_titles=[
            "A-priori variance ratios: "
            + ", ".join(
                [f"{f.name!s}={r:.3f}" for f, r in zip(params.factors, params.ratios[i], strict=True) if f.grouped]
            )
            for i in range(len(Minv))
        ],
    )
    for i in range(len(Minv)):
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(Minv[i]),
                x=encoded_colnames,
                y=encoded_colnames[::-1],
                colorbar_len=0.75 / len(Minv),
                colorbar_x=1,
                colorbar_y=1 - (i + 0.75 / 2 + 0.05 * i) / len(Minv),
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(title="Estimation covariance plot", title_x=0.5)

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


def prediction_variance(
    Y: pd.DataFrame,
    params: Parameters,
    x: str,
    y: str | None = None,
    n: int = 60,
    fixed: Mapping[str, float | str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Computes the relative prediction variance on a grid over two factors,
    or along `x` when `y` is None.
    The factors in `fixed` are held at their value. Every grid point is the
    average over 64 random settings of the other factors (the same for every
    grid point), leaving out the settings that the constraints exclude.
    Mixture components that are not on an axis need a value in `fixed`,
    as averaging over them is not supported yet.
    Covariates such as a time trend are held at 0, the middle of the experiment.
    The mixture component that pyoptex leaves implicit (the rest) cannot be on an axis.

    When a constraint involves a factor that is averaged over, the average can jump
    where the constraint starts to exclude some of that factor's settings, because it
    is then taken over fewer settings. Plotting that factor on one of the axes, or
    fixing it with `fixed`, shows where the jump comes from.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    x : str
        The name of the factor on the horizontal axis.
    y : str or None
        The name of the factor on the vertical axis. None gives the prediction
        variance along `x`.
    n : int
        The number of grid points along a continuous factor, from its minimum
        to its maximum (from the lowest to the highest level of a mixture component).
        A categorical factor has its levels.
    fixed : dict or None
        The values of factors that are not on an axis, as {factor name: value}
        in the factor's units (a level name for a categorical factor).
        The other factors are averaged over.

    Returns
    -------
    xs : np.array(1d)
        The coordinates along `x` (the level indices of a categorical factor).
    ys : np.array(1d) or None
        The coordinates along `y`, None along one factor.
    pred_var : np.array(3d) or np.array(2d)
        The relative prediction variances for each of the a-priori variance
        ratio sets provided, of shape (sets, len(ys), len(xs)), or (sets, len(xs))
        along one factor. NaN where the constraints exclude every setting.
    """
    Minv = estimation_variance_matrix(Y, params)
    return _prediction_variance(params, Minv, x, y, n, fixed)


def plot_prediction_variance(
    Y: pd.DataFrame,
    params: Parameters,
    x: str,
    y: str | None = None,
    n: int = 60,
    fixed: Mapping[str, float | str] | None = None,
) -> go.Figure:
    """
    Plots the relative prediction variance over two factors, as computed by
    :py:func:`prediction_variance <pyoptex.doe.cost_optimal.evaluate.prediction_variance>`.
    One is plotted for each set of a-priori variance components, on the same colour scale.
    Combinations that the constraints exclude are left blank. The runs of the design
    are drawn on top, one dot per combination with the number of runs; only the runs
    at the values in `fixed` are counted. Without `y`, it is
    a curve along `x`, with the runs on it.
    Mixture components that are not on an axis need a value in `fixed`,
    as averaging over them is not supported yet.
    Covariates such as a time trend are held at 0, the middle of the experiment.
    The mixture component that pyoptex leaves implicit (the rest) cannot be on an axis.

    When a constraint involves a factor that is averaged over, the average can jump
    where the constraint starts to exclude some of that factor's settings, because it
    is then taken over fewer settings. Plotting that factor on one of the axes, or
    fixing it with `fixed`, shows where the jump comes from.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters.
    x : str
        The name of the factor on the horizontal axis.
    y : str or None
        The name of the factor on the vertical axis. None gives the prediction
        variance along `x`.
    n : int
        The number of grid points along a continuous factor.
    fixed : dict or None
        The values of factors that are not on an axis, as {factor name: value}
        in the factor's units (a level name for a categorical factor).
        The other factors are averaged over.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure of the relative prediction variance.
    """
    Minv = estimation_variance_matrix(Y, params)
    titles = [
        "A-priori variance ratios: "
        + ", ".join([f"{f.name!s}={r:.3f}" for f, r in zip(params.factors, params.ratios[i], strict=True) if f.grouped])
        for i in range(len(Minv))
    ]
    return _plot_prediction_variance(Y, params, Minv, x, y, n, fixed, titles)
