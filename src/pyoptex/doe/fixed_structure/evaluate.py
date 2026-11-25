"""
Module containing all the evaluation functions for fixed structure designs.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from ...utils.design import encode_design
from ...utils.model import model2encnames
from ..utils.evaluate import _plot_prediction_variance, _prediction_variance
from ..utils.power import _power_table, _term_columns
from .metric import Iopt
from .utils import Parameters


def evaluate_metrics(Y, params, metrics):
    """
    Evaluate the design on a set of metrics.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.
    metrics : list(:py:class:`Metric <pyoptex.doe.fixed_structure.metric.Metric>`)
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

    # Initialize the metrics
    for metric in metrics:
        metric.preinit(params)
        metric.init(Y, X, params)

    # Compute the metrics
    return [metric.call(Y, X, params) for metric in metrics]


def fraction_of_design_space(Y, params, N=10000):
    """
    Computes the fraction of the design space. It returns an array of relative
    prediction variances corresponding to the quantiles of np.linspace(0, 1, `N`).

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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

    # Initialize Iopt
    iopt = Iopt(n=N, cov=params.fn.metric.cov)
    iopt.preinit(params)
    iopt.init(Y, X, params)

    # Compute information matrix
    if iopt.cov is not None:
        _, X = iopt.cov(Y, X)
    M = X.T @ params.Vinv @ X

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
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
        name = ", ".join([f"plot {i + 1} = {r:.3f}" for r in params.ratios[i]]) if len(params.ratios) > 0 else None
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


def _model_matrix(Y, params):
    """
    The model matrix of the design, with the covariate columns after the model's columns.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    X : np.array(2d)
        The model matrix, with the covariates.
    nmodel : int
        The number of columns of Y2X, where the covariate columns begin.
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
    nmodel = X.shape[1]

    # Add the covariates
    if params.fn.metric.cov is not None:
        _, X = params.fn.metric.cov(Y, X)

    return X, nmodel


def estimation_variance_matrix(Y, params):
    """
    Computes the parameter estimation covariance matrix.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    est_var : np.array(3d)
        The mutiple parameter estimation covariance matrices for each of the
        a-priori variance ratio sets.
    """
    # Compute information matrix
    X, _ = _model_matrix(Y, params)
    M = X.T @ params.Vinv @ X

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
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
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
        subplot_titles=(
            [
                "A-priori variance ratios: " + ", ".join([f"plot {i + 1} = {r:.3f}" for r in params.ratios[i]])
                for i in range(len(Minv))
            ]
            if len(params.ratios) > 0
            else None
        ),
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
    of :py:func:`estimation_variance_matrix <pyoptex.doe.fixed_structure.evaluate.estimation_variance_matrix>`.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The simulation parameters.

    Returns
    -------
    est_var : np.array(2d)
        The parameter estimation variances. This is the diagonal of
        :py:func:`estimation_variance_matrix <pyoptex.doe.fixed_structure.evaluate.estimation_variance_matrix>`
    """
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, params)
    return np.stack([np.diag(Minv[i]) for i in range(len(Minv))])


def power(
    Y: pd.DataFrame,
    params: Parameters,
    model: pd.DataFrame | None = None,
    sizes: float | Sequence[float] | Mapping[str, float] | pd.Series = 1,
    alpha: float = 0.05,
    ddf: str = "kenward-roger",
) -> pd.DataFrame:
    """
    Computes the power of the test of each term of the model (except the intercept),
    with Kenward-Roger or Satterthwaite degrees of freedom. A term with several columns,
    such as a categorical factor, is tested with one F-test on all its columns.
    Designs with block effects (`block_effects`) are not supported yet.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
        The simulation parameters, of a fixed structure or a split-plot design.
    model : pd.DataFrame or None
        The model given to model2Y2X; with a custom Y2X, its columns must follow the order of encode_model.
        Without a model (None), each column of Y2X is tested on its own (also an intercept column),
        for a custom Y2X; the result is a DataFrame whose columns (the column indices) can be renamed.
    sizes : float, list(float), dict(str, float) or pd.Series
        The size of the effect of each tested term, or of all of them: the coefficient
        on the coded scale, in units of the residual standard deviation. The columns of
        a term with several columns get it with alternating signs.
        A dict or pd.Series is matched by name, a list by position.
    alpha : float
        The significance level.
    ddf : str
        The denominator degrees of freedom, "kenward-roger" or "satterthwaite".

    Returns
    -------
    power : pd.DataFrame
        The power, one row per set of a-priori variance ratios and one column per tested term.
        NaN when the design cannot estimate a variance component the test needs.
        NaN also, with a warning, when power is not defined for the term: its degrees of freedom are below 1,
        or the Kenward-Roger approximation does not hold in this small design (ddf="satterthwaite" may give a value).
    """
    X, nmodel = _model_matrix(Y, params)
    tests = _term_columns(params, model, nmodel)
    return _power_table(tests, X, list(params.Zs), params.Vinv, sizes, alpha, ddf)


def prediction_variance(
    Y: pd.DataFrame,
    params: Parameters,
    x: str,
    y: str | None = None,
    n: int = 60,
    fixed: Mapping[str, float | str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Computes the relative prediction variance on a grid over `x` and `y`, or along `x` when `y` is None.
    Factors in `fixed` are held at the given value. The other factors are averaged over,
    using only the settings that the constraints allow.
    Covariates such as a time trend are held at 0, the middle of the experiment.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
        The simulation parameters, of a fixed structure or a split-plot design.
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
        The other factors are averaged over. Mixture components that are not on an axis must be
        in `fixed`, as averaging over them is not supported yet.

    Returns
    -------
    xs : np.array(1d)
        The coordinates along `x` (the level indices of a categorical factor).
    ys : np.array(1d) or None
        The coordinates along `y`, None along one factor.
    pred_var : np.array(3d) or np.array(2d)
        The relative prediction variances for each of the a-priori variance
        ratio sets provided, of shape (sets, len(ys), len(xs)), or (sets, len(xs))
        along one factor. NaN where the constraints allow no setting.
    """
    Minv = estimation_variance_matrix(Y, params)
    xs, ys, pred_var, _ = _prediction_variance(params, Minv, x, y, n, fixed)
    return xs, ys, pred_var


def plot_prediction_variance(
    Y: pd.DataFrame,
    params: Parameters,
    x: str,
    y: str | None = None,
    n: int = 60,
    fixed: Mapping[str, float | str] | None = None,
) -> go.Figure:
    """
    Plots the relative prediction variance of
    :py:func:`prediction_variance <pyoptex.doe.fixed_structure.evaluate.prediction_variance>` over `x` and `y`, or along `x` when `y` is None,
    one subplot per set of a-priori variance ratios. Settings that the constraints exclude are blank.
    The runs are drawn on top, with the number of runs at each combination.
    When factors are fixed, only the runs at those values are shown.

    Parameters
    ----------
    Y : pd.DataFrame
        The denormalized, decoded design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
        The simulation parameters, of a fixed structure or a split-plot design.
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
        The other factors are averaged over. Mixture components that are not on an axis must be
        in `fixed`, as averaging over them is not supported yet.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The plotly figure of the relative prediction variance.
    """
    Minv = estimation_variance_matrix(Y, params)
    titles = (
        [
            "A-priori variance ratios: " + ", ".join([f"plot {i + 1} = {r:.3f}" for r in params.ratios[i]])
            for i in range(len(Minv))
        ]
        if len(params.ratios) > 0
        else None
    )
    return _plot_prediction_variance(Y, params, Minv, x, y, n, fixed, titles)
