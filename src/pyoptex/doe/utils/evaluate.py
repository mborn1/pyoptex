"""
Module containing all the generic evaluation functions
"""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...utils.design import encode_design
from ...utils.model import model2encnames
from ..cost_optimal.utils import Parameters as CostOptimalParameters
from ..fixed_structure.splitk_plot.utils import Parameters as SplitkParameters
from ..fixed_structure.utils import Parameters as FixedStructureParameters


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
            bkps = np.concatenate(([0], np.flatnonzero(np.diff(y)) + 1, [len(y)]))
            bkps = np.floor(np.diff(bkps) / 2).astype(int) + bkps[:-1]

            max_len = max(len(lvl) for lvl in factor.levels)
            txt = np.full(len(y), "", dtype=f"U{max_len}")
            txt[bkps] = factor.denormalize(y[bkps])
        else:
            txt = np.full(len(y), "")
        return txt

    txt = np.stack([create_text(y, f) for y, f in zip(Y.to_numpy().T, factors, strict=True)], axis=1)

    # Normalize categorical factors for same scale
    for f in factors:
        if f.is_categorical:
            Y[str(f.name)] = Y[str(f.name)] / (len(f.levels) - 1) * 2 - 1

    # Convert to numpy for plotting
    Y = Y.to_numpy()

    # Create the figure
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(Y),
            x=col_names,
            y=np.arange(len(Y))[::-1].astype(str),
            text=np.flipud(txt),
            texttemplate="%{text}",
            hovertemplate="<b>Factor</b>: %{x}<br><b>Run</b>: %{y}<br><b>Level</b>: %{customdata}<br><b>Normalized level</b>: %{z}",
            customdata=np.flipud(customdata),
        )
    )
    fig.update_layout(title=f"Design: {len(Y)} runs", title_x=0.5)

    # Top-down plotting
    return fig


def correlation_map(Y, factors, Y2X, model=None, method="pearson"):
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
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
    Y = Y.copy()

    # Create the design parameters
    effect_types = np.array([1 if f.is_continuous else len(f.levels) for f in factors])
    coords = [f.coords_ for f in factors]

    # Normalize Y
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names].astype(float).to_numpy()

    # Encode the design
    Y = encode_design(Y, effect_types, coords)

    # Define the metric inputs
    X = Y2X(Y)

    # Determine the encoded column names
    if model is None:
        encoded_colnames = np.arange(X.shape[1])
    else:
        col_names = [str(f.name) for f in factors]
        encoded_colnames = model2encnames(model[col_names], effect_types)

    # Compute the correlations
    corr = pd.DataFrame(X, columns=encoded_colnames).corr(method=method)
    return corr


def plot_correlation_map(Y, factors, Y2X, model=None, method="pearson", drop_nans=True, corr_abs=False):
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
    corr_abs : bool
        Whether to drop the sign of the correlation.
        If True, the colors in the figure only display magnitudes
        using a sequential colormap ('Cividis').
        If False, the colors of the figure show the sign, using
        a diverging colormap ('RdBu').

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The figure of the map of correlations.
    """
    # Compute correlation map
    corr = correlation_map(Y, factors, Y2X, model, method)

    # Iteratively drop entire rows and columns of nans
    if drop_nans:
        while True:
            bad = np.all(np.isnan(corr), axis=1).to_numpy()
            if not bad.any():
                break
            corr = corr.iloc[~bad, ~bad]

    # User can specify if they are interested in sign of correlation or just magnitude
    if corr_abs:
        z = np.flipud(corr.abs().to_numpy())
        zmin, zmax, colorscale, title_suffix = 0, 1, "Cividis", " (absolute)"
    else:
        z = np.flipud(corr.to_numpy())
        zmin, zmax, colorscale, title_suffix = -1, 1, "RdBu", ""

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=corr.columns,
            y=corr.columns[::-1],
            colorscale=colorscale,
            text=np.flipud(corr.map(lambda v: f"{v:+.3f}").to_numpy()),
            hovertemplate="<b>Factor 1</b>: %{x}<br><b>Factor 2</b>: %{y}<br><b>Correlation</b>: %{text}<extra></extra>",
            zmin=zmin,
            zmax=zmax,
        )
    )
    fig.update_layout(title=f"Correlation map{title_suffix}", title_x=0.5)

    return fig


############################################################
# Design space and prediction variance


def _shown_factors(factors, x, y):
    """
    The indices of the factors on the horizontal and vertical axis.
    Without `y` (None), both are the index of `x`: the view along one factor.
    """
    names = [str(f.name) for f in factors]
    for name in (x,) if y is None else (x, y):
        assert str(name) in names, f"Unknown factor {name}, the factors are {names}"
    assert y is None or str(x) != str(y), f"Factor {x} is given as both x and y; leave y out for one factor"
    ix = names.index(str(x))
    return ix, ix if y is None else names.index(str(y))


def _levels(f):
    """The normalized levels of a factor: the level indices of a categorical factor."""
    if f.is_categorical:
        return np.arange(len(f.levels), dtype=np.float64)
    return np.asarray(f.coords_, dtype=np.float64)[:, 0]


def _to_plot(f, norm):
    """The plot coordinates of normalized values: the factor's units, the level index of a categorical factor."""
    return norm if f.is_categorical else norm * f.scale + f.mean


def _axis_grid(f, n):
    """
    The normalized grid along an axis: n points from the minimum to the maximum of a continuous factor,
    from the lowest to the highest level of a mixture component, the level indices of a categorical factor.
    """
    if f.is_categorical:
        return _levels(f)
    if f.is_mixture:
        levels = _levels(f)
        return np.linspace(levels.min(), levels.max(), n)
    return np.linspace(-1, 1, n)


def _axis_layout(f, name):
    """
    The axis of a factor: a categorical factor at its level indices with the level names as ticks,
    a continuous factor with a tick at every level (up to 10 levels, else Plotly's ticks).
    """
    if f.is_categorical:
        k = len(f.levels)
        return dict(
            title=name,
            zeroline=False,
            tickvals=list(range(k)),
            ticktext=[str(v) for v in f.levels],
            range=[-0.5, k - 0.5],
        )
    levels = np.unique(_to_plot(f, _levels(f)))
    ticks = dict(tickvals=[float(v) for v in levels]) if len(levels) <= 10 else {}
    return dict(title=name, zeroline=False, **ticks)


def _format_value(f, value):
    """A value in the factor's units as text: the level name of a categorical factor."""
    return str(value) if f.is_categorical else f"{float(value):.4g}"


def _fixed_settings(factors, fixed, ix, iy):
    """Check `fixed` ({factor name: value in the factor's units}) and return {factor index: normalized value}."""
    names = [str(f.name) for f in factors]
    norm = {}
    for name, value in fixed.items():
        assert name in names, f"Unknown factor {name} in fixed, the factors are {names}"
        i = names.index(name)
        assert i not in (ix, iy), f"Factor {name} is on an axis, so it cannot be in fixed"
        f = factors[i]
        if f.is_categorical:
            levels = list(f.levels)
            assert value in levels, f"Factor {name} has no level {value} (in fixed), its levels are {levels}"
            norm[i] = float(levels.index(value))
        elif f.is_mixture:
            lo, hi = _levels(f).min(), _levels(f).max()
            assert lo <= value <= hi, (
                f"Mixture component {name} must be between its levels {lo} and {hi}, but is {value}"
            )
            norm[i] = float(f.normalize(value))
        else:
            assert f.min <= value <= f.max, f"Factor {name} must be between {f.min} and {f.max}, but is {value}"
            norm[i] = float(f.normalize(value))
    return norm


def _fixed_labels(factors, fixed):
    """The fixed values as text: for a title, and as lines to add to a hover."""
    names = [str(f.name) for f in factors]
    parts = [f"{name} = {_format_value(factors[names.index(name)], value)}" for name, value in fixed.items()]
    return ", ".join(parts), "".join(f"<br>{part}" for part in parts)


def _other_settings(factors, ix, iy, fixed_norm, variance):
    """
    The settings of the factors that are not on the axes, normalized and decoded (a categorical factor
    as its level index), one per row; the columns of the factors on the axes are 0.
    The factors in `fixed_norm` are at their value. The others are at 64 random settings (seed 0):
    a continuous factor uniform between its minimum and maximum, a categorical factor at each level
    equally often, in a random order.
    A mixture component that is not fixed is at its lowest level for the design space, and raises an
    AssertionError for the prediction variance (`variance`).

    Returns the settings and the indices of the factors that are averaged over.
    """
    hidden = [i for i in range(len(factors)) if i not in (ix, iy) and i not in fixed_norm]
    mixtures = [i for i in hidden if factors[i].is_mixture]
    if variance:
        assert len(mixtures) == 0, "Averaging over mixture components is not supported yet: give their values in fixed."
    averaged = [i for i in hidden if i not in mixtures]

    nsettings = 64
    settings = np.zeros((nsettings if len(averaged) > 0 else 1, len(factors)))
    for i, v in fixed_norm.items():
        settings[:, i] = v
    for i in mixtures:
        settings[:, i] = _levels(factors[i]).min()
    rng = np.random.default_rng(0)
    for i in averaged:
        f = factors[i]
        settings[:, i] = (
            rng.permutation(np.arange(nsettings) % len(f.levels)) if f.is_categorical else rng.uniform(-1, 1, nsettings)
        )
    return settings, averaged


def _grid_rows(points, settings):
    """
    Every combination of grid point ({factor index: normalized values}) and setting, in chunks:
    yields the slice of the grid points and the decoded rows, grid point by grid point.
    """
    npoints = len(next(iter(points.values())))
    chunk = max(1, 200_000 // len(settings))
    for start in range(0, npoints, chunk):
        sl = slice(start, min(start + chunk, npoints))
        rows = np.repeat(settings[np.newaxis], sl.stop - sl.start, axis=0)
        for i, v in points.items():
            rows[:, :, i] = v[sl, np.newaxis]
        yield sl, rows.reshape(-1, settings.shape[1])


def _excluded(params, Yenc):
    """Which rows of the encoded design the constraints in `params` exclude."""
    return np.broadcast_to(np.asarray(params.fn.constraints(Yenc), dtype=np.bool_), (len(Yenc),))


def _clear_constraints(params):
    """Remove the compiled constraints, so that the parameters can still be pickled for multiprocessing."""
    clear = getattr(params.fn.constraints, "clear", None)
    if clear is not None:
        clear()


def _allowed(params, points, settings):
    """
    Which grid points ({factor index: normalized values}) the constraints allow for at least one setting.
    The grid points are encoded once; setting by setting, only the points that are still excluded are checked.
    """
    settings = np.unique(settings, axis=0)
    npoints = len(next(iter(points.values())))

    # The grid points and the settings, encoded once
    rows = np.repeat(settings[:1], npoints, axis=0)
    for i, v in points.items():
        rows[:, i] = v
    Yenc = encode_design(rows, params.effect_types, params.coords)
    Senc = encode_design(settings, params.effect_types, params.coords)
    shown = np.zeros(Yenc.shape[1], dtype=np.bool_)
    for i in points:
        shown[params.colstart[i] : params.colstart[i + 1]] = True

    # Setting by setting, the points that no earlier setting allows
    ok = np.zeros(npoints, dtype=np.bool_)
    todo = np.arange(npoints)
    for s in range(len(settings)):
        rows = Yenc[todo]
        rows[:, ~shown] = Senc[s, ~shown]
        excluded = _excluded(params, rows)
        ok[todo[~excluded]] = True
        todo = todo[excluded]
        if len(todo) == 0:
            break
    return ok


def _runs_trace(Y, factors, ix, iy, fixed, fixed_hover, y_at=None):
    """
    The runs as dots, one per combination of the factors on the axes, sized by and labelled with
    the number of runs there, counting only the runs at the values in `fixed` (a categorical factor equal,
    a continuous one close). Along one factor, the dots are at height `y_at(x)`, or 0.
    """
    names = [str(f.name) for f in factors]
    match = np.ones(len(Y), dtype=np.bool_)
    for name, value in fixed.items():
        if factors[names.index(name)].is_categorical:
            match &= (Y[name] == value).to_numpy()
        else:
            match &= np.isclose(Y[name].to_numpy(dtype=np.float64), float(value))
    Y = Y[match]

    def coords(f):
        # The plot coordinates of the runs: the level index of a categorical factor
        if f.is_categorical:
            return np.asarray(f.normalize(Y[str(f.name)]), dtype=np.int64)
        return Y[str(f.name)].to_numpy(dtype=np.float64)

    fx, fy = factors[ix], factors[iy]
    cols = {"x": coords(fx)} if ix == iy else {"x": coords(fx), "y": coords(fy)}
    runs = pd.DataFrame(cols).groupby(list(cols)).size().reset_index(name="runs")
    rx = runs["x"].to_numpy(dtype=np.float64)
    ry = runs["y"].to_numpy(dtype=np.float64) if ix != iy else (y_at(rx) if y_at is not None else np.zeros(len(runs)))
    count = runs["runs"].to_numpy()

    def label(f, v):
        return str(f.levels[int(v)]) if f.is_categorical else f"{v:.4g}"

    hover = [
        f"{names[ix]}: {label(fx, a)}"
        + (f"<br>{names[iy]}: {label(fy, b)}" if ix != iy else "")
        + f"{fixed_hover}<br>Runs: {c}"
        for a, b, c in zip(rx, ry, count, strict=True)
    ]
    return go.Scatter(
        x=rx,
        y=ry,
        mode="markers+text",
        text=count.astype(str),
        textposition="middle center",
        hovertext=hover,
        hoverinfo="text",
        cliponaxis=False,
        marker=dict(
            size=22 + 10 * (count - 1) / max(count.max(initial=1) - 1, 1),
            color="#1f1f1f",
            line=dict(color="white", width=1.5),
        ),
        textfont=dict(color="white", size=12),
        showlegend=False,
        name="Runs",
    )


def _level_trace(xp, yp, lines):
    """The allowed levels: lines (NaN between lines and where not allowed), or markers along a categorical factor."""
    return go.Scatter(
        x=xp,
        y=yp,
        mode="lines" if lines else "markers",
        line=dict(color="rgba(120,120,120,0.9)", width=4),
        marker=dict(size=10, color="rgba(120,120,120,0.9)"),
        hoverinfo="skip",
        showlegend=False,
        name="Levels",
    )


def plot_design_space(
    Y: pd.DataFrame,
    params: FixedStructureParameters | SplitkParameters | CostOptimalParameters,
    x: str,
    y: str | None = None,
    fixed: Mapping[str, float | str] | None = None,
) -> go.Figure:
    """
    Plots the design projected on two factors: the levels each factor
    may take as thick lines between its lowest and highest level (points along a
    categorical factor), only where the constraints allow them, the region that
    the constraints exclude as a shaded background (when both factors are continuous),
    and the runs as dots, one per combination, sized by and labelled with the number
    of runs there. A categorical factor is drawn at its level indices,
    with the level names on the axis.

    The constraints are those in `params`, including the mixture rule that
    `default_fn` adds. The factors in `fixed` are held at their value, and only
    the runs at those values are counted. A combination is allowed when the
    constraints allow it for at least one of 64 random settings of the other
    factors (the same for every combination). A very narrow allowed range of a factor
    that is not fixed can therefore be missed. A mixture component that is not on
    an axis or in `fixed` is at its lowest level.
    The mixture component that pyoptex leaves implicit (the rest) cannot be on an axis.

    Parameters
    ----------
    Y : pd.DataFrame
        The decoded and denormalized design.
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters of any module (fixed structure, split-plot or cost optimal).
    x : str
        The name of the factor on the horizontal axis.
    y : str or None
        The name of the factor on the vertical axis. None gives the levels
        of `x` on one line, with the runs per level.
    fixed : dict or None
        The values of factors that are not on an axis, as {factor name: value}
        in the factor's units (a level name for a categorical factor).

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The figure of the design space.
    """
    try:
        assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
        n = 300  # the number of points along a continuous axis
        factors = list(params.factors)
        fixed = dict(fixed or {})
        ix, iy = _shown_factors(factors, x, y)
        fx, fy = factors[ix], factors[iy]
        settings, _ = _other_settings(factors, ix, iy, _fixed_settings(factors, fixed, ix, iy), variance=False)
        fixed_text, fixed_hover = _fixed_labels(factors, fixed)

        def along(f):
            # The normalized points along a factor: between its lowest and highest level, its levels if categorical
            levels = _levels(f)
            return levels if f.is_categorical else np.linspace(levels.min(), levels.max(), n)

        def level_lines(i_along, i_at):
            # One line along factor i_along at every level of factor i_at, NaN between lines and where not allowed
            fa, fb = factors[i_along], factors[i_at]
            a, b = along(fa), _levels(fb)
            A, B = np.meshgrid(a, b)  # rows: the levels of fb
            ok = _allowed(params, {i_along: A.ravel(), i_at: B.ravel()}, settings).reshape(A.shape)
            gap = np.full((len(b), 1), np.nan)
            ap = np.hstack([np.where(ok, _to_plot(fa, A), np.nan), gap]).ravel()
            bp = np.hstack([np.where(ok, _to_plot(fb, B), np.nan), gap]).ravel()
            return ap, bp

        fig = go.Figure()

        if ix == iy:
            # One factor: its allowed levels on one line, with the runs per level
            a = along(fx)
            ok = _allowed(params, {ix: a}, settings)
            fig.add_trace(_level_trace(np.where(ok, _to_plot(fx, a), np.nan), np.zeros(len(a)), fx.is_continuous))
            fig.add_trace(_runs_trace(Y, factors, ix, iy, fixed, fixed_hover))
            fig.update_layout(xaxis=_axis_layout(fx, x), yaxis=dict(visible=False, range=[-1, 1]), height=220)

        else:
            if fx.is_continuous and fy.is_continuous:
                # The excluded region
                gx, gy = _axis_grid(fx, n), _axis_grid(fy, n)
                GX, GY = np.meshgrid(gx, gy)  # rows: y, columns: x
                ok = _allowed(params, {ix: GX.ravel(), iy: GY.ravel()}, settings).reshape(GX.shape)
                if not ok.all():
                    fig.add_trace(
                        go.Heatmap(
                            z=ok.astype(int),
                            x=_to_plot(fx, gx),
                            y=_to_plot(fy, gy),
                            zmin=0,
                            zmax=1,
                            colorscale=[(0, "rgba(150,150,150,0.55)"), (1, "rgba(0,0,0,0)")],
                            showscale=False,
                            hoverinfo="skip",
                            name="Excluded",
                        )
                    )

            if fx.is_categorical and fy.is_categorical:
                # The allowed combinations
                GX, GY = np.meshgrid(_levels(fx), _levels(fy))
                ok = _allowed(params, {ix: GX.ravel(), iy: GY.ravel()}, settings)
                fig.add_trace(_level_trace(GX.ravel()[ok], GY.ravel()[ok], False))
            else:
                # The lines along x at every level of y, and along y at every level of x
                xp, yp = level_lines(ix, iy)
                fig.add_trace(_level_trace(xp, yp, fx.is_continuous))
                yp, xp = level_lines(iy, ix)
                fig.add_trace(_level_trace(xp, yp, fy.is_continuous))

            # The runs on top
            fig.add_trace(_runs_trace(Y, factors, ix, iy, fixed, fixed_hover))
            fig.update_layout(xaxis=_axis_layout(fx, x), yaxis=_axis_layout(fy, y))

        if fixed_text:
            fig.update_layout(title=fixed_text, title_x=0.5)
        return fig
    finally:
        _clear_constraints(params)


def _prediction_variance(
    params: FixedStructureParameters | SplitkParameters | CostOptimalParameters,
    Minv: np.ndarray,
    x: str,
    y: str | None,
    n: int,
    fixed: Mapping[str, float | str] | None,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Computes the relative prediction variance x' Minv x on a grid over the factors `x` and `y`,
    for each set of a-priori variance ratios. The factors in `fixed` are at their value, and every
    grid point is the average over 64 random settings of the other factors (the same for every grid point),
    leaving out the settings that the constraints exclude.

    Parameters
    ----------
    params : Parameters
        The simulation parameters (fixed structure, split-plot or cost optimal).
    Minv : np.array(3d)
        The parameter estimation covariance matrices, one for each set of a-priori variance ratios.
        Columns beyond those of the model (covariates such as a time trend) are held at 0.
    x : str
        The name of the factor on the horizontal axis.
    y : str or None
        The name of the factor on the vertical axis. None gives the prediction variance along `x`.
    n : int
        The number of grid points along a continuous axis. A categorical axis has its levels.
    fixed : dict or None
        The values of factors that are not on an axis, as {factor name: value} in the factor's units.
    extra : np.array(1d) or None
        Normalized values to add to the grid along one continuous factor, such as the levels of the runs.

    Returns
    -------
    xs : np.array(1d)
        The plot coordinates along `x` (the level indices of a categorical factor).
    ys : np.array(1d) or None
        The plot coordinates along `y`, None along one factor.
    pv : np.array(3d) or np.array(2d)
        The relative prediction variance of shape (sets, len(ys), len(xs)), or (sets, len(xs))
        along one factor. NaN where the constraints exclude every setting.
    """
    try:
        factors = list(params.factors)
        fixed = dict(fixed or {})
        ix, iy = _shown_factors(factors, x, y)
        settings, _ = _other_settings(factors, ix, iy, _fixed_settings(factors, fixed, ix, iy), variance=True)
        Minv = np.asarray(Minv)

        # The grid points
        fx, fy = factors[ix], factors[iy]
        gx = _axis_grid(fx, n)
        ys: np.ndarray | None
        shape: tuple[int, ...]
        if ix == iy:
            if extra is not None and fx.is_continuous:
                gx = np.union1d(gx, extra)
            ys, shape, points = None, (len(gx),), {ix: gx}
        else:
            gy = _axis_grid(fy, n)
            GX, GY = np.meshgrid(gx, gy)  # rows: y, columns: x
            ys, shape, points = _to_plot(fy, gy), GX.shape, {ix: GX.ravel(), iy: GY.ravel()}

        # The relative prediction variance, averaged over the settings that the constraints allow, covariates held at 0
        pv = np.empty((len(Minv), int(np.prod(shape))))
        for sl, rows in _grid_rows(points, settings):
            Yenc = encode_design(rows, params.effect_types, params.coords)
            X = params.fn.Y2X(Yenc)
            if X.shape[1] < Minv.shape[-1]:
                X = np.hstack([X, np.zeros((len(X), Minv.shape[-1] - X.shape[1]))])
            v = np.sum((X @ Minv) * X, axis=-1)
            v[:, _excluded(params, Yenc)] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pv[:, sl] = np.nanmean(v.reshape(len(Minv), -1, len(settings)), axis=-1)

        return _to_plot(fx, gx), ys, pv.reshape(len(Minv), *shape)
    finally:
        _clear_constraints(params)


def _plot_prediction_variance(Y, params, Minv, x, y, n, fixed, titles):
    """
    Plots the relative prediction variance of
    :py:func:`_prediction_variance <pyoptex.doe.utils.evaluate._prediction_variance>`,
    one subplot per set of a-priori variance ratios, with the design's runs on top.

    Parameters
    ----------
    Y : pd.DataFrame
        The decoded and denormalized design.
    params : Parameters
        The simulation parameters (fixed structure, split-plot or cost optimal).
    Minv : np.array(3d)
        The parameter estimation covariance matrices, one for each set of a-priori variance ratios.
    x : str
        The name of the factor on the horizontal axis.
    y : str or None
        The name of the factor on the vertical axis, None along `x` only.
    n : int
        The number of grid points along a continuous axis.
    fixed : dict or None
        The values of factors that are not on an axis, as {factor name: value} in the factor's units.
    titles : list(str) or None
        The title of each subplot.

    Returns
    -------
    fig : :py:class:`plotly.graph_objects.Figure`
        The figure.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a denormalized and decoded dataframe"
    factors = list(params.factors)
    fixed = dict(fixed or {})
    ix, iy = _shown_factors(factors, x, y)
    fx, fy = factors[ix], factors[iy]

    # Along one continuous factor, the levels of the runs are on the grid, so that the runs sit on the curve
    extra = fx.normalize(Y[str(fx.name)].to_numpy(dtype=np.float64)) if ix == iy and fx.is_continuous else None
    xs, ys, pv = _prediction_variance(params, Minv, x, y, n, fixed, extra)

    # The labels
    averaged = [i for i in range(len(factors)) if i not in (ix, iy) and str(factors[i].name) not in fixed]
    fixed_text, fixed_hover = _fixed_labels(factors, fixed)
    word = "Average relative prediction variance" if len(averaged) > 0 else "Relative prediction variance"
    parts = [fixed_text] if fixed_text else []
    if len(averaged) > 0:
        parts.append("averaged over " + ", ".join(str(factors[i].name) for i in averaged))
    value = f"{word}: %{{{'y' if ix == iy else 'z'}:.3f}}<extra></extra>"

    # One shared colour scale (vertical axis along one factor) over the subplots
    finite = pv[np.isfinite(pv)]
    zlo, zhi = (float(finite.min()), float(finite.max())) if len(finite) > 0 else (np.nan, np.nan)
    colour = dict(colorscale="Blues", zmin=zlo, zmax=zhi, colorbar=dict(title=word.replace(" ", "<br>")))

    nsets = len(pv)
    curve_colour = "#2a78d6"
    fig = make_subplots(rows=nsets, cols=1, vertical_spacing=0.07, subplot_titles=titles)

    for i in range(nsets):
        if ix == iy:
            # Along one factor: the curve (points at the levels without runs for a categorical factor), the runs on it
            runs = _runs_trace(Y, factors, ix, iy, fixed, fixed_hover, lambda v, i=i: np.interp(v, xs, pv[i]))
            if fx.is_continuous:
                curve = go.Scatter(
                    x=xs,
                    y=pv[i],
                    mode="lines",
                    line=dict(color=curve_colour, width=2),
                    hovertemplate=f"{x}: %{{x:.4~g}}{fixed_hover}<br>{value}",
                )
            else:
                empty = ~np.isin(xs, runs.x)
                curve = go.Scatter(
                    x=xs[empty],
                    y=pv[i][empty],
                    text=[str(v) for v, e in zip(fx.levels, empty, strict=True) if e],
                    mode="markers",
                    marker=dict(size=8, color=curve_colour),
                    hovertemplate=f"{x}: %{{text}}{fixed_hover}<br>{value}",
                )
            fig.add_trace(curve.update(showlegend=False), row=i + 1, col=1)
            fig.add_trace(runs, row=i + 1, col=1)
            continue

        if np.isfinite(zhi):
            first = i == 0
            if fx.is_continuous and fy.is_continuous:
                # Filled contours at Plotly's own levels, labelled
                fig.add_trace(
                    go.Contour(
                        z=pv[i],
                        x=xs,
                        y=ys,
                        contours=dict(coloring="fill", showlabels=True, labelformat=".3~g"),
                        showscale=first,
                        hovertemplate=f"{x}: %{{x:.4~g}}<br>{y}: %{{y:.4~g}}{fixed_hover}<br>{value}",
                        **colour,
                    ),
                    row=i + 1,
                    col=1,
                )
            elif fx.is_categorical and fy.is_categorical:
                # One block per combination
                text = [[f"{x}: {lx}<br>{y}: {ly}" for lx in fx.levels] for ly in fy.levels]
                fig.add_trace(
                    go.Heatmap(
                        z=pv[i],
                        x=xs,
                        y=ys,
                        text=text,
                        showscale=first,
                        hovertemplate=f"%{{text}}{fixed_hover}<br>{value}",
                        **colour,
                    ),
                    row=i + 1,
                    col=1,
                )
            else:
                # One continuous and one categorical factor: a strip of filled contours at every level
                size = float(f"{(zhi - zlo) / 8:.1g}") if zhi > zlo else 1.0
                contours = dict(
                    coloring="fill", start=np.floor(zlo / size) * size, end=zhi, size=size, showlabels=False
                )
                fc = fx if fx.is_categorical else fy
                for k, level in enumerate(fc.levels):
                    band = np.linspace(k - 0.3, k + 0.3, 3)
                    if fx.is_categorical:
                        strip = dict(x=band, y=ys, z=np.tile(pv[i][:, k, np.newaxis], (1, len(band))))
                        where = f"{x}: {level}<br>{y}: %{{y:.4~g}}"
                    else:
                        strip = dict(x=xs, y=band, z=np.tile(pv[i][k], (len(band), 1)))
                        where = f"{x}: %{{x:.4~g}}<br>{y}: {level}"
                    fig.add_trace(
                        go.Contour(
                            contours=contours,
                            autocontour=False,
                            showscale=first and k == 0,
                            hovertemplate=f"{where}{fixed_hover}<br>{value}",
                            **strip,
                            **colour,
                        ),
                        row=i + 1,
                        col=1,
                    )

        # The runs on top
        fig.add_trace(_runs_trace(Y, factors, ix, iy, fixed, fixed_hover), row=i + 1, col=1)

    fig.update_xaxes(**_axis_layout(fx, x))
    if ix == iy:
        fig.update_yaxes(title=word, range=[0, (zhi if np.isfinite(zhi) else 1) * 1.08])
    else:
        fig.update_yaxes(**_axis_layout(fy, y))

    if not np.isfinite(zhi):
        fig.add_annotation(
            text="The constraints exclude every setting on this grid",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    if len(parts) > 0:
        fig.update_layout(title=" · ".join(parts), title_x=0.5)
    if nsets > 1:
        fig.update_layout(height=400 * nsets)

    return fig
