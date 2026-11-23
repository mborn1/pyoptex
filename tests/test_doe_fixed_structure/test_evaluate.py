import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_parameters, default_fn
from pyoptex.doe.fixed_structure.cov import cov_time_trend
from pyoptex.doe.fixed_structure.evaluate import estimation_variance_matrix, power, prediction_variance
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.doe.utils.evaluate import plot_design_space
from pyoptex.utils.model import model2Y2X, partial_rsm_names

FACTORS = [
    Factor("x"),
    Factor("y"),
    Factor("C", type="categorical", levels=["L1", "L2", "L3"]),
]

# A design with a repeated run
Y_DESIGN = pd.DataFrame(
    [(-1, -1, "L1"), (1, -1, "L2"), (-1, 1, "L3"), (0, 0, "L1"), (0, 0, "L1"), (1, 0, "L3")],
    columns=["x", "y", "C"],
)


def _params(factors, script=None, nruns=6, cov=None):
    """The parameters of a D-optimal main effects design, with the constraints in the script and a covariate."""
    constraints = parse_constraints_script(script, factors) if script is not None else None
    Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
    return create_parameters(factors, default_fn(factors, Dopt(cov=cov), Y2X, constraints=constraints), nruns)


def _run_counts(fig):
    """The number of runs at each dot, by its (x, y)."""
    runs = next(t for t in fig.data if t.name == "Runs")
    return {(x, y): int(n) for x, y, n in zip(runs.x, runs.y, runs.text, strict=True)}


def _excluded(fig):
    """The grid of the shaded region and which of its points are excluded, or None when nothing is shaded."""
    excluded = next((t for t in fig.data if t.name == "Excluded"), None)
    if excluded is None:
        return None
    gx, gy = np.meshgrid(excluded.x, excluded.y)
    return gx, gy, np.asarray(excluded.z) == 0


def _level_points(fig):
    """The (x, y) points of the level lines, without the gaps where the constraints exclude them."""
    traces = [t for t in fig.data if t.name == "Levels"]
    x = np.concatenate([np.asarray(t.x, dtype=float) for t in traces])
    y = np.concatenate([np.asarray(t.y, dtype=float) for t in traces])
    keep = ~np.isnan(x) & ~np.isnan(y)
    return x[keep], y[keep]


def test_plot_design_space_runs():
    """One dot per combination, labelled with its number of runs; with fixed, only the runs at that value."""
    params = _params(FACTORS)
    fig = plot_design_space(Y_DESIGN, params, "x", "y")
    assert _run_counts(fig) == {(-1, -1): 1, (1, -1): 1, (-1, 1): 1, (0, 0): 2, (1, 0): 1}
    fig = plot_design_space(Y_DESIGN, params, "x", "y", fixed={"C": "L1"})
    assert _run_counts(fig) == {(-1, -1): 1, (0, 0): 2}


def test_plot_design_space_constraints():
    """The excluded region and the level lines follow the constraints in the parameters."""
    # The excluded region is exactly the corner, and no level line passes through it
    fig = plot_design_space(Y_DESIGN, _params(FACTORS, "(`x` > 0.5) & (`y` > 0.5)"), "x", "y")
    gx, gy, excluded = _excluded(fig)
    np.testing.assert_array_equal(excluded, (gx > 0.5) & (gy > 0.5))
    x, y = _level_points(fig)
    assert not np.any((x > 0.5) & (y > 0.5))

    # A rule on a factor that is not shown: C = L1 allows every point, unless C is fixed at L2
    params = _params(FACTORS, '(`C` == "L2") & (`x` > 0.5)')
    assert _excluded(plot_design_space(Y_DESIGN, params, "x", "y")) is None
    gx, gy, excluded = _excluded(plot_design_space(Y_DESIGN, params, "x", "y", fixed={"C": "L2"}))
    np.testing.assert_array_equal(excluded, gx > 0.5)

    # A categorical axis shows the level names; the L2 line stops at x = 0.5, the others cover the range
    fig = plot_design_space(Y_DESIGN, params, "x", "C")
    assert fig.layout.yaxis.ticktext == ("L1", "L2", "L3")
    x, y = _level_points(fig)
    assert 0.49 < x[y == 1].max() <= 0.5
    for level in (0, 2):
        assert x[y == level].min() == -1 and x[y == level].max() == 1


def test_plot_design_space_mixture():
    """The mixture rule that default_fn adds shades where the components sum to more than 1, over their levels."""
    factors = [Factor("A", type="mixture", levels=[0, 0.4, 0.8]), Factor("B", type="mixture", levels=[0, 0.4, 0.8])]
    Y = pd.DataFrame([(0, 0), (0.8, 0), (0, 0.8), (0.4, 0.4)], columns=["A", "B"])
    gx, gy, excluded = _excluded(plot_design_space(Y, _params(factors, nruns=4), "A", "B"))
    np.testing.assert_array_equal(excluded, gx + gy > 1)
    for g in (gx, gy):
        assert g.min() == 0 and g.max() == 0.8


def test_prediction_variance():
    """
    The relative prediction variance x0' (X'X)^-1 x0 on the grid, with a time trend held at 0,
    NaN where the constraints exclude every setting.
    """
    runs = [(x, y, c) for x in (-1, 1) for y in (-1, 1) for c in ("L1", "L2", "L3") if not (c == "L2" and x > 0.5)]
    Y = pd.DataFrame(runs, columns=["x", "y", "C"])
    params = _params(FACTORS, '(`C` == "L2") & (`x` > 0.5)', nruns=len(Y), cov=cov_time_trend(2, len(Y)))

    # By hand: intercept, x, y, the effect coding of C and the time trend (first half -1, second half 1);
    # at the center, with C fixed at L1 and the time trend at 0
    effects = {"L1": [1, 0], "L2": [0, 1], "L3": [-1, -1]}
    t = np.repeat([-1.0, 1.0], len(runs) // 2)
    X = np.array([[1, x, y, *effects[c], ti] for (x, y, c), ti in zip(runs, t, strict=True)], dtype=float)
    x0 = np.array([1, 0, 0, 1, 0, 0], dtype=float)
    expected = x0 @ np.linalg.inv(X.T @ X) @ x0

    n = 9
    xs, ys, pv = prediction_variance(Y, params, "x", "y", n=n, fixed={"C": "L1"})
    assert pv.shape == (1, n, n)
    assert xs[n // 2] == 0 and ys[n // 2] == 0
    np.testing.assert_allclose(pv[0, n // 2, n // 2], expected)

    # Only the grid points at L2 with x > 0.5 are excluded (averaged over y, which the rule does not involve)
    xs, ys, pv = prediction_variance(Y, params, "x", "C", n=n)
    np.testing.assert_array_equal(ys, [0, 1, 2])
    np.testing.assert_array_equal(np.isnan(pv[0]), (ys[:, np.newaxis] == 1) & (xs[np.newaxis, :] > 0.5))


def test_level_combinations():
    """
    A rule written for the levels of a factor that is not on an axis, which its random settings never meet,
    through its levels.
    """
    rule = "(`x` > 0.5) & (`y` > -1)"
    runs = [(x, y, c) for x in (-1, 1) for y in (-1, 1) for c in ("L1", "L2", "L3") if not (x > 0.5 and y > -1)]
    Y = pd.DataFrame(runs, columns=["x", "y", "C"])
    params = _params(FACTORS, rule, nruns=len(Y))

    # At x = 1, only y at its lowest level is allowed: there, by hand (C fixed at L1)
    effects = {"L1": [1, 0], "L2": [0, 1], "L3": [-1, -1]}
    X = np.array([[1, x, y, *effects[c]] for x, y, c in runs], dtype=float)
    x0 = np.array([1, 1, -1, 1, 0], dtype=float)
    xs, _, pv = prediction_variance(Y, params, "x", fixed={"C": "L1"})
    assert np.all(np.isfinite(pv))
    np.testing.assert_allclose(pv[0, xs == 1], x0 @ np.linalg.inv(X.T @ X) @ x0)

    # The level lines along x reach x = 1
    x, y = _level_points(plot_design_space(Y, params, "x", "C"))
    for level in (0, 1, 2):
        assert x[y == level].max() == 1


ddfs = pytest.mark.parametrize("ddf", ["kenward-roger", "satterthwaite"])


def _expected_power(k, df, noncentrality, alpha=0.05):
    """The power of an F-test with k and df degrees of freedom."""
    return stats.ncf.sf(stats.f.ppf(1 - alpha, k, df), k, df, noncentrality)


def _power_params(factors, model, nruns):
    """The parameters of a design with this model."""
    return create_parameters(factors, default_fn(factors, Dopt(), model2Y2X(model, factors)), nruns)


def _split_plot(nplots):
    """W per whole plot of 4 runs (-1, 1, -1, 1, ...) with ratio 1, S at -1, 1, -1, 1 within each whole plot."""
    factors = [Factor("W", RandomEffect(np.repeat(np.arange(nplots), 4), ratio=1)), Factor("S")]
    Y = pd.DataFrame({"W": np.repeat(np.resize([-1, 1], nplots), 4), "S": np.tile([-1, 1, -1, 1], nplots)})
    return factors, Y


@ddfs
def test_power_factorial(ddf):
    """2^3 full factorial with main effects, completely randomized: 4 residual df and noncentrality 8 * size^2."""
    factors = [Factor(name) for name in "ABC"]
    Y = pd.DataFrame(list(itertools.product([-1, 1], repeat=3)), columns=list("ABC"))
    model = partial_rsm_names({name: "lin" for name in "ABC"})
    params = _power_params(factors, model, len(Y))

    pw = power(Y, params, model, ddf=ddf)
    assert list(pw.columns) == ["A", "B", "C"]
    np.testing.assert_allclose(pw.to_numpy(), _expected_power(1, 4, 8))

    # Without an effect, the power is the significance level
    np.testing.assert_allclose(power(Y, params, model, sizes=0, ddf=ddf).to_numpy(), 0.05, rtol=1e-12)


@ddfs
def test_power_categorical(ddf):
    """A 3-level categorical factor with 4 runs per level: one F-test on its 2 columns with 9 residual df."""
    factors = [Factor("C", type="categorical", levels=["L1", "L2", "L3"])]
    Y = pd.DataFrame({"C": ["L1", "L2", "L3"] * 4})
    model = partial_rsm_names({"C": "lin"})
    params = _power_params(factors, model, len(Y))

    # The coefficients (1, -1) of the two columns
    Phi = estimation_variance_matrix(Y, params)[0]
    b = np.array([1.0, -1.0])
    pw = power(Y, params, model, ddf=ddf)
    assert list(pw.columns) == ["C"]
    np.testing.assert_allclose(pw.to_numpy(), _expected_power(2, 9, b @ np.linalg.solve(Phi[1:, 1:], b)))


@ddfs
def test_power_split_plot(ddf):
    """
    Balanced split-plot with 4 whole plots: W is tested against the whole plots (2 df, variance (1 + 1/4) / 4),
    S and W * S within them (10 df, variance 1/16).
    """
    factors, Y = _split_plot(4)
    model = partial_rsm_names({"W": "tfi", "S": "tfi"})
    pw = power(Y, _power_params(factors, model, len(Y)), model, ddf=ddf)
    assert list(pw.columns) == ["W", "S", "W * S"]
    np.testing.assert_allclose(pw["W"], _expected_power(1, 2, 3.2))
    np.testing.assert_allclose(pw[["S", "W * S"]].to_numpy(), _expected_power(1, 10, 16))


@ddfs
def test_power_unestimable_whole_plot_variance(ddf):
    """
    With 2 whole plots, the whole-plot variance cannot be estimated: W, which depends on it, has no power.
    S does not depend on it (5 df within the whole plots, variance 1/8).
    """
    factors, Y = _split_plot(2)
    model = partial_rsm_names({"W": "lin", "S": "lin"})
    pw = power(Y, _power_params(factors, model, len(Y)), model, ddf=ddf)
    assert np.isnan(pw.loc[0, "W"])
    np.testing.assert_allclose(pw.loc[0, "S"], _expected_power(1, 5, 8))
