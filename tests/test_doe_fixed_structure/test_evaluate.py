import itertools
import warnings

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
from pyoptex.doe.utils.power import _power, _power_table
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
@pytest.mark.parametrize(
    ("runs", "df"), [(["L1", "L2", "L3"] * 4, 9), (["L1", "L2", "L3", "L1", "L2"], 2)], ids=["9df", "2df"]
)
def test_power_categorical(ddf, runs, df):
    """
    A 3-level categorical factor, completely randomized: one F-test on its 2 columns, with 9 residual df
    (4 runs per level) or 2 (L1, L2, L3, L1, L2: Kenward-Roger's A2 equals the number of columns).
    """
    factors = [Factor("C", type="categorical", levels=["L1", "L2", "L3"])]
    Y = pd.DataFrame({"C": runs})
    model = partial_rsm_names({"C": "lin"})
    params = _power_params(factors, model, len(Y))

    # The coefficients (1, -1) of the two columns
    Phi = estimation_variance_matrix(Y, params)[0]
    b = np.array([1.0, -1.0])
    pw = power(Y, params, model, ddf=ddf)
    assert list(pw.columns) == ["C"]
    np.testing.assert_allclose(pw.to_numpy(), _expected_power(2, df, b @ np.linalg.solve(Phi[1:, 1:], b)))


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
def test_power_without_model(ddf):
    """Without a model, each column of Y2X is tested on its own: for main effects, the same powers as the model."""
    factors, Y = _split_plot(4)
    model = partial_rsm_names({"W": "lin", "S": "lin"})
    params = _power_params(factors, model, len(Y))
    by_column = power(Y, params, ddf=ddf)
    assert list(by_column.columns) == ["0", "1", "2"]  # the intercept, W and S
    np.testing.assert_allclose(by_column.iloc[:, 1:].to_numpy(), power(Y, params, model, ddf=ddf).to_numpy())


def test_power_sizes_by_name():
    """
    Sizes as a Series in another order than the terms give the same powers as the matching list.
    A key that names no tested term (W*S without spaces) raises an error.
    """
    factors, Y = _split_plot(4)
    model = partial_rsm_names({"W": "tfi", "S": "tfi"})
    params = _power_params(factors, model, len(Y))
    by_name = power(Y, params, model, sizes=pd.Series({"W * S": 2, "W": 1, "S": 0.5}))
    np.testing.assert_allclose(by_name.to_numpy(), power(Y, params, model, sizes=[1, 0.5, 2]).to_numpy())
    assert len(np.unique(by_name.to_numpy())) == 3

    with pytest.raises(AssertionError, match=r"unknown tests W\*S; the tests are W, S, W \* S"):
        power(Y, params, model, sizes={"W * S": 2, "W": 1, "S": 0.5, "W*S": 2})


def _mixed_model(groupings, ratios):
    """V^-1 and the derivatives of V (I and ZZ' per grouping) for V = I + sum of ratio * ZZ'."""
    n = len(groupings[0])
    Gs = [np.eye(n), *(np.equal.outer(Z, Z).astype(float) for Z in groupings)]
    V = Gs[0] + sum(r * G for r, G in zip(ratios, Gs[1:], strict=True))
    return np.linalg.inv(V), Gs


@ddfs
def test_power_estimable_variance(ddf):
    """
    A term has power only if the design can estimate the variance components its test needs:
    - With 2 whole plots, the whole-plot variance cannot be estimated: W, which depends on it, has no power.
      S does not depend on it (5 df within the whole plots, variance 1/8).
    - A whole-plot ratio of 1e5 shrinks the information of the whole-plot variance, but it is still estimable:
      W is tested against the 6 whole plots, less the intercept and W, F(1, 4).
    - In a saturated split-plot (6 runs, 6 columns), no variance component can be estimated, so no term has power.
      At ratios other than 1, the information is rounding noise instead of exactly 0.
    """
    # 2 whole plots
    factors, Y = _split_plot(2)
    model = partial_rsm_names({"W": "lin", "S": "lin"})
    pw = power(Y, _power_params(factors, model, len(Y)), model, ddf=ddf)
    assert np.isnan(pw.loc[0, "W"])
    np.testing.assert_allclose(pw.loc[0, "S"], _expected_power(1, 5, 8))

    # A ratio of 1e5
    Z = np.repeat(np.arange(6), 4)
    W = np.repeat([-1, 1, -1, 1, 0, 0], 4)
    S = np.tile([-1, 1, 1, -1], 6)
    X = np.column_stack((np.ones(len(Z)), W, S)).astype(float)
    Vinv, Gs = _mixed_model([Z], [1e5])
    pw, _ = _power(X, Vinv, Gs, [np.array([1])], np.ones(1), 0.05, ddf)
    Phi = np.linalg.inv(X.T @ Vinv @ X)
    np.testing.assert_allclose(pw[0], _expected_power(1, 4, 1 / Phi[1, 1]))

    # Saturated: 3 whole plots of 2 runs, W a 3-level categorical whole-plot factor (effect coding), S at -1, 1
    Z = np.repeat(np.arange(3), 2)
    W1, W2 = np.repeat([1, 0, -1], 2), np.repeat([0, 1, -1], 2)
    S = np.tile([-1, 1], 3)
    X = np.column_stack((np.ones(6), W1, W2, S, W1 * S, W2 * S)).astype(float)
    terms = [np.array([1, 2]), np.array([3]), np.array([4, 5])]
    for ratio in (1, 0.5, 10):
        Vinv, Gs = _mixed_model([Z], [ratio])
        assert np.all(np.isnan(_power(X, Vinv, Gs, terms, np.ones(3), 0.05, ddf)[0])), f"ratio {ratio}"


def test_power_kenward_roger_undefined_df():
    """
    In a very small design, Kenward-Roger's df is negative (pbkrtest: -0.12) for the test of x1 and x2:
    NaN with a warning that names it. x3 has a defined df. Satterthwaite gives a value (df 2), without a warning.
    In another one, Kenward-Roger's df of a 3-column term is 1.85, but its F-scaling is negative, and with it
    the noncentrality: NaN with the warning.
    """
    Z = np.array([0, 1, 2, 2, 2, 3, 3, 3])
    x1 = [1, 0, 1, 1, 1, -1, -1, -1]
    x2 = [0, 0, 1, 1, -1, 0, 1, 1]
    x3 = [0, -1, -1, 0, 1, 1, 1, -1]
    x4 = [1, 1, 1, 0, 1, 0, 0, 0]
    x5 = [1, 0, -1, -1, -1, -1, -1, -1]
    X = np.column_stack((np.ones(len(Z)), x1, x2, x3, x4, x5)).astype(float)
    Vinv, _ = _mixed_model([Z], [1])
    tests = [("x1 and x2", np.array([1, 2])), ("x3", np.array([3]))]

    with pytest.warns(UserWarning, match=r"Power is not defined for x1 and x2: ") as record:
        pw = _power_table(tests, X, [Z], Vinv[np.newaxis], 1, 0.05, "kenward-roger")
    assert len(record) == 1
    assert np.isnan(pw.loc[0, "x1 and x2"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pw = _power_table(tests, X, [Z], Vinv[np.newaxis], 1, 0.05, "satterthwaite")
    np.testing.assert_allclose(pw.loc[0, "x1 and x2"], 0.100441, atol=1e-5)

    # A negative noncentrality
    Z = np.array([0, 1, 1, 1, 1, 2, 3, 3])
    u1 = [1, 1, 0, 0, 1, 1, -1, 1]
    u2 = [1, -1, 1, 1, -1, 1, 1, 1]
    u3 = [-1, -1, 1, 1, -1, -1, 0, 1]
    X = np.column_stack((np.ones(len(Z)), u1, u2, u3)).astype(float)
    Vinv, _ = _mixed_model([Z], [1])
    with pytest.warns(UserWarning, match=r"Power is not defined for u1 to u3: "):
        pw = _power_table([("u1 to u3", np.array([1, 2, 3]))], X, [Z], Vinv[np.newaxis], 1, 0.05, "kenward-roger")
    assert np.isnan(pw.loc[0, "u1 to u3"])


def test_power_kenward_roger_pbkrtest():
    """Kenward-Roger power of unbalanced split-plot designs, against pbkrtest 0.5.2 (lme4 at fixed ratios)."""
    # B: one whole-plot grouping, a 3-level categorical whole-plot factor (2 columns) and its interaction with S
    Z = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6])
    C1 = np.array([1, 1, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1])
    C2 = np.array([0, 0, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0])
    S = np.array([1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1])
    X = np.column_stack((np.ones(len(Z)), C1, C2, S, C1 * S, C2 * S)).astype(float)
    Vinv, Gs = _mixed_model([Z], [2])
    terms = [np.array([1, 2]), np.array([3]), np.array([4, 5])]
    pw, _ = _power(X, Vinv, Gs, terms, np.ones(3), 0.05, "kenward-roger")
    np.testing.assert_allclose(pw, [0.136384, 0.978210, 0.873589], atol=1e-5)

    # C: whole plots and subplots within them
    Zw = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    Zs = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9])
    W = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    Sp = np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 0])
    R = np.array([1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1])
    X = np.column_stack((np.ones(len(Zw)), W, Sp, R, W * Sp, Sp * R)).astype(float)
    Vinv, Gs = _mixed_model([Zw, Zs], [1, 0.5])
    terms = [np.array([i]) for i in range(1, 6)]
    pw, _ = _power(X, Vinv, Gs, terms, np.ones(5), 0.05, "kenward-roger")
    np.testing.assert_allclose(pw, [0.172227, 0.582855, 0.971032, 0.587741, 0.929311], atol=1e-5)
