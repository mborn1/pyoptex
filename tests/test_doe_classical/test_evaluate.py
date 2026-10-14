from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from pyoptex._seed import set_seed
from pyoptex.doe import fixed_structure
from pyoptex.doe.classical import Factor, central_composite, fractional_factorial, full_factorial, plackett_burman
from pyoptex.doe.classical.evaluate import (
    a_efficiency,
    alias_matrix,
    average_prediction_variance,
    d_efficiency,
    design_info,
    estimation_variance,
    g_efficiency,
    power,
    variance_inflation,
)
from pyoptex.doe.fixed_structure.evaluate import evaluate_metrics
from pyoptex.doe.fixed_structure.metric import Dopt, Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names


def _make_simple_design():
    """2^3 full factorial with main-effects model."""
    factors = [Factor("A"), Factor("B"), Factor("C")]
    Y = full_factorial(factors)
    model = partial_rsm_names({"A": "lin", "B": "lin", "C": "lin"})
    Y2X = model2Y2X(model, factors)
    return Y, factors, Y2X


def _make_rank_deficient_design():
    """Repeated one-level design for an intercept + linear-effect model."""
    factors = [Factor("A", levels=[-1, 1])]
    Y = pd.DataFrame({"A": [-1, -1]})
    model = partial_rsm_names({"A": "lin"})
    Y2X = model2Y2X(model, factors)
    return Y, factors, Y2X


def _make_three_level_design():
    """3^2 full factorial with main-effects model: X'X = diag(9, 6, 6)."""
    factors = [Factor("A"), Factor("B")]
    Y = full_factorial(factors, n_levels=3)
    model = partial_rsm_names({"A": "lin", "B": "lin"})
    Y2X = model2Y2X(model, factors)
    return Y, factors, Y2X


class TestDEfficiency:
    def test_orthogonal_design(self):
        """A 2^3 with main-effects model is exactly 100% D-efficient."""
        Y, factors, Y2X = _make_simple_design()
        d_eff = d_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(d_eff, 100.0, rtol=1e-10)

    def test_orthogonal_two_level_is_100(self):
        """An orthogonal +/-1 2^3 design with a main-effects model is exactly 100% D-efficient."""
        factors = [Factor(n, levels=[-1, 1]) for n in "ABC"]
        Y = full_factorial(factors)
        model = partial_rsm_names({"A": "lin", "B": "lin", "C": "lin"})
        Y2X = model2Y2X(model, factors)
        d_eff = d_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(d_eff, 100.0, rtol=1e-10)

    def test_three_level_known_value(self):
        """3^2 main-effects design: D-eff = 100 * det(diag(9, 6, 6))^(1/3) / 9."""
        Y, factors, Y2X = _make_three_level_design()
        d_eff = d_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(d_eff, 100.0 * (9 * 6 * 6) ** (1 / 3) / 9, rtol=1e-10)

    def test_large_plackett_burman_is_100(self):
        """PB48 with 47 main effects: orthogonal, so exactly 100%."""
        factors = [Factor(f"X{i}") for i in range(47)]
        Y = plackett_burman(factors)
        model = partial_rsm_names({str(f.name): "lin" for f in factors})
        Y2X = model2Y2X(model, factors)
        np.testing.assert_allclose(d_efficiency(Y, factors, Y2X), 100.0, rtol=1e-10)

    def test_no_overflow_for_large_model(self):
        """Saturated 2^8 with all interactions: det(X'X) = 256^256 overflows, the log-determinant does not."""
        names = [f"X{i}" for i in range(8)]
        factors = [Factor(n) for n in names]
        Y = full_factorial(factors)
        model = pd.DataFrame(list(product([0, 1], repeat=8)), columns=names)
        Y2X = model2Y2X(model, factors)
        np.testing.assert_allclose(d_efficiency(Y, factors, Y2X), 100.0, rtol=1e-10)


class TestAEfficiency:
    def test_orthogonal_design(self):
        """A 2^3 with main-effects model is exactly 100% A-efficient."""
        Y, factors, Y2X = _make_simple_design()
        a_eff = a_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(a_eff, 100.0, rtol=1e-10)

    def test_three_level_known_value(self):
        """3^2 main-effects design: A-eff = 100 * p / (n * trace(diag(1/9, 1/6, 1/6))) = 75%."""
        Y, factors, Y2X = _make_three_level_design()
        a_eff = a_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(a_eff, 75.0, rtol=1e-10)

    def test_returns_float(self):
        Y, factors, Y2X = _make_simple_design()
        result = a_efficiency(Y, factors, Y2X)
        assert isinstance(result, float)


class TestGEfficiency:
    def test_two_level_main_effects_is_100(self):
        """For a 2^3 with main-effects model the max prediction variance p/n sits at the corners (design points)."""
        set_seed(42)
        Y, factors, Y2X = _make_simple_design()
        np.testing.assert_allclose(g_efficiency(Y, factors, Y2X, n_samples=1000), 100.0, rtol=1e-10)

    def test_three_level_known_value(self):
        """3^2 main-effects design: max d(x) = 1/9 + 1/6 + 1/6 at the corners, so G = 3 / (9 * 4/9) = 75%."""
        set_seed(42)
        Y, factors, Y2X = _make_three_level_design()
        np.testing.assert_allclose(g_efficiency(Y, factors, Y2X, n_samples=1000), 75.0, rtol=1e-10)

    def test_at_most_100(self):
        set_seed(42)
        factors = [Factor(n) for n in "ABC"]
        Y = central_composite(factors, alpha="face-centered")
        Y2X = model2Y2X(partial_rsm_names({n: "quad" for n in "ABC"}), factors)
        g_eff = g_efficiency(Y, factors, Y2X, n_samples=2000)
        assert 0 < g_eff <= 100.0

    def test_deterministic_with_set_seed(self):
        factors = [Factor(n) for n in "ABC"]
        Y = central_composite(factors, alpha="face-centered")
        Y2X = model2Y2X(partial_rsm_names({n: "quad" for n in "ABC"}), factors)
        set_seed(1)
        g1 = g_efficiency(Y, factors, Y2X, n_samples=500)
        set_seed(1)
        g2 = g_efficiency(Y, factors, Y2X, n_samples=500)
        assert g1 == g2

    def test_rank_deficient_is_zero(self):
        Y, factors, Y2X = _make_rank_deficient_design()
        assert g_efficiency(Y, factors, Y2X, n_samples=100) == 0.0


class TestAveragePredictionVariance:
    def test_two_level_main_effects_known_value(self):
        """d(x) = (1 + x1^2 + x2^2 + x3^2) / 8 averages to (1 + 3 * 1/3) / 8 = 0.25 over the cube."""
        set_seed(42)
        Y, factors, Y2X = _make_simple_design()
        apv = average_prediction_variance(Y, factors, Y2X, n_samples=20000)
        np.testing.assert_allclose(apv, 0.25, rtol=0.02)

    def test_rank_deficient_is_infinite(self):
        Y, factors, Y2X = _make_rank_deficient_design()
        assert average_prediction_variance(Y, factors, Y2X, n_samples=100) == np.inf


class TestVarianceInflation:
    def test_orthogonal_design_is_one(self):
        Y, factors, Y2X = _make_simple_design()
        vif = variance_inflation(Y, factors, Y2X)
        assert np.isnan(vif[0])  # intercept
        np.testing.assert_allclose(vif[1:], 1.0, rtol=1e-10)

    def test_correlated_columns(self):
        """Two main effects with correlation r have VIF 1 / (1 - r^2)."""
        factors = [Factor("A"), Factor("B")]
        Y = pd.DataFrame({"A": [-1, -1, 1, 1, 1, -1], "B": [-1, -1, 1, 1, -1, 1]}, dtype=float)
        Y2X = model2Y2X(partial_rsm_names({"A": "lin", "B": "lin"}), factors)
        r = np.corrcoef(Y["A"], Y["B"])[0, 1]
        np.testing.assert_allclose(variance_inflation(Y, factors, Y2X)[1:], 1 / (1 - r**2), rtol=1e-10)

    def test_rank_deficient_is_infinite(self):
        factors = [Factor("A"), Factor("B")]
        Y = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, 1, -1, 1]}, dtype=float)
        Y2X = model2Y2X(partial_rsm_names({"A": "lin", "B": "lin"}), factors)
        assert np.all(np.isinf(variance_inflation(Y, factors, Y2X)[1:]))


class TestPower:
    def test_matches_noncentral_f(self):
        """For one parameter, the two-sided t-test power equals the noncentral F(1, n - p) power."""
        Y, factors, Y2X = _make_simple_design()
        pw = power(Y, factors, Y2X, effect_size=1.0, alpha=0.05)
        lam = 1.0 / (1 / 8)  # effect^2 / var, var = 1/8
        expected = stats.ncf.sf(stats.f.ppf(0.95, 1, 4), 1, 4, lam)
        np.testing.assert_allclose(pw, expected, rtol=1e-8)

    def test_zero_effect_gives_alpha(self):
        Y, factors, Y2X = _make_simple_design()
        np.testing.assert_allclose(power(Y, factors, Y2X, effect_size=0.0, alpha=0.1), 0.1, rtol=1e-8)

    def test_increases_with_effect_size(self):
        Y, factors, Y2X = _make_simple_design()
        small = power(Y, factors, Y2X, effect_size=0.5)
        large = power(Y, factors, Y2X, effect_size=2.0)
        assert np.all(large > small)

    def test_per_parameter_effect_size(self):
        Y, factors, Y2X = _make_simple_design()
        pw = power(Y, factors, Y2X, effect_size=np.array([0.0, 1.0, 1.0, 2.0]))
        assert pw[0] < pw[1] == pw[2] < pw[3]

    def test_saturated_design_is_nan(self):
        """Without degrees of freedom for the error there is no test."""
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors)
        Y2X = model2Y2X(partial_rsm_names({"A": "tfi", "B": "tfi"}), factors)  # 4 parameters, 4 runs
        assert np.all(np.isnan(power(Y, factors, Y2X)))


def _fixed_structure_metrics(Y, names, model, n_samples):
    """Dopt and Iopt from the optimal-design module, without random effects."""
    fs_factors = [fixed_structure.Factor(n) for n in names]
    Y2X = model2Y2X(model, fs_factors)
    params = fixed_structure.create_parameters(fs_factors, fixed_structure.default_fn(fs_factors, Dopt(), Y2X), len(Y))
    set_seed(42)
    return evaluate_metrics(Y, params, [Dopt(), Iopt(n=n_samples)])


class TestConsistencyWithOptimalDesignMetrics:
    """The classical metrics must give the same numbers as the optimizer's metrics on the same design."""

    def test_d_efficiency_matches_dopt(self):
        names = ["A", "B", "C"]
        factors = [Factor(n) for n in names]
        model = partial_rsm_names({n: "quad" for n in names})
        Y = central_composite(factors, alpha="face-centered")
        dopt, _ = _fixed_structure_metrics(Y, names, model, n_samples=100)
        d_eff = d_efficiency(Y, factors, model2Y2X(model, factors))
        np.testing.assert_allclose(d_eff, 100 * dopt / len(Y), rtol=1e-10)

    def test_average_prediction_variance_matches_iopt(self):
        names = ["A", "B", "C"]
        factors = [Factor(n) for n in names]
        model = partial_rsm_names({n: "quad" for n in names})
        Y = central_composite(factors, alpha="face-centered")
        _, iopt = _fixed_structure_metrics(Y, names, model, n_samples=2000)
        set_seed(42)
        apv = average_prediction_variance(Y, factors, model2Y2X(model, factors), n_samples=2000)
        np.testing.assert_allclose(apv, -iopt, rtol=1e-10)


class TestEstimationVariance:
    def test_shape(self):
        Y, factors, Y2X = _make_simple_design()
        var = estimation_variance(Y, factors, Y2X)
        assert var.ndim == 1

    def test_known_values(self):
        """For a 2^3 with main-effects model X'X = 8I, so every variance is 1/8."""
        Y, factors, Y2X = _make_simple_design()
        var = estimation_variance(Y, factors, Y2X)
        np.testing.assert_allclose(var, np.full(4, 1 / 8), rtol=1e-10)

    def test_orthogonal_equal_variance(self):
        """For an orthogonal 2^k with main-effects model, all main-effect variances should be equal."""
        Y, factors, Y2X = _make_simple_design()
        var = estimation_variance(Y, factors, Y2X)
        # Skip intercept (index 0), main effects should all be equal
        np.testing.assert_allclose(var[1:], var[1], rtol=1e-10)

    def test_rank_deficient_design_has_infinite_variance(self):
        Y, factors, Y2X = _make_rank_deficient_design()
        var = estimation_variance(Y, factors, Y2X)
        np.testing.assert_array_equal(var, np.full(2, np.inf))


class TestAliasMatrix:
    def test_shape(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = full_factorial(factors)

        model_primary = partial_rsm_names({"A": "lin", "B": "lin", "C": "lin"})
        model_potential = partial_rsm_names({"A": "tfi", "B": "tfi", "C": "tfi"})
        Y2X_primary = model2Y2X(model_primary, factors)
        Y2X_potential = model2Y2X(model_potential, factors)

        A = alias_matrix(Y, factors, Y2X_primary, Y2X_potential)
        assert A.ndim == 2
        assert A.shape[0] == 4  # intercept + 3 main effects
        assert A.shape[1] == 7  # intercept + 3 main + 3 interactions

    def test_resolution_iii(self):
        """In a 2^(3-1) design with C=AB, each main effect is fully aliased with one interaction."""
        factors = [Factor(n, levels=[-1, 1]) for n in "ABC"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})

        model_primary = partial_rsm_names({"A": "lin", "B": "lin", "C": "lin"})
        Y2X_primary = model2Y2X(model_primary, factors)

        # Potential terms = the two-factor interactions ONLY (AB, AC, BC),
        # so the alias matrix isolates main-effect <-> interaction confounding.
        interactions = pd.DataFrame([[1, 1, 0], [1, 0, 1], [0, 1, 1]], columns=["A", "B", "C"])
        Y2X_potential = model2Y2X(interactions, factors)

        A = alias_matrix(Y, factors, Y2X_primary, Y2X_potential)
        assert A.shape == (4, 3)  # [intercept, A, B, C] x [AB, AC, BC]

        # Each main effect (rows 1..3) is fully aliased with exactly one interaction.
        for r in range(1, 4):
            row = np.abs(A[r])
            np.testing.assert_allclose(np.max(row), 1.0, atol=1e-10)
            np.testing.assert_allclose(np.sum(row), 1.0, atol=1e-10)
        # The intercept (row 0) is not aliased with any two-factor interaction.
        np.testing.assert_allclose(A[0], 0.0, atol=1e-10)

    def test_rank_deficient_primary_model_returns_undefined_aliases(self):
        Y, factors, Y2X_primary = _make_rank_deficient_design()
        Y2X_potential = model2Y2X(pd.DataFrame([[1]], columns=["A"]), factors)
        A = alias_matrix(Y, factors, Y2X_primary, Y2X_potential)
        assert A.shape == (2, 1)
        assert np.all(np.isnan(A))


class TestDesignInfo:
    def test_keys(self):
        set_seed(42)
        Y, factors, Y2X = _make_simple_design()
        info = design_info(Y, factors, Y2X, n_samples=500)
        assert "n_runs" in info
        assert "n_params" in info
        assert "d_efficiency" in info
        assert "a_efficiency" in info
        assert "estimation_variance" in info
        assert info["n_runs"] == 8  # 2^3 default full factorial
        assert info["n_params"] == 4
        np.testing.assert_allclose(info["d_efficiency"], 100.0, rtol=1e-10)
        np.testing.assert_allclose(info["a_efficiency"], 100.0, rtol=1e-10)
        np.testing.assert_allclose(info["estimation_variance"], np.full(4, 1 / 8), rtol=1e-10)
        np.testing.assert_allclose(info["g_efficiency"], 100.0, rtol=1e-10)
        np.testing.assert_allclose(info["average_prediction_variance"], 0.25, rtol=0.05)

    def test_matches_individual_metrics(self):
        Y, factors, Y2X = _make_three_level_design()
        set_seed(42)
        info = design_info(Y, factors, Y2X, n_samples=100)
        assert info["d_efficiency"] == d_efficiency(Y, factors, Y2X)
        assert info["a_efficiency"] == a_efficiency(Y, factors, Y2X)
        np.testing.assert_array_equal(info["estimation_variance"], estimation_variance(Y, factors, Y2X))
        set_seed(42)
        assert info["g_efficiency"] == g_efficiency(Y, factors, Y2X, n_samples=100)
        set_seed(42)
        assert info["average_prediction_variance"] == average_prediction_variance(Y, factors, Y2X, n_samples=100)

    def test_rank_deficient_design_reports_zero_efficiency_and_infinite_variance(self):
        set_seed(42)
        Y, factors, Y2X = _make_rank_deficient_design()
        info = design_info(Y, factors, Y2X, n_samples=100)
        assert info["n_runs"] == 2
        assert info["n_params"] == 2
        assert info["d_efficiency"] == 0.0
        assert info["a_efficiency"] == 0.0
        assert info["g_efficiency"] == 0.0
        assert info["average_prediction_variance"] == np.inf
        np.testing.assert_array_equal(info["estimation_variance"], np.full(2, np.inf))
