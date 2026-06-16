import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.classical import Factor, fractional_factorial, full_factorial
from pyoptex.doe.classical.evaluate import (
    a_efficiency,
    alias_matrix,
    d_efficiency,
    design_info,
    estimation_variance,
)
from pyoptex.utils.model import model2Y2X, partial_rsm_names


def _make_simple_design():
    """Full factorial with main-effects model."""
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


class TestDEfficiency:
    def test_orthogonal_design(self):
        """A 2^3 with main-effects model should have high D-efficiency."""
        Y, factors, Y2X = _make_simple_design()
        d_eff = d_efficiency(Y, factors, Y2X)
        assert d_eff > 0

    def test_orthogonal_two_level_is_100(self):
        """An orthogonal +/-1 2^3 design with a main-effects model is exactly 100% D-efficient."""
        factors = [Factor(n, levels=[-1, 1]) for n in "ABC"]
        Y = full_factorial(factors)
        model = partial_rsm_names({"A": "lin", "B": "lin", "C": "lin"})
        Y2X = model2Y2X(model, factors)
        d_eff = d_efficiency(Y, factors, Y2X)
        np.testing.assert_allclose(d_eff, 100.0, rtol=1e-10)

    def test_positive(self):
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors)
        model = partial_rsm_names({"A": "lin", "B": "lin"})
        Y2X = model2Y2X(model, factors)
        d_eff = d_efficiency(Y, factors, Y2X)
        assert d_eff > 0


class TestAEfficiency:
    def test_orthogonal_design(self):
        Y, factors, Y2X = _make_simple_design()
        a_eff = a_efficiency(Y, factors, Y2X)
        assert a_eff > 0

    def test_returns_float(self):
        Y, factors, Y2X = _make_simple_design()
        result = a_efficiency(Y, factors, Y2X)
        assert isinstance(result, float)


class TestEstimationVariance:
    def test_shape(self):
        Y, factors, Y2X = _make_simple_design()
        var = estimation_variance(Y, factors, Y2X)
        assert var.ndim == 1

    def test_positive_diagonal(self):
        Y, factors, Y2X = _make_simple_design()
        var = estimation_variance(Y, factors, Y2X)
        assert np.all(var > 0)

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
        assert info["n_runs"] == 27  # 3^3 default full factorial

    def test_rank_deficient_design_reports_zero_efficiency_and_infinite_variance(self):
        set_seed(42)
        Y, factors, Y2X = _make_rank_deficient_design()
        info = design_info(Y, factors, Y2X, n_samples=100)
        assert info["n_runs"] == 2
        assert info["n_params"] == 2
        assert info["d_efficiency"] == 0.0
        assert info["a_efficiency"] == 0.0
        np.testing.assert_array_equal(info["estimation_variance"], np.full(2, np.inf))
