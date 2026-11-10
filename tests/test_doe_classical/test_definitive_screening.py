from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, definitive_screening
from pyoptex.doe.classical._paley import _galois_field

# Even numbers of factors with an implemented conference matrix of that order
_SUPPORTED = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32]


def _coded(k, **kwargs):
    """DSD for k default factors, as a +/-1 / 0 coded numpy array."""
    return definitive_screening([Factor(f"X{i}") for i in range(k)], **kwargs).to_numpy(dtype=np.float64)


def _second_order_columns(X):
    """All two-factor interaction columns and centered quadratic columns."""
    k = X.shape[1]
    tfi = [X[:, i] * X[:, j] for i, j in combinations(range(k), 2)]
    quad = [X[:, i] ** 2 - np.mean(X[:, i] ** 2) for i in range(k)]
    return np.column_stack(tfi), np.column_stack(quad)


class TestDefinitiveScreening:
    @pytest.mark.parametrize("m", _SUPPORTED)
    def test_supported_sizes(self, m):
        """2m + 1 runs and orthogonal main effects: X'X = 2(m - 1) I, i.e. C'C = (m - 1) I for each half."""
        X = _coded(m)
        assert X.shape == (2 * m + 1, m)
        np.testing.assert_allclose(X.T @ X, 2 * (m - 1) * np.eye(m), atol=1e-10)

    @pytest.mark.parametrize("m", _SUPPORTED)
    def test_main_effects_orthogonal_to_second_order(self, m):
        X = _coded(m)
        tfi, quad = _second_order_columns(X)
        np.testing.assert_allclose(X.T @ tfi, 0.0, atol=1e-10)
        np.testing.assert_allclose(X.T @ quad, 0.0, atol=1e-10)

    def test_three_levels_with_three_zeros_per_column(self):
        X = _coded(6)
        assert set(np.unique(X)) == {-1.0, 0.0, 1.0}
        np.testing.assert_array_equal(np.sum(X == 0, axis=0), 3)

    def test_two_factor_interactions_not_fully_aliased(self):
        tfi, _ = _second_order_columns(_coded(6))
        corr = np.corrcoef(tfi, rowvar=False)
        assert np.max(np.abs(corr[~np.eye(len(corr), dtype=bool)])) < 1 - 1e-10

    @pytest.mark.parametrize("k", [3, 5, 7, 9])
    def test_odd_number_of_factors(self, k):
        """An odd number of factors uses the next even order: 2(k + 1) + 1 runs."""
        X = _coded(k)
        assert X.shape == (2 * (k + 1) + 1, k)
        np.testing.assert_allclose(X.T @ X, 2 * k * np.eye(k), atol=1e-10)

    def test_order_28_from_gf27(self):
        """Order 28 uses the Paley construction over GF(27), a field with 3^3 elements."""
        X = _coded(28)
        assert X.shape == (57, 28)
        np.testing.assert_allclose(X.T @ X, 2 * 27 * np.eye(28), atol=1e-10)

    @pytest.mark.parametrize("p, e", [(3, 2), (3, 3), (5, 2)])
    def test_galois_field(self, p, e):
        """GF(p^e): every nonzero element has a multiplicative inverse; GF(27) uses x^3 + 2x + 1."""
        elems, mul = _galois_field(p, e)
        one = (1,) + (0,) * (e - 1)
        assert len(elems) == p**e
        assert all(any(mul(a, b) == one for b in elems[1:]) for a in elems[1:])
        if (p, e) == (3, 3):
            x = (0, 1, 0)
            assert mul(mul(x, x), x) == (2, 1, 0)  # x^3 = -2x - 1 = x + 2

    def test_fake_factors_add_runs(self):
        X = _coded(6, n_fake=2)
        assert X.shape == (17, 6)  # built for 8 factors

    @pytest.mark.parametrize("k, expected_order", [(21, 24), (22, 24), (35, 38), (36, 38)])
    def test_missing_order_uses_next_available(self, k, expected_order):
        """No conference matrix of order 22 exists and order 36 is not implemented."""
        X = _coded(k)
        assert X.shape == (2 * expected_order + 1, k)
        np.testing.assert_allclose(X.T @ X, 2 * (expected_order - 1) * np.eye(k), atol=1e-10)

    def test_center_runs(self):
        X = _coded(6, n_center=3)
        assert X.shape == (12 + 3, 6)
        np.testing.assert_array_equal(X[-3:], 0.0)

    def test_custom_range(self):
        factors = [Factor("A", min=10, max=20)] + [Factor(f"X{i}") for i in range(5)]
        Y = definitive_screening(factors)
        assert isinstance(Y, pd.DataFrame)
        assert set(Y["A"].unique()) == {10.0, 15.0, 20.0}

    def test_categorical_raises(self):
        factors = [Factor("A"), Factor("B"), Factor("C", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="must be continuous"):
            definitive_screening(factors)

    def test_too_few_factors_raises(self):
        with pytest.raises(AssertionError, match="at least 3 factors"):
            definitive_screening([Factor("A"), Factor("B")])
