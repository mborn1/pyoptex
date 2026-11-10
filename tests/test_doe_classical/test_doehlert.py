import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, doehlert
from pyoptex.doe.classical.doehlert import _regular_simplex
from pyoptex.doe.classical.evaluate import estimation_variance
from pyoptex.utils.model import model2Y2X, partial_rsm_names


class TestDoehlert:
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
    def test_run_count(self, k):
        Y = doehlert([Factor(f"X{i}") for i in range(k)])
        assert isinstance(Y, pd.DataFrame)
        assert Y.shape == (k**2 + k + 1, k)

    @pytest.mark.parametrize(
        "k, expected_levels",
        [(2, [5, 3]), (3, [5, 7, 3]), (4, [5, 7, 7, 3]), (5, [5, 7, 7, 7, 3])],
    )
    def test_number_of_levels(self, k, expected_levels):
        """Published level counts: 5 for the first factor, 3 for the last, 7 in between."""
        Y = doehlert([Factor(f"X{i}") for i in range(k)])
        assert [Y[c].round(10).nunique() for c in Y.columns] == expected_levels

    def test_levels_equally_spaced_within_range(self):
        Y = doehlert([Factor("A", min=0, max=10), Factor("B", min=0, max=6), Factor("C", min=0, max=2)])
        np.testing.assert_allclose(np.sort(Y["A"].unique()), [0, 2.5, 5, 7.5, 10])
        np.testing.assert_allclose(np.sort(Y["B"].round(10).unique()), [0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(np.sort(Y["C"].unique()), [0, 1, 2])

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_regular_simplex(self, k):
        """All vertices are at unit distance from each other."""
        V = _regular_simplex(k)
        dists = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
        np.testing.assert_allclose(dists[~np.eye(k + 1, dtype=bool)], 1.0)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_uniform_shell(self, k):
        """Before scaling to the factor ranges, all non-center runs lie on the unit sphere."""
        V = _regular_simplex(k)
        shell = np.array([V[i] - V[j] for i in range(k + 1) for j in range(k + 1) if i != j])
        np.testing.assert_allclose(np.linalg.norm(shell, axis=1), 1.0)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_full_quadratic_model_estimable(self, k):
        factors = [Factor(f"X{i}") for i in range(k)]
        Y2X = model2Y2X(partial_rsm_names({str(f.name): "quad" for f in factors}), factors)
        assert np.all(np.isfinite(estimation_variance(doehlert(factors), factors, Y2X)))

    def test_center_runs(self):
        Y = doehlert([Factor("A"), Factor("B"), Factor("C")], n_center=3)
        assert len(Y) == 12 + 3
        np.testing.assert_array_equal(Y.iloc[:3].values, 0.0)

    def test_categorical_raises(self):
        with pytest.raises(AssertionError, match="must be continuous"):
            doehlert([Factor("A"), Factor("B", type="categorical", levels=["X", "Y"])])

    def test_one_factor_raises(self):
        with pytest.raises(AssertionError, match="at least 2 factors"):
            doehlert([Factor("A")])
