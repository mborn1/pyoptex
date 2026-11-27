import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, box_behnken
from pyoptex.doe.classical.evaluate import estimation_variance
from pyoptex.utils.model import model2Y2X, partial_rsm_names


class TestBoxBehnken:
    def test_three_factors(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors, n_center=3)
        assert isinstance(Y, pd.DataFrame)
        # 3 pairs * 4 + 3 center = 15
        assert len(Y) == 15

    def test_four_factors(self):
        factors = [Factor(f"X{i}") for i in range(4)]
        Y = box_behnken(factors, n_center=1)
        # 6 pairs * 4 + 1 center = 25
        assert len(Y) == 25

    @pytest.mark.parametrize("k, expected", [(3, 12), (4, 24), (5, 40)])
    def test_textbook_run_counts(self, k, expected):
        """For 3, 4 and 5 factors the design matches the tabulated BBD sizes (excluding center points)."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = box_behnken(factors, n_center=0)
        assert len(Y) == expected

    @pytest.mark.parametrize("k, expected", [(3, 15), (4, 27), (5, 46)])
    def test_default_center_points(self, k, expected):
        """Default center points follow the tabulated designs of Box & Behnken (1960)."""
        factors = [Factor(f"X{i}") for i in range(k)]
        assert len(box_behnken(factors)) == expected

    @pytest.mark.parametrize(
        "k, expected",
        [(3, 15), (4, 27), (5, 46), (6, 54), (7, 62), (9, 130), (10, 170), (11, 188), (12, 204), (16, 396)],
    )
    def test_paper_run_counts(self, k, expected):
        """Total runs N of the designs in Table 4 of Box & Behnken (1960), center points included."""
        assert len(box_behnken([Factor(f"X{i}") for i in range(k)])) == expected

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16])
    def test_balanced_and_quadratic_estimable(self, k):
        """Every factor is varied equally often, every pair of factors is varied together,
        and the full quadratic model is estimable."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = box_behnken(factors, n_center=3)
        X = Y.to_numpy(dtype=np.float64)
        assert set(np.unique(X)) == {-1.0, 0.0, 1.0}

        nonzero = X != 0
        assert len(set(nonzero.sum(axis=0))) == 1
        together = nonzero.T.astype(int) @ nonzero.astype(int)
        assert np.all(together[~np.eye(k, dtype=bool)] > 0)

        Y2X = model2Y2X(partial_rsm_names({str(f.name): "quad" for f in factors}), factors)
        assert np.all(np.isfinite(estimation_variance(Y, factors, Y2X)))

    def test_six_factors_uses_blocks_of_three(self):
        """The textbook 6-factor design has 48 non-center runs, not the 60 of all pairs."""
        X = box_behnken([Factor(f"X{i}") for i in range(6)], n_center=0).to_numpy()
        assert X.shape == (48, 6)
        np.testing.assert_array_equal(np.sum(X != 0, axis=1), 3)

    def test_eleven_factors_half_fractions(self):
        """For 11 factors every block is the half fraction of a 2^5 with x1 x2 x3 x4 x5 = +1."""
        X = box_behnken([Factor(f"X{i}") for i in range(11)], n_center=0).to_numpy()
        assert X.shape == (11 * 16, 11)
        np.testing.assert_array_equal(np.sum(X != 0, axis=1), 5)
        np.testing.assert_array_equal(np.prod(np.where(X != 0, X, 1), axis=1), 1.0)

    def test_no_tabulated_default_raises(self):
        factors = [Factor(f"X{i}") for i in range(8)]
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            box_behnken(factors)
        assert len(box_behnken(factors, n_center=2)) == 28 * 4 + 2

    def test_three_levels_only(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors)
        for col in Y.columns:
            assert set(Y[col].unique()) <= {-1.0, 0.0, 1.0}

    def test_no_corner_points(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors)
        Ynorm = Y.copy()
        for f in factors:
            Ynorm[str(f.name)] = f.normalize(Ynorm[str(f.name)])
        vals = Ynorm.values
        for row in vals:
            nonzero = np.count_nonzero(row)
            assert nonzero <= 2, "Box-Behnken should have at most 2 non-zero factors per run"

    def test_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5), Factor("C", min=-3, max=3)]
        Y = box_behnken(factors)
        assert Y["A"].min() >= 10.0
        assert Y["A"].max() <= 20.0

    def test_too_few_factors_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(AssertionError, match="at least 3 factors"):
            box_behnken(factors)

    def test_categorical_raises(self):
        factors = [
            Factor("A"),
            Factor("B"),
            Factor("C", type="categorical", levels=["X", "Y"]),
        ]
        with pytest.raises(AssertionError, match="must be continuous"):
            box_behnken(factors)
