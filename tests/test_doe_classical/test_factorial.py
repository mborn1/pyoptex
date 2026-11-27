import numpy as np
import pandas as pd
import pytest

from pyoptex._seed import set_seed
from pyoptex.doe.classical import Factor, full_factorial, randomize


class TestFullFactorial:
    def test_two_continuous_default(self):
        """Default continuous factors use 2 levels (min, max)."""
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors)
        assert isinstance(Y, pd.DataFrame)
        assert list(Y.columns) == ["A", "B"]
        assert len(Y) == 4  # 2^2

    def test_three_continuous_default(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = full_factorial(factors)
        assert len(Y) == 8  # 2^3, default 2 levels per continuous

    def test_two_level_via_explicit_levels(self):
        """Users can get a 2-level factorial by specifying levels."""
        factors = [Factor("A", levels=[-1, 1]), Factor("B", levels=[-1, 1])]
        Y = full_factorial(factors)
        assert len(Y) == 4  # 2^2

    def test_continuous_custom_levels(self):
        factors = [Factor("A", levels=[-1, 0, 1]), Factor("B", levels=[-1, 0, 1])]
        Y = full_factorial(factors)
        assert len(Y) == 9  # 3^2

    def test_continuous_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5)]
        Y = full_factorial(factors)
        assert len(Y) == 4  # 2^2, default 2 levels (min, max)
        assert set(Y["A"].unique()) == {10.0, 20.0}
        assert set(Y["B"].unique()) == {0.0, 5.0}

    def test_categorical(self):
        factors = [
            Factor("A", type="categorical", levels=["L1", "L2", "L3"]),
            Factor("B", type="categorical", levels=["X", "Y"]),
        ]
        Y = full_factorial(factors)
        assert len(Y) == 6  # 3 * 2
        assert set(Y["A"].unique()) == {"L1", "L2", "L3"}
        assert set(Y["B"].unique()) == {"X", "Y"}

    def test_mixed_factors(self):
        factors = [
            Factor("A"),
            Factor("B", type="categorical", levels=["X", "Y"]),
        ]
        Y = full_factorial(factors)
        assert len(Y) == 4  # 2 * 2

    def test_center_points(self):
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors, n_center=3)
        assert len(Y) == 7  # 4 factorial + 3 center
        centers = Y.iloc[4:]
        np.testing.assert_allclose(centers["A"].values, 0.0)
        np.testing.assert_allclose(centers["B"].values, 0.0)

    def test_center_points_repeated_per_categorical_level(self):
        """n_center center runs are added for every categorical level."""
        factors = [Factor("A"), Factor("B"), Factor("C", type="categorical", levels=["X", "Y", "Z"])]
        Y = full_factorial(factors, n_center=2)
        assert len(Y) == 12 + 6  # 2 * 2 * 3 factorial + 2 center runs at each of 3 levels
        centers = Y.iloc[12:]
        np.testing.assert_allclose(centers[["A", "B"]].values, 0.0)
        assert centers["C"].value_counts().to_dict() == {"X": 2, "Y": 2, "Z": 2}

    def test_center_points_two_categoricals(self):
        factors = [
            Factor("A"),
            Factor("B", type="categorical", levels=["X", "Y"]),
            Factor("C", type="categorical", levels=["P", "Q", "R"]),
        ]
        Y = full_factorial(factors, n_center=1)
        assert len(Y) == 12 + 6  # one center run per (B, C) combination

    def test_center_points_midpoint_of_explicit_levels(self):
        """The center lies in the middle of the factor's levels, not of [min, max]."""
        factors = [Factor("A", min=0, max=10, levels=[2, 4]), Factor("B")]
        Y = full_factorial(factors, n_center=1)
        assert Y["A"].iloc[-1] == pytest.approx(3.0)

    def test_center_points_all_categorical_raises(self):
        factors = [Factor("A", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="at least one continuous factor"):
            full_factorial(factors, n_center=1)

    def test_n_levels_three(self):
        """n_levels=3 reproduces a 3^k design with levels (min, mid, max)."""
        factors = [Factor("A", min=10, max=20), Factor("B"), Factor("C")]
        Y = full_factorial(factors, n_levels=3)
        assert len(Y) == 27  # 3^3
        assert set(Y["A"].unique()) == {10.0, 15.0, 20.0}
        assert set(Y["B"].unique()) == {-1.0, 0.0, 1.0}

    def test_explicit_levels_override_n_levels(self):
        factors = [Factor("A", min=1, max=5, levels=[1, 3, 5]), Factor("B")]
        Y = full_factorial(factors, n_levels=2)
        assert len(Y) == 6  # 3 explicit * 2 default
        assert set(Y["A"].unique()) == {1.0, 3.0, 5.0}
        assert set(Y["B"].unique()) == {-1.0, 1.0}

    @pytest.mark.parametrize("n_levels, expected", [(2, 2 * 3 * 2), (3, 3 * 3 * 2), (4, 4 * 3 * 2)])
    def test_mixed_default_explicit_categorical(self, n_levels, expected):
        """Only continuous factors without explicit levels follow n_levels."""
        factors = [
            Factor("A"),
            Factor("B", levels=[-1, 0, 1]),
            Factor("C", type="categorical", levels=["X", "Y"]),
        ]
        Y = full_factorial(factors, n_levels=n_levels)
        assert len(Y) == expected
        assert Y["A"].nunique() == n_levels
        assert Y["B"].nunique() == 3
        assert Y["C"].nunique() == 2

    def test_n_levels_below_two_raises(self):
        with pytest.raises(AssertionError, match="n_levels must be at least 2"):
            full_factorial([Factor("A")], n_levels=1)

    def test_empty_factors_raises(self):
        with pytest.raises(AssertionError, match="At least one factor"):
            full_factorial([])


class TestRandomize:
    def test_same_rows_different_order(self):
        set_seed(42)
        Y = full_factorial([Factor(n) for n in "ABCD"])
        Yr = randomize(Y)
        assert not Yr.index.equals(Y.index)
        # The index keeps the standard order, so sorting restores the original design.
        pd.testing.assert_frame_equal(Yr.sort_index(), Y)

    def test_reproducible_with_set_seed(self):
        Y = full_factorial([Factor(n) for n in "ABCD"])
        set_seed(42)
        Y1 = randomize(Y)
        set_seed(42)
        Y2 = randomize(Y)
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_does_not_modify_input(self):
        Y = full_factorial([Factor(n) for n in "ABC"])
        Y_copy = Y.copy()
        randomize(Y)
        pd.testing.assert_frame_equal(Y, Y_copy)
