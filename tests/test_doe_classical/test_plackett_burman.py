import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, plackett_burman


class TestPlackettBurman:
    def test_basic_3_factors(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = plackett_burman(factors)
        assert isinstance(Y, pd.DataFrame)
        assert len(Y.columns) == 3
        assert len(Y) == 4

    def test_7_factors(self):
        factors = [Factor(f"X{i}") for i in range(7)]
        Y = plackett_burman(factors)
        assert len(Y) == 8  # smallest power of 2 >= 8

    def test_11_factors(self):
        factors = [Factor(f"X{i}") for i in range(11)]
        Y = plackett_burman(factors)
        assert len(Y) == 12

    def test_values_are_two_level(self):
        factors = [Factor(f"X{i}") for i in range(5)]
        Y = plackett_burman(factors)
        for col in Y.columns:
            assert set(Y[col].unique()) == {-1.0, 1.0}

    def test_continuous_explicit_levels(self):
        """Explicit continuous levels are honored even with default min/max."""
        factors = [Factor(f"X{i}", levels=[10, 20]) for i in range(5)]
        Y = plackett_burman(factors)
        for col in Y.columns:
            assert set(Y[col].unique()) == {10.0, 20.0}

    def test_categorical_levels_mapped(self):
        """Categorical two-level factors map back to their level names."""
        factors = [
            Factor("A", type="categorical", levels=["Lo", "Hi"]),
            Factor("B", type="categorical", levels=["Off", "On"]),
            Factor("C"),
        ]
        Y = plackett_burman(factors)
        assert set(Y["A"].unique()) == {"Lo", "Hi"}
        assert set(Y["B"].unique()) == {"Off", "On"}

    def test_three_level_raises(self):
        factors = [Factor("A", levels=[-1, 0, 1])]
        with pytest.raises(AssertionError, match="exactly 2 levels"):
            plackett_burman(factors)

    @pytest.mark.parametrize(
        "k, expected_N",
        [
            (3, 4),
            (7, 8),
            (11, 12),
            (15, 16),
            (19, 20),
            (23, 24),
            (27, 28),
            (35, 36),
            (39, 40),
            (43, 44),
            (47, 48),
            (51, 52),
            (55, 56),
            (59, 60),
        ],
    )
    def test_saturated_size(self, k, expected_N):
        """Saturated designs (k = N-1) should have exactly N runs."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = plackett_burman(factors)
        assert len(Y) == expected_N

    @pytest.mark.parametrize(
        "k, expected_N",
        [
            (3, 4),
            (7, 8),
            (11, 12),
            (15, 16),
            (19, 20),
            (23, 24),
            (27, 28),
            (31, 32),
            (35, 36),
            (39, 40),
            (43, 44),
            (47, 48),
            (51, 52),
            (55, 56),
            (59, 60),
            (63, 64),
        ],
    )
    def test_orthogonality(self, k, expected_N):
        """X'X should equal N*I for a saturated PB design."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = plackett_burman(factors)
        X = Y.values
        XtX = X.T @ X
        np.testing.assert_allclose(XtX, expected_N * np.eye(k), atol=1e-10)

    def test_25_factors_uses_28(self):
        """25 factors should land on PB28 (next multiple of 4 >= 26)."""
        factors = [Factor(f"X{i}") for i in range(25)]
        Y = plackett_burman(factors)
        assert len(Y) == 28

    def test_30_factors_uses_32(self):
        """30 factors should land on PB32 (Hadamard)."""
        factors = [Factor(f"X{i}") for i in range(30)]
        Y = plackett_burman(factors)
        assert len(Y) == 32

    def test_33_factors_uses_36(self):
        """33 factors should land on PB36."""
        factors = [Factor(f"X{i}") for i in range(33)]
        Y = plackett_burman(factors)
        assert len(Y) == 36

    def test_no_center_points_by_default(self):
        Y = plackett_burman([Factor(f"X{i}") for i in range(7)])
        assert len(Y) == 8

    def test_center_points(self):
        factors = [Factor(f"X{i}", levels=[10, 20]) for i in range(5)] + [Factor("Y", min=0, max=4)]
        Y = plackett_burman(factors, n_center=3)
        assert len(Y) == 8 + 3
        centers = Y.iloc[8:]
        np.testing.assert_allclose(centers[[f"X{i}" for i in range(5)]].values, 15.0)
        np.testing.assert_allclose(centers["Y"].values, 2.0)

    def test_center_points_with_categorical_raise(self):
        factors = [Factor("A"), Factor("B", type="categorical", levels=["Lo", "Hi"])]
        with pytest.raises(AssertionError, match="Center points are not defined for categorical"):
            plackett_burman(factors, n_center=1)

    @pytest.mark.parametrize("k, expected_N", [(48, 52), (52, 56), (56, 60), (60, 64)])
    def test_every_multiple_of_four_up_to_64(self, k, expected_N):
        """Up to 63 factors N is the smallest multiple of 4 that is >= k + 1."""
        factors = [Factor(f"X{i}") for i in range(k)]
        assert len(plackett_burman(factors)) == expected_N

    def test_64_factors_jumps_to_128(self):
        """Sizes 68 to 124 are not implemented, so 64 factors land on PB128."""
        factors = [Factor(f"X{i}") for i in range(64)]
        assert len(plackett_burman(factors)) == 128
