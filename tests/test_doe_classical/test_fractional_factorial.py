import numpy as np
import pytest

from pyoptex.doe.classical import Factor, fractional_factorial


class TestFractionalFactorial:
    """Tests for the generalized fractional factorial generator."""

    def test_half_fraction_two_level(self):
        factors = [
            Factor("A", levels=[-1, 1]),
            Factor("B", levels=[-1, 1]),
            Factor("C", levels=[-1, 1]),
        ]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 4  # 2^(3-1)
        assert set(Y["C"].unique()) == {-1.0, 1.0}

    def test_quarter_fraction_two_level(self):
        # Classical resolution-III 2^(5-2) design: base A, B, C with
        # generators D = AB and E = AC.
        factors = [Factor(n, levels=[-1, 1]) for n in "ABCDE"]
        Y = fractional_factorial(factors, generators={"D": ["A", "B"], "E": ["A", "C"]})
        assert len(Y) == 8  # 2^(5-2)
        np.testing.assert_allclose(Y["D"].values, Y["A"].values * Y["B"].values)
        np.testing.assert_allclose(Y["E"].values, Y["A"].values * Y["C"].values)

    def test_duplicate_generators_raise(self):
        factors = [Factor(n, levels=[-1, 1]) for n in "ABCD"]
        with pytest.raises(AssertionError, match="perfectly aliased duplicate"):
            fractional_factorial(factors, generators={"C": ["A", "B"], "D": ["B", "A"]})

    def test_center_points(self):
        factors = [Factor("A", levels=[-1, 1]), Factor("B", levels=[-1, 1])]
        Y = fractional_factorial(factors, generators={}, n_center=2)
        assert len(Y) == 6  # 4 + 2

    def test_categorical_two_level(self):
        factors = [
            Factor("A", type="categorical", levels=["Lo", "Hi"]),
            Factor("B", levels=[-1, 1]),
        ]
        Y = fractional_factorial(factors, generators={})
        assert len(Y) == 4
        assert set(Y["A"].unique()) == {"Lo", "Hi"}

    def test_custom_range(self):
        factors = [
            Factor("A", min=100, max=200, levels=[100, 200]),
            Factor("B", min=0, max=10, levels=[0, 10]),
        ]
        Y = fractional_factorial(factors, generators={})
        assert set(Y["A"].unique()) == {100.0, 200.0}
        assert set(Y["B"].unique()) == {0.0, 10.0}

    def test_half_fraction_three_generators(self):
        """2^(4-1) with D = ABC: D equals the coded product A*B*C exactly."""
        factors = [Factor(n, levels=[-1, 1]) for n in "ABCD"]
        Y = fractional_factorial(factors, generators={"D": ["A", "B", "C"]})
        assert len(Y) == 8  # 2^(4-1)
        np.testing.assert_allclose(Y["D"].values, Y["A"].values * Y["B"].values * Y["C"].values)

    def test_three_level_fraction(self):
        """3^(3-1) fractional factorial."""
        factors = [
            Factor("A", levels=[-1, 0, 1]),
            Factor("B", levels=[-1, 0, 1]),
            Factor("C", levels=[-1, 0, 1]),
        ]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 9  # 3^(3-1) = 9
        assert set(Y["C"].unique()) == {-1.0, 0.0, 1.0}

    def test_default_continuous_three_level(self):
        """Default continuous factors have 3 levels from coords_."""
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 9  # 3^(3-1)

    def test_three_level_categorical(self):
        """3-level categorical fractional factorial."""
        factors = [
            Factor("A", type="categorical", levels=["X", "Y", "Z"]),
            Factor("B", type="categorical", levels=["X", "Y", "Z"]),
            Factor("C", type="categorical", levels=["X", "Y", "Z"]),
        ]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 9
        assert set(Y["C"].unique()) == {"X", "Y", "Z"}

    def test_level_mismatch_raises(self):
        factors = [
            Factor("A", levels=[-1, 1]),
            Factor("B", levels=[-1, 0, 1]),
        ]
        with pytest.raises(AssertionError, match=r"levels.*must match"):
            fractional_factorial(factors, generators={"B": ["A"]})

    def test_generator_not_in_factors_raises(self):
        factors = [Factor("A", levels=[-1, 1])]
        with pytest.raises(AssertionError, match="not in the factor list"):
            fractional_factorial(factors, generators={"Z": ["A"]})

    def test_empty_factors_raises(self):
        with pytest.raises(AssertionError, match="At least one factor"):
            fractional_factorial([], generators={})

    def test_confounding_structure_two_level(self):
        """In a 2^(3-1) design with C = AB, C equals the coded product A*B exactly."""
        factors = [Factor(n, levels=[-1, 1]) for n in "ABC"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        ab = Y["A"].values * Y["B"].values
        # Exact equality (correct sign), not merely |correlation| == 1.
        np.testing.assert_allclose(Y["C"].values, ab)

    def test_confounding_structure_three_level(self):
        """In a 3^(3-1) design, C = (A+B) mod 3 in level-index space."""
        factors = [Factor(n, levels=[-1, 0, 1]) for n in "ABC"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        idx_a = ((Y["A"].values + 1) / 1).astype(int)  # map -1,0,1 -> 0,1,2
        idx_b = ((Y["B"].values + 1) / 1).astype(int)
        idx_c = ((Y["C"].values + 1) / 1).astype(int)
        np.testing.assert_array_equal(idx_c, (idx_a + idx_b) % 3)
