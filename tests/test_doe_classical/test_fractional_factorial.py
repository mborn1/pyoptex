from itertools import combinations, product

import numpy as np
import pandas as pd
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

    def test_center_points_default_levels(self):
        factors = [Factor(n) for n in "ABC"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]}, n_center=3)
        assert len(Y) == 7  # 2^(3-1) + 3
        np.testing.assert_allclose(Y.iloc[4:].values, 0.0)

    def test_center_points_n_levels_three(self):
        factors = [Factor("A"), Factor("B")]
        Y = fractional_factorial(factors, generators={}, n_levels=3, n_center=2)
        assert len(Y) == 11  # 3^2 + 2

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

    def test_default_continuous_two_level(self):
        """Default continuous factors have 2 levels, giving a 2^(3-1)."""
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 4  # 2^(3-1)
        np.testing.assert_allclose(Y["C"].values, Y["A"].values * Y["B"].values)

    def test_n_levels_three(self):
        """n_levels=3 gives a 3^(3-1) for default continuous factors."""
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]}, n_levels=3)
        assert len(Y) == 9  # 3^(3-1)
        assert set(Y["C"].unique()) == {-1.0, 0.0, 1.0}

    def test_non_prime_levels_raise(self):
        """Mod-L generators are only valid for prime L; 4 levels is rejected."""
        factors = [Factor(n, levels=[-3, -1, 1, 3]) for n in "ABC"]
        with pytest.raises(AssertionError, match="prime number of levels"):
            fractional_factorial(factors, generators={"C": ["A", "B"]})

    def test_non_prime_base_without_generator_allowed(self):
        """A 4-level factor that is not generated is just part of the full factorial base."""
        factors = [Factor("A", levels=[-3, -1, 1, 3]), Factor("B"), Factor("C")]
        Y = fractional_factorial(factors, generators={"C": ["B"]})
        assert len(Y) == 8  # 4 * 2

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

    def test_l9_orthogonal_array(self):
        """3^(4-2) with C = A+B, D = A+2B is the L9: every pair of columns is a full 3x3 grid."""
        factors = [Factor(n) for n in "ABCD"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"], "D": {"A": 1, "B": 2}}, n_levels=3)
        assert len(Y) == 9
        for a, b in combinations("ABCD", 2):
            pairs = set(zip(Y[a], Y[b], strict=True))
            assert len(pairs) == 9, f"Columns {a} and {b} do not form a full 3x3 grid"

    def test_coefficients_three_level(self):
        """D = A + 2B (mod 3) in level-index space."""
        factors = [Factor(n, levels=[-1, 0, 1]) for n in "ABD"]
        Y = fractional_factorial(factors, generators={"D": {"A": 1, "B": 2}})
        idx = {n: (Y[n].values + 1).astype(int) for n in "ABD"}
        np.testing.assert_array_equal(idx["D"], (idx["A"] + 2 * idx["B"]) % 3)

    def test_negative_coefficient_three_level_is_mod_l(self):
        """For L > 2, -1 is equivalent to L - 1: A - B == A + 2B (mod 3)."""
        factors = [Factor(n) for n in "ABD"]
        Y1 = fractional_factorial(factors, generators={"D": ["A", "-B"]}, n_levels=3)
        Y2 = fractional_factorial(factors, generators={"D": {"A": 1, "B": 2}}, n_levels=3)
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_list_is_shorthand_for_unit_coefficients(self):
        factors = [Factor(n) for n in "ABCD"]
        Y1 = fractional_factorial(factors, generators={"D": ["A", "B", "C"]})
        Y2 = fractional_factorial(factors, generators={"D": {"A": 1, "B": 1, "C": 1}})
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_fold_over(self):
        """D = -ABC gives exactly the 8 runs of the 2^4 that D = ABC leaves out."""
        factors = [Factor(n) for n in "ABCD"]
        Y1 = fractional_factorial(factors, generators={"D": ["A", "B", "C"]})
        Y2 = fractional_factorial(factors, generators={"D": ["-A", "B", "C"]})
        np.testing.assert_allclose(Y2["D"].values, -Y2["A"].values * Y2["B"].values * Y2["C"].values)
        runs1 = set(map(tuple, Y1.values))
        runs2 = set(map(tuple, Y2.values))
        assert len(runs1) == len(runs2) == 8
        assert runs1.isdisjoint(runs2)
        assert runs1 | runs2 == set(product([-1.0, 1.0], repeat=4))

    def test_fold_over_dict_form(self):
        factors = [Factor(n) for n in "ABCD"]
        Y1 = fractional_factorial(factors, generators={"D": ["-A", "B", "C"]})
        Y2 = fractional_factorial(factors, generators={"D": {"A": 1, "B": -1, "C": 1}})
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_sign_does_not_avoid_duplicate_check(self):
        """D = ABC and E = -ABC are perfectly aliased."""
        factors = [Factor(n) for n in "ABCDE"]
        with pytest.raises(AssertionError, match="perfectly aliased duplicate"):
            fractional_factorial(factors, generators={"D": ["A", "B", "C"], "E": ["-A", "B", "C"]})

    def test_proportional_coefficients_are_duplicates(self):
        """For 3 levels, 2A + 2B (mod 3) relabels the levels of A + B."""
        factors = [Factor(n) for n in "ABCD"]
        with pytest.raises(AssertionError, match="perfectly aliased duplicate"):
            fractional_factorial(factors, generators={"C": ["A", "B"], "D": {"A": 2, "B": 2}}, n_levels=3)

    def test_two_level_coefficient_must_be_unit(self):
        factors = [Factor(n) for n in "ABC"]
        with pytest.raises(AssertionError, match="must be 1 or -1"):
            fractional_factorial(factors, generators={"C": {"A": 1, "B": 2}})

    def test_coefficient_multiple_of_l_raises(self):
        factors = [Factor(n) for n in "ABC"]
        with pytest.raises(AssertionError, match="multiple of 3"):
            fractional_factorial(factors, generators={"C": {"A": 1, "B": 3}}, n_levels=3)

    def test_non_integer_coefficient_raises(self):
        factors = [Factor(n) for n in "ABC"]
        with pytest.raises(AssertionError, match="must be an integer"):
            fractional_factorial(factors, generators={"C": {"A": 1, "B": 1.5}}, n_levels=3)

    def test_center_points_with_categorical_raise(self):
        factors = [Factor("A"), Factor("B"), Factor("C", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="Center points are not defined for categorical"):
            fractional_factorial(factors, generators={"C": ["A", "B"]}, n_center=1)

    def test_categorical_without_center_points_allowed(self):
        factors = [Factor("A"), Factor("B"), Factor("C", type="categorical", levels=["X", "Y"])]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        assert len(Y) == 4

    def test_confounding_structure_three_level(self):
        """In a 3^(3-1) design, C = (A+B) mod 3 in level-index space."""
        factors = [Factor(n, levels=[-1, 0, 1]) for n in "ABC"]
        Y = fractional_factorial(factors, generators={"C": ["A", "B"]})
        idx_a = ((Y["A"].values + 1) / 1).astype(int)  # map -1,0,1 -> 0,1,2
        idx_b = ((Y["B"].values + 1) / 1).astype(int)
        idx_c = ((Y["C"].values + 1) / 1).astype(int)
        np.testing.assert_array_equal(idx_c, (idx_a + idx_b) % 3)
