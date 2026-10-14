import numpy as np
import pytest

from pyoptex.doe.classical import Factor, central_composite


class TestCentralComposite:
    def test_face_centered(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="face-centered", n_center=1)
        # 2^2 + 2*2 + 1 = 4 + 4 + 1 = 9
        assert len(Y) == 9
        assert Y["A"].min() >= -1.0
        assert Y["A"].max() <= 1.0

    def test_rotatable(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="rotatable", n_center=1)
        assert len(Y) == 9
        alpha = 4**0.25  # (2^2)^(1/4)
        np.testing.assert_allclose(Y["A"].max(), alpha, rtol=1e-10)

    def test_inscribed(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="inscribed", n_center=1)
        assert len(Y) == 9
        np.testing.assert_allclose(Y["A"].max(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(Y["A"].min(), -1.0, rtol=1e-10)

    def test_custom_alpha(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha=1.5, n_center=2)
        assert len(Y) == 10
        np.testing.assert_allclose(Y["A"].max(), 1.5, rtol=1e-10)

    def test_three_factors(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = central_composite(factors, alpha="face-centered", n_center=2)
        # 2^3 + 2*3 + 2 = 8 + 6 + 2 = 16
        assert len(Y) == 16

    def test_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5)]
        Y = central_composite(factors, alpha="face-centered", n_center=1)
        assert Y["A"].min() >= 10.0
        assert Y["A"].max() <= 20.0
        assert Y["B"].min() >= 0.0
        assert Y["B"].max() <= 5.0

    @pytest.mark.parametrize("k, expected_center", [(2, 5), (3, 6), (4, 7), (5, 10)])
    def test_default_center_points_full_cube(self, k, expected_center):
        """Default center points follow the Box & Hunter (1957) uniform-precision table."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = central_composite(factors)
        assert len(Y) == 2**k + 2 * k + expected_center

    @pytest.mark.parametrize("k, expected_center", [(5, 6), (6, 9), (7, 14)])
    def test_default_center_points_half_cube(self, k, expected_center):
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = central_composite(factors, cube="half")
        assert len(Y) == 2 ** (k - 1) + 2 * k + expected_center

    def test_explicit_center_points_override_default(self):
        Y = central_composite([Factor("A"), Factor("B")], n_center=2)
        assert len(Y) == 4 + 4 + 2

    def test_no_tabulated_default_raises(self):
        factors = [Factor(f"X{i}") for i in range(8)]
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            central_composite(factors)
        assert len(central_composite(factors, n_center=4)) == 256 + 16 + 4

    def test_half_cube_five_factors(self):
        """k = 5 with E = ABCD: 16 cube + 10 axial + center runs, rotatable alpha = 16^(1/4) = 2."""
        factors = [Factor(n) for n in "ABCDE"]
        Y = central_composite(factors, cube="half", n_center=3)
        assert len(Y) == 16 + 10 + 3
        np.testing.assert_allclose(Y["A"].max(), 2.0, rtol=1e-10)
        cube = Y.iloc[:16]
        np.testing.assert_allclose(cube["E"].values, cube[["A", "B", "C", "D"]].prod(axis=1).values)

    def test_half_cube_too_few_factors_raises(self):
        factors = [Factor(n) for n in "ABCD"]
        with pytest.raises(AssertionError, match="at least 5 factors"):
            central_composite(factors, cube="half")

    def test_custom_generators(self):
        factors = [Factor(n) for n in "ABCDE"]
        Y1 = central_composite(factors, cube={"E": ["A", "B", "C", "D"]})
        Y2 = central_composite(factors, cube="half")
        np.testing.assert_allclose(Y1.values, Y2.values)

    def test_custom_generators_without_table_entry_require_n_center(self):
        """A 2^(6-2) cube has no tabulated center points."""
        factors = [Factor(f"X{i}") for i in range(6)]
        generators = {"X4": ["X0", "X1", "X2"], "X5": ["X1", "X2", "X3"]}
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            central_composite(factors, cube=generators)
        Y = central_composite(factors, cube=generators, n_center=4)
        assert len(Y) == 16 + 12 + 4
        np.testing.assert_allclose(Y["X0"].max(), 2.0, rtol=1e-10)

    def test_unknown_cube_raises(self):
        with pytest.raises(ValueError, match="Unknown cube type"):
            central_composite([Factor("A"), Factor("B")], cube="quarter")

    def test_categorical_raises(self):
        factors = [Factor("A"), Factor("B", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="must be continuous"):
            central_composite(factors)

    def test_one_factor_raises(self):
        factors = [Factor("A")]
        with pytest.raises(AssertionError, match="at least 2 factors"):
            central_composite(factors)

    def test_nonpositive_alpha_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(AssertionError, match="alpha must be strictly positive"):
            central_composite(factors, alpha=0.0)

    def test_unknown_alpha_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(ValueError, match="Unknown alpha type"):
            central_composite(factors, alpha="bogus")
