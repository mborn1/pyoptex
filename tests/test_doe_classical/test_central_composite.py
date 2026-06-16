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
