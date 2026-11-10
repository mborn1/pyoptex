import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, supersaturated


def _s_values(Y):
    """Off-diagonal inner products s_ij of the +/-1 coded columns."""
    X = Y.to_numpy(dtype=np.float64)
    S = X.T @ X
    return S[~np.eye(len(S), dtype=bool)]


class TestSupersaturated:
    def test_lin_12_run(self):
        """From PB12: 6 runs for 10 factors with E(s^2) = 4 (Lin, 1993)."""
        Y = supersaturated([Factor(f"X{i}") for i in range(10)])
        assert isinstance(Y, pd.DataFrame)
        assert Y.shape == (6, 10)
        np.testing.assert_allclose(np.mean(_s_values(Y) ** 2), 4.0)

    @pytest.mark.parametrize("N", [12, 20, 24, 28, 36, 44, 48])
    def test_saturated_sizes(self, N):
        """Shape (N/2, N-2), balanced columns, no identical columns, and E(s^2) at the Nguyen (1996) lower bound."""
        m = N - 2
        Y = supersaturated([Factor(f"X{i}") for i in range(m)])
        assert Y.shape == (N // 2, m)
        np.testing.assert_array_equal(Y.sum(axis=0).values, 0.0)

        s = _s_values(Y)
        n = N // 2
        assert np.max(np.abs(s)) < n
        bound = n**2 * (m - n + 1) / ((m - 1) * (n - 1))
        np.testing.assert_allclose(np.mean(s**2), bound, rtol=1e-10)

    def test_fewer_factors_than_saturated(self):
        """15 factors do not fit in 6 runs, so the 10-run design (from PB20) is used."""
        Y = supersaturated([Factor(f"X{i}") for i in range(15)])
        assert Y.shape == (10, 15)

    def test_not_supersaturated_raises(self):
        with pytest.raises(AssertionError, match="would not be supersaturated"):
            supersaturated([Factor(f"X{i}") for i in range(5)])

    def test_too_many_factors_raises(self):
        with pytest.raises(ValueError, match="Too many factors"):
            supersaturated([Factor(f"X{i}") for i in range(47)])

    def test_three_level_raises(self):
        factors = [Factor(f"X{i}") for i in range(9)] + [Factor("Z", levels=[-1, 0, 1])]
        with pytest.raises(AssertionError, match="exactly 2 levels"):
            supersaturated(factors)

    def test_levels_mapped(self):
        factors = [Factor(f"X{i}", min=0, max=10) for i in range(9)]
        factors.append(Factor("Cat", type="categorical", levels=["Lo", "Hi"]))
        Y = supersaturated(factors)
        assert set(Y["X0"].unique()) == {0.0, 10.0}
        assert set(Y["Cat"].unique()) == {"Lo", "Hi"}
