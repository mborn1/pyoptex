from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, plackett_burman
from pyoptex.doe.classical.plackett_burman import _SUPPORTED_SIZES, _pb_matrix


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

    def test_n_runs_more_runs_than_factors(self):
        """6 factors in the 12-run design: the first 6 columns of PB12, orthogonal."""
        Y = plackett_burman([Factor(f"X{i}") for i in range(6)], n_runs=12)
        assert Y.shape == (12, 6)
        X = Y.to_numpy(dtype=np.float64)
        np.testing.assert_allclose(X.T @ X, 12 * np.eye(6), atol=1e-10)
        np.testing.assert_array_equal(X, _pb_matrix(12)[:, :6])

    @pytest.mark.parametrize("n_runs", [20, 24])
    def test_n_runs_larger_nonregular_sizes(self, n_runs):
        Y = plackett_burman([Factor(f"X{i}") for i in range(7)], n_runs=n_runs)
        assert Y.shape == (n_runs, 7)
        X = Y.to_numpy(dtype=np.float64)
        np.testing.assert_allclose(X.T @ X, n_runs * np.eye(7), atol=1e-10)

    def test_n_runs_equal_to_default(self):
        factors = [Factor(f"X{i}") for i in range(6)]
        pd.testing.assert_frame_equal(plackett_burman(factors, n_runs=8), plackett_burman(factors))

    def test_n_runs_with_center_points(self):
        Y = plackett_burman([Factor(f"X{i}") for i in range(6)], n_runs=12, n_center=2)
        assert len(Y) == 12 + 2

    @pytest.mark.parametrize("n_runs", [6, 30])
    def test_invalid_n_runs_raises(self, n_runs):
        """Fewer runs than k + 1 (6 for 6 factors) or an unsupported size (30)."""
        with pytest.raises(ValueError, match="supported sizes"):
            plackett_burman([Factor(f"X{i}") for i in range(6)], n_runs=n_runs)

    def test_larger_power_of_two_raises(self):
        """The first 6 columns of the 16-run Sylvester design are the 8-run design replicated twice."""
        with pytest.raises(ValueError, match=r"replicated 2 times.*fractional_factorial\(factors, min_aberration_gen"):
            plackett_burman([Factor(f"X{i}") for i in range(6)], n_runs=16)
        X16 = _pb_matrix(16)[:, :6]
        np.testing.assert_array_equal(X16[:8], X16[8:])

    def test_larger_power_of_two_without_replication_raises(self):
        """10 factors in 16 runs is not replicated, but the minimum aberration design is better."""
        with pytest.raises(ValueError, match=r"regular resolution III fraction.*min_aberration_generators"):
            plackett_burman([Factor(f"X{i}") for i in range(10)], n_runs=16)

    def test_larger_power_of_two_equal_to_full_factorial_raises(self):
        """4 factors in 16 runs is the full 2^4 factorial."""
        with pytest.raises(
            ValueError, match=r"full 16-run factorial of the 4 factors instead: full_factorial\(factors\)"
        ):
            plackett_burman([Factor(f"X{i}") for i in range(4)], n_runs=16)

    def test_larger_power_of_two_beyond_full_factorial_raises(self):
        """3 factors in 16 runs: more runs than the full 2^3 factorial."""
        with pytest.raises(ValueError, match=r"full_factorial\(factors\) for all 8 runs"):
            plackett_burman([Factor(f"X{i}") for i in range(3)], n_runs=16)

    @pytest.mark.parametrize("N, k", [(40, 8), (40, 15), (40, 19), (56, 10), (56, 27)])
    def test_doubled_sizes_are_fold_overs(self, N, k):
        """40 and 56 runs with k <= N/2 - 1: the fold-over of PB20 / PB28, not the smaller design run twice."""
        X = plackett_burman([Factor(f"X{i}") for i in range(k)], n_runs=N).to_numpy(dtype=np.float64)
        assert len({tuple(row) for row in X}) == N
        np.testing.assert_allclose(X.T @ X, N * np.eye(k), atol=1e-10)
        for cols in combinations(range(k), 3):
            assert abs(np.prod(X[:, list(cols)], axis=1).sum()) < 1e-10
        np.testing.assert_array_equal(X[: N // 2], _pb_matrix(N // 2)[:, :k])  # first stage: the original design
        np.testing.assert_array_equal(X[N // 2 :], -X[: N // 2])  # second stage: its mirror image

    def test_no_design_is_two_identical_halves(self):
        """For every size up to 64 and k >= 3, the design is orthogonal and not the same design run twice.
        (With 1 or 2 factors any balanced design has the same rows as its mirror image.)"""
        for N in [n for n in _SUPPORTED_SIZES if n <= 64]:
            for k in range(3, N):
                try:
                    Y = plackett_burman([Factor(f"X{i}") for i in range(k)], n_runs=N)
                except ValueError:
                    continue
                X = Y.to_numpy(dtype=np.float64)
                np.testing.assert_allclose(X.T @ X, N * np.eye(k), atol=1e-10, err_msg=f"N={N}, k={k}")
                halves = [Counter(map(tuple, X[: N // 2])), Counter(map(tuple, X[N // 2 :]))]
                assert halves[0] != halves[1], f"N={N}, k={k} is the same design run twice"

    def test_64_factors_jumps_to_128(self):
        """Sizes 68 to 124 are not implemented, so 64 factors land on PB128."""
        factors = [Factor(f"X{i}") for i in range(64)]
        assert len(plackett_burman(factors)) == 128
