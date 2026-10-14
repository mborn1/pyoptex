from itertools import combinations

import numpy as np
import pytest

from pyoptex.doe.classical import Factor, central_composite, fractional_factorial, min_aberration_generators
from pyoptex.doe.classical._min_aberration import _MA_GENERATORS


def _factors(k):
    return [Factor(f"X{i}") for i in range(k)]


def _resolution(X, max_resolution):
    """Smallest number of columns whose element-wise product is constant (the shortest word)."""
    n, k = X.shape
    for r in range(1, min(k, max_resolution) + 1):
        idx = np.array(list(combinations(range(k), r)))
        products = np.prod(X[:, idx], axis=2)
        if np.any(np.abs(products.sum(axis=0)) == n):
            return r
    return np.inf


def test_catalogue_resolution():
    """Every catalogued design has its stated number of runs and resolution."""
    for (k, n_runs), (resolution, _) in _MA_GENERATORS.items():
        factors = _factors(k)
        Y = fractional_factorial(factors, min_aberration_generators(factors, n_runs=n_runs))
        X = Y.to_numpy(dtype=np.int8)
        assert X.shape == (n_runs, k), f"({k}, {n_runs}) has shape {X.shape}"
        assert len({tuple(row) for row in X}) == n_runs, f"({k}, {n_runs}) has duplicate runs"
        assert _resolution(X, resolution) == resolution, f"({k}, {n_runs}) does not have resolution {resolution}"


def test_textbook_five_factors_sixteen_runs():
    assert min_aberration_generators(_factors(5), n_runs=16) == {"X4": ["X0", "X1", "X2", "X3"]}


def test_textbook_seven_factors_eight_runs():
    gens = min_aberration_generators(_factors(7), n_runs=8)
    assert gens == {"X3": ["X0", "X1"], "X4": ["X0", "X2"], "X5": ["X1", "X2"], "X6": ["X0", "X1", "X2"]}


def test_uses_factor_names():
    factors = [Factor(name) for name in ["Temp", "Time", "pH", "Speed", "Load"]]
    assert min_aberration_generators(factors, n_runs=16) == {"Load": ["Temp", "Time", "pH", "Speed"]}


@pytest.mark.parametrize(
    "k, resolution, expected_runs",
    [(5, 3, 8), (5, 4, 16), (5, 5, 16), (6, 4, 16), (6, 5, 32), (8, 5, 64), (11, 4, 32), (4, 5, 16), (3, 4, 8)],
)
def test_smallest_design_by_resolution(k, resolution, expected_runs):
    factors = _factors(k)
    Y = fractional_factorial(factors, min_aberration_generators(factors, resolution=resolution))
    assert len(Y) == expected_runs
    assert _resolution(Y.to_numpy(dtype=np.int8), k) >= resolution


def test_full_factorial_has_no_generators():
    assert min_aberration_generators(_factors(4), n_runs=16) == {}


def test_resolution_out_of_range_raises():
    with pytest.raises(AssertionError, match="at most 128 runs"):
        min_aberration_generators(_factors(20), resolution=5)


def test_unavailable_run_size_raises():
    with pytest.raises(AssertionError, match="No catalogued fractional factorial for 5 factors in 4 runs"):
        min_aberration_generators(_factors(5), n_runs=4)


def test_exactly_one_criterion():
    with pytest.raises(AssertionError, match="exactly one"):
        min_aberration_generators(_factors(5))
    with pytest.raises(AssertionError, match="exactly one"):
        min_aberration_generators(_factors(5), n_runs=16, resolution=5)


def test_three_level_factor_raises():
    factors = [*_factors(4), Factor("Z", levels=[-1, 0, 1])]
    with pytest.raises(AssertionError, match="exactly 2 levels"):
        min_aberration_generators(factors, n_runs=16)


def test_as_central_composite_cube():
    """A resolution V cube for a CCD: for 5 factors this is the half fraction E = ABCD."""
    factors = _factors(5)
    Y1 = central_composite(factors, cube=min_aberration_generators(factors, resolution=5))
    Y2 = central_composite(factors, cube="half")
    np.testing.assert_allclose(Y1.values, Y2.values)
