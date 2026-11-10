from itertools import combinations
from math import comb

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


# (A3, A4, A5) as printed in Mee (2009), Appendix G, Tables G.1-G.3, G.4a and G.5a,
# keyed by (number of factors, number of runs). Design 22-17.1 (22, 32) is the one whose
# generator text in the book is corrected; its printed word length pattern is used as is.
# fmt: off
_MEE_PRINTED_WLP = {
    (4, 8): (0, 1, 0), (5, 8): (2, 1, 0), (6, 8): (4, 3, 0), (7, 8): (7, 7, 0),
    (5, 16): (0, 0, 1), (6, 16): (0, 3, 0), (7, 16): (0, 7, 0), (8, 16): (0, 14, 0), (9, 16): (4, 14, 8),
    (10, 16): (8, 18, 16), (11, 16): (12, 26, 28), (12, 16): (16, 39, 48), (13, 16): (22, 55, 72), (14, 16): (28, 77, 112),
    (15, 16): (35, 105, 168),
    (6, 32): (0, 0, 0), (7, 32): (0, 1, 2), (8, 32): (0, 3, 4), (9, 32): (0, 6, 8), (10, 32): (0, 10, 16),
    (11, 32): (0, 25, 0), (12, 32): (0, 38, 0), (13, 32): (0, 55, 0), (14, 32): (0, 77, 0), (15, 32): (0, 105, 0),
    (16, 32): (0, 140, 0), (17, 32): (8, 140, 112), (18, 32): (16, 148, 224), (19, 32): (24, 164, 344), (20, 32): (32, 188, 480),
    (21, 32): (40, 220, 641), (22, 32): (48, 263, 832), (23, 32): (56, 315, 1064), (24, 32): (64, 378, 1344), (25, 32): (76, 442, 1656),
    (26, 32): (88, 518, 2032), (27, 32): (100, 606, 2484), (28, 32): (112, 707, 3024), (29, 32): (126, 819, 3640), (30, 32): (140, 945, 4368),
    (31, 32): (155, 1085, 5208),
    (7, 64): (0, 0, 0), (8, 64): (0, 0, 2), (9, 64): (0, 1, 4), (10, 64): (0, 2, 8), (11, 64): (0, 4, 14),
    (12, 64): (0, 6, 24), (13, 64): (0, 14, 28), (14, 64): (0, 22, 40), (15, 64): (0, 30, 60), (16, 64): (0, 43, 81),
    (17, 64): (0, 59, 108), (18, 64): (0, 78, 144), (19, 64): (0, 100, 192), (20, 64): (0, 125, 256), (21, 64): (0, 204, 0),
    (22, 64): (0, 250, 0), (23, 64): (0, 304, 0), (24, 64): (0, 365, 0), (25, 64): (0, 435, 0), (26, 64): (0, 515, 0),
    (27, 64): (0, 605, 0), (28, 64): (0, 706, 0), (29, 64): (0, 819, 0), (30, 64): (0, 945, 0), (31, 64): (0, 1085, 0),
    (8, 128): (0, 0, 0), (9, 128): (0, 0, 0), (10, 128): (0, 0, 3), (11, 128): (0, 0, 6), (12, 128): (0, 1, 8),
    (13, 128): (0, 2, 16), (14, 128): (0, 3, 24), (15, 128): (0, 7, 32), (16, 128): (0, 10, 48), (17, 128): (0, 15, 60),
    (18, 128): (0, 20, 80), (19, 128): (0, 27, 120), (20, 128): (0, 36, 152), (21, 128): (0, 51, 200), (22, 128): (0, 65, 248),
    (23, 128): (0, 83, 316), (24, 128): (0, 102, 384), (25, 128): (0, 124, 482), (26, 128): (0, 152, 568), (27, 128): (0, 180, 690),
    (28, 128): (0, 210, 840), (29, 128): (0, 266, 945), (30, 128): (0, 335, 972), (31, 128): (0, 391, 1134),
}
# fmt: on


def _word_length_pattern(X, lengths):
    """
    Number of defining words of each length, by the MacWilliams identity.

    The runs of a regular 2^(k-p) design, coded 0/1, form a binary linear code
    whose dual is the set of defining words, so the number of words of length j is
    ``A_j = (1/n) * sum over runs of K_j(w)``, with w the number of ones in the run
    and ``K_j(w) = sum_s (-1)^s C(w, s) C(k - w, j - s)`` the Krawtchouk polynomial.
    """
    n, k = X.shape
    weights = np.sum(X < 0, axis=1).tolist()
    pattern = []
    for j in lengths:
        total = sum(sum((-1) ** s * comb(w, s) * comb(k - w, j - s) for s in range(j + 1)) for w in weights)
        assert total % n == 0
        pattern.append(total // n)
    return tuple(pattern)


def test_word_length_pattern_as_printed_in_mee():
    """Every catalogued design has the word length pattern (A3, A4, A5) printed in Mee (2009), Appendix G."""
    assert set(_MEE_PRINTED_WLP) == set(_MA_GENERATORS) - {(3, 4)}
    for (k, n_runs), expected in _MEE_PRINTED_WLP.items():
        factors = _factors(k)
        Y = fractional_factorial(factors, min_aberration_generators(factors, n_runs=n_runs))
        pattern = _word_length_pattern(Y.to_numpy(dtype=np.int64), (3, 4, 5))
        assert pattern == expected, f"({k}, {n_runs}) has word length pattern {pattern}, Mee prints {expected}"


def test_word_length_pattern_matches_enumeration():
    """The MacWilliams computation agrees with enumerating all products of generators on small designs."""
    for (k, n_runs), (_, gens) in _MA_GENERATORS.items():
        if len(gens) > 10:
            continue
        p = int(np.log2(n_runs))
        words = [(1 << (p + i)) | sum(1 << j for j in gen) for i, gen in enumerate(gens)]
        lengths = []
        for mask in range(1, 2 ** len(words)):
            word = 0
            for i, w in enumerate(words):
                if mask >> i & 1:
                    word ^= w
            lengths.append(bin(word).count("1"))
        expected = tuple(lengths.count(j) for j in (3, 4, 5))
        factors = _factors(k)
        X = fractional_factorial(factors, min_aberration_generators(factors, n_runs=n_runs)).to_numpy(dtype=np.int64)
        assert _word_length_pattern(X, (3, 4, 5)) == expected, f"({k}, {n_runs})"


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
