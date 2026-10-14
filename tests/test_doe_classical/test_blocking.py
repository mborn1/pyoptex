from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex._seed import set_seed
from pyoptex.doe.classical import (
    Factor,
    box_behnken,
    central_composite,
    fractional_factorial,
    full_factorial,
    min_aberration_generators,
    randomize,
)


def _factors(k):
    return [Factor(f"X{i}") for i in range(k)]


def _names(factors):
    return [str(f.name) for f in factors]


def _model_columns(X, quadratic):
    """Main effects, two-factor interactions and optionally quadratic terms, with their factor indices."""
    k = X.shape[1]
    terms = [((i,), X[:, i]) for i in range(k)]
    terms += [((i, j), X[:, i] * X[:, j]) for i, j in combinations(range(k), 2)]
    if quadratic:
        terms += [((i, i), X[:, i] ** 2) for i in range(k)]
    return terms


def _non_orthogonal_terms(Y, factors, quadratic):
    """Model terms whose mean differs between blocks, i.e. that are not orthogonal to the blocks."""
    X = Y[_names(factors)].to_numpy(dtype=np.float64)
    blocks = Y["Block"].to_numpy()
    bad = []
    for term, col in _model_columns(X, quadratic):
        if any(abs(col[blocks == b].mean() - col.mean()) > 1e-10 for b in np.unique(blocks)):
            bad.append(term)
    return bad


class TestFactorialBlocking:
    def test_two_cubed_in_two_blocks(self):
        """The textbook scheme: ABC confounded with blocks."""
        factors = [Factor(n) for n in "ABC"]
        Y = full_factorial(factors, n_blocks=2)
        assert Y.attrs["confounded_with_blocks"] == [["A", "B", "C"]]
        assert Y["Block"].tolist() == [1] * 4 + [2] * 4
        abc = Y["A"] * Y["B"] * Y["C"]
        assert abc.groupby(Y["Block"]).nunique().tolist() == [1, 1]  # ABC is constant within each block
        assert abc.groupby(Y["Block"]).first().tolist() in ([-1.0, 1.0], [1.0, -1.0])
        assert _non_orthogonal_terms(Y, factors, quadratic=False) == []

    @pytest.mark.parametrize(
        "k, n_blocks, orders",
        [
            (3, 4, [2, 2, 2]),
            (4, 2, [4]),
            (4, 4, [2, 3, 3]),
            (5, 2, [5]),
            (5, 4, [3, 3, 4]),
            (6, 4, [4, 4, 4]),
            (6, 8, [3, 3, 3, 3, 4, 4, 4]),
            (7, 8, [4, 4, 4, 4, 4, 4, 4]),
        ],
    )
    def test_minimum_aberration_schemes(self, k, n_blocks, orders):
        """Orders of the confounded effects, as in the textbook blocking tables (e.g. Montgomery)."""
        factors = _factors(k)
        Y = full_factorial(factors, n_blocks=n_blocks)
        assert sorted(len(e) for e in Y.attrs["confounded_with_blocks"]) == orders
        assert np.all(np.bincount(Y["Block"])[1:] == 2**k // n_blocks)

    @pytest.mark.parametrize("k, n_blocks", [(3, 2), (4, 4), (5, 4), (5, 8), (6, 8)])
    def test_only_confounded_effects_differ_between_blocks(self, k, n_blocks):
        factors = _factors(k)
        Y = full_factorial(factors, n_blocks=n_blocks)
        name2idx = {n: i for i, n in enumerate(_names(factors))}
        confounded = {tuple(sorted(name2idx[n] for n in e)) for e in Y.attrs["confounded_with_blocks"]}
        assert set(_non_orthogonal_terms(Y, factors, quadratic=False)) == {t for t in confounded if len(t) <= 2}

    def test_center_points_per_block(self):
        Y = full_factorial(_factors(3), n_blocks=2, n_center=2)
        assert len(Y) == 8 + 2 * 2
        for _, block in Y.groupby("Block"):
            assert len(block) == 6
            np.testing.assert_array_equal(block.iloc[-2:][["X0", "X1", "X2"]].values, 0.0)

    def test_fractional_resolution_iv_in_two_blocks(self):
        """In the 2^(7-3) of resolution IV, the blocks can be defined without confounding any two-factor interaction."""
        factors = _factors(7)
        Y = fractional_factorial(factors, min_aberration_generators(factors, resolution=4), n_blocks=2)
        assert len(Y) == 16
        assert [len(e) for e in Y.attrs["confounded_with_blocks"]] == [3]
        assert _non_orthogonal_terms(Y, factors, quadratic=False) == []

    def test_fractional_blocks_keep_the_fraction(self):
        factors = _factors(5)
        gens = {"X4": ["X0", "X1", "X2", "X3"]}
        Y = fractional_factorial(factors, gens, n_blocks=2)
        X = Y[_names(factors)].to_numpy()
        np.testing.assert_array_equal(X[:, 4], np.prod(X[:, :4], axis=1))

    def test_main_effect_confounded_raises(self):
        """In the saturated 2^(7-4) every alias set contains a main effect, so blocks would confound one."""
        factors = _factors(7)
        with pytest.raises(AssertionError, match="confound a main effect"):
            fractional_factorial(factors, min_aberration_generators(factors, n_runs=8), n_blocks=2)

    def test_too_many_blocks_raises(self):
        with pytest.raises(AssertionError, match="too small"):
            full_factorial(_factors(3), n_blocks=8)

    def test_non_power_of_two_raises(self):
        with pytest.raises(AssertionError, match="power of 2"):
            full_factorial(_factors(4), n_blocks=3)

    def test_three_levels_raise(self):
        with pytest.raises(AssertionError, match="two-level factors"):
            full_factorial(_factors(3), n_levels=3, n_blocks=2)

    def test_factor_named_block_raises(self):
        with pytest.raises(AssertionError, match="cannot be named 'Block'"):
            full_factorial([Factor("Block"), Factor("A")], n_blocks=2)


class TestCentralCompositeBlocking:
    @pytest.mark.parametrize(
        "k, cube, n_blocks, n_runs, alpha",
        [
            (2, "full", 2, 14, 1.4142),
            (3, "full", 3, 20, 1.6330),
            (4, "full", 3, 30, 2.0000),
            (5, "full", 5, 54, 2.3664),
            (5, "half", 2, 33, 2.0000),
            (6, "full", 9, 90, 2.8284),
            (6, "half", 3, 54, 2.3664),
            (7, "full", 17, 169, 3.3333),
            (7, "half", 9, 90, 2.8284),
        ],
    )
    def test_tabulated_orthogonal_blocking(self, k, cube, n_blocks, n_runs, alpha):
        """The rotatable CCDs that block orthogonally (Box & Hunter, 1957, Table 4): runs, alpha and
        orthogonal blocks. For 7 factors with a full cube the printed alphas are swapped and for 7 factors
        with a half cube the printed total is 80 instead of 90; the values here are the corrected ones."""
        factors = _factors(k)
        Y = central_composite(factors, alpha="orthogonal", cube=cube, n_blocks=n_blocks)
        assert len(Y) == n_runs
        assert Y["Block"].nunique() == n_blocks
        np.testing.assert_allclose(Y["X0"].max(), alpha, atol=5e-5)
        assert _non_orthogonal_terms(Y, factors, quadratic=True) == []

    def test_default_alpha_is_orthogonal_when_blocked(self):
        factors = _factors(3)
        pd.testing.assert_frame_equal(
            central_composite(factors, n_blocks=3), central_composite(factors, alpha="orthogonal", n_blocks=3)
        )
        assert _non_orthogonal_terms(central_composite(factors, n_blocks=3), factors, quadratic=True) == []

    def test_axial_points_in_last_block(self):
        Y = central_composite(_factors(3), alpha="orthogonal", n_blocks=3)
        axial = Y[Y["Block"] == 3][["X0", "X1", "X2"]].to_numpy()
        assert axial.shape == (6 + 2, 3)
        np.testing.assert_array_equal(np.sum(axial != 0, axis=1), [1] * 6 + [0] * 2)

    def test_explicit_center_points(self):
        Y = central_composite(_factors(3), alpha="orthogonal", n_blocks=3, n_center=(3, 4))
        assert np.bincount(Y["Block"])[1:].tolist() == [4 + 3, 4 + 3, 6 + 4]
        assert _non_orthogonal_terms(Y, _factors(3), quadratic=True) == []

    def test_no_tabulated_blocking_requires_center_points(self):
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            central_composite(_factors(3), n_blocks=2)
        assert len(central_composite(_factors(3), n_blocks=2, n_center=2)) == 8 + 2 + 6 + 2

    def test_orthogonal_alpha_requires_blocks(self):
        with pytest.raises(AssertionError, match="requires a blocked design"):
            central_composite(_factors(3), alpha="orthogonal")


class TestBoxBehnkenBlocking:
    @pytest.mark.parametrize(
        "k, n_blocks, block_size",
        [(4, 3, 9), (5, 2, 23), (6, 2, 27), (7, 2, 31), (9, 5, 26), (9, 10, 13), (10, 2, 85), (12, 2, 102),
         (16, 6, 66), (16, 12, 33)],
    )  # fmt: skip
    def test_orthogonal_blocks_of_the_paper(self, k, n_blocks, block_size):
        """Every blocking of Box & Behnken (1960), with the block sizes stated in the paper."""
        factors = _factors(k)
        Y = box_behnken(factors, n_blocks=n_blocks)
        assert np.bincount(Y["Block"])[1:].tolist() == [block_size] * n_blocks
        assert len(Y) == len(box_behnken(factors))  # center points divided over the blocks
        assert _non_orthogonal_terms(Y, factors, quadratic=True) == []

    def test_center_points_per_block(self):
        Y = box_behnken(_factors(4), n_blocks=3, n_center=2)
        assert len(Y) == 24 + 3 * 2

    def test_invalid_number_of_blocks_raises(self):
        with pytest.raises(AssertionError, match="1 or 3 blocks"):
            box_behnken(_factors(4), n_blocks=2)
        with pytest.raises(AssertionError, match="1 or 5 or 10 blocks"):
            box_behnken(_factors(9), n_blocks=2)
        with pytest.raises(AssertionError, match="only be run in 1 block "):
            box_behnken(_factors(3), n_blocks=2)


class TestRandomizeWithinBlocks:
    def test_blocks_keep_their_order(self):
        set_seed(42)
        Y = full_factorial(_factors(4), n_blocks=4, n_center=1)
        Yr = randomize(Y)
        assert Yr["Block"].tolist() == Y["Block"].tolist()
        assert not Yr.index.equals(Y.index)
        pd.testing.assert_frame_equal(Yr.sort_index(), Y)

    def test_ignore_blocks(self):
        set_seed(42)
        Y = full_factorial(_factors(4), n_blocks=4)
        assert randomize(Y, block_column=None)["Block"].tolist() != Y["Block"].tolist()
