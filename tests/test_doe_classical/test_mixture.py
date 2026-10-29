from itertools import product
from math import comb

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import (
    Factor,
    box_behnken,
    central_composite,
    definitive_screening,
    doehlert,
    fractional_factorial,
    full_factorial,
    simplex_centroid,
    simplex_lattice,
)


def _mixture_factors(q):
    """The first q - 1 components of a q-component mixture."""
    return [Factor(f"X{i}", type="mixture") for i in range(q - 1)]


def _full_blend(Y):
    """All q components: the q - 1 design columns plus the implied last one."""
    X = Y.to_numpy(dtype=np.float64)
    return np.column_stack([X, 1 - X.sum(axis=1)])


class TestSimplexLattice:
    def test_three_two_lattice(self):
        """{3, 2} lattice: 3 pure components and 3 binary 50/50 blends."""
        Y = simplex_lattice(_mixture_factors(3), 2)
        assert isinstance(Y, pd.DataFrame)
        assert list(Y.columns) == ["X0", "X1"]
        blends = {tuple(row) for row in _full_blend(Y)}
        expected = {(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)}
        assert blends == expected

    @pytest.mark.parametrize("q, m", [(3, 1), (3, 3), (4, 2), (4, 3), (5, 4)])
    def test_run_count_and_proportions(self, q, m):
        X = _full_blend(simplex_lattice(_mixture_factors(q), m))
        assert X.shape == (comb(q + m - 1, m), q)
        np.testing.assert_allclose(X.sum(axis=1), 1.0)
        np.testing.assert_allclose(X * m, np.round(X * m), atol=1e-12)  # multiples of 1/m
        assert X.min() >= -1e-12
        assert len({tuple(row) for row in np.round(X, 12)}) == len(X)  # no duplicate blends

    def test_invalid_m_raises(self):
        with pytest.raises(AssertionError, match="m must be at least 1"):
            simplex_lattice(_mixture_factors(3), 0)


class TestSimplexCentroid:
    def test_three_components(self):
        """3 pure components, 3 binary blends and the overall centroid: 7 runs."""
        X = _full_blend(simplex_centroid(_mixture_factors(3)))
        assert X.shape == (7, 3)
        np.testing.assert_allclose(X.sum(axis=1), 1.0)
        np.testing.assert_allclose(X[-1], 1 / 3)

    @pytest.mark.parametrize("q", [2, 3, 4, 5])
    def test_run_count(self, q):
        X = _full_blend(simplex_centroid(_mixture_factors(q)))
        assert X.shape == (2**q - 1, q)
        np.testing.assert_allclose(X.sum(axis=1), 1.0)
        # Each blend has equal proportions on its non-zero components.
        for row in X:
            nonzero = row[row > 1e-12]
            np.testing.assert_allclose(nonzero, 1 / len(nonzero))


@pytest.mark.parametrize("design", [lambda f: simplex_lattice(f, 2), simplex_centroid])
class TestMixtureValidation:
    def test_non_mixture_factor_raises(self, design):
        with pytest.raises(AssertionError, match="must be a mixture factor"):
            design([Factor("A", type="mixture"), Factor("B")])

    def test_binding_upper_bound_raises(self, design):
        """B ranges up to 0.5, below its implied maximum 0.2 + (1 - 0.2) = 1: not a simplex."""
        factors = [Factor("A", type="mixture"), Factor("B", type="mixture", levels=[0.2, 0.5])]
        with pytest.raises(AssertionError, match=r"maximum 0.5, below its implied maximum 1.*not a simplex"):
            design(factors)

    def test_lower_bounds_summing_to_one_raise(self, design):
        factors = [Factor("A", type="mixture", levels=[0.5, 1]), Factor("B", type="mixture", levels=[0.5, 1])]
        with pytest.raises(AssertionError, match="sum to less than 1"):
            design(factors)


def _bounded_factors():
    """3 components with lower bounds 0.1, 0.2 and (implied, via last_min) 0.1: R = 1 - 0.4 = 0.6."""
    return [Factor("A", type="mixture", levels=[0.1, 1]), Factor("B", type="mixture", levels=[0.2, 1])]


_LOWER = np.array([0.1, 0.2, 0.1])


@pytest.mark.parametrize("design", [lambda f, **kw: simplex_lattice(f, 2, **kw), simplex_centroid])
class TestLowerBounds:
    def test_proportions(self, design):
        X = _full_blend(design(_bounded_factors(), last_min=0.1))
        np.testing.assert_allclose(X.sum(axis=1), 1.0)
        assert np.all(X >= _LOWER - 1e-12)

    def test_vertices(self, design):
        """The pure pseudocomponents are at L_i + (1 - sum L) for component i and L_j for the others."""
        X = _full_blend(design(_bounded_factors(), last_min=0.1))
        for i in range(3):
            vertex = _LOWER.copy()
            vertex[i] += 0.6
            assert np.any(np.all(np.isclose(X, vertex), axis=1)), f"vertex {vertex} missing"

    def test_transformed_from_pseudocomponents(self, design):
        """x = L + (1 - sum L) z, with z the design without bounds."""
        z = _full_blend(design(_mixture_factors(3)))
        X = _full_blend(design(_bounded_factors(), last_min=0.1))
        np.testing.assert_allclose(X, _LOWER + 0.6 * z)


class TestAugment:
    def test_centroid_design(self):
        """Simplex centroid + q axial check blends: 2^q - 1 + q runs."""
        for q in (3, 4):
            X = _full_blend(simplex_centroid(_mixture_factors(q), augment=True))
            assert X.shape == (2**q - 1 + q, q)
            axial = X[-q:]
            expected = np.full((q, q), 1 / (2 * q))
            np.fill_diagonal(expected, (q + 1) / (2 * q))
            np.testing.assert_allclose(axial, expected)

    def test_lattice_adds_centroid_when_missing(self):
        """The {3, 2} lattice has no centroid run, the {3, 3} lattice does."""
        assert len(simplex_lattice(_mixture_factors(3), 2, augment=True)) == 6 + 1 + 3
        assert len(simplex_lattice(_mixture_factors(3), 3, augment=True)) == 10 + 3

    def test_axial_blends_with_lower_bounds(self):
        X = _full_blend(simplex_centroid(_bounded_factors(), last_min=0.1, augment=True))
        np.testing.assert_allclose(X[-3], _LOWER + 0.6 * np.array([4, 1, 1]) / 6)
        np.testing.assert_allclose(X.sum(axis=1), 1.0)


def test_unchanged_without_bounds_and_augment():
    """Without lower bounds or augmentation the designs are the plain simplex designs."""
    X = simplex_lattice(_mixture_factors(4), 3).to_numpy()
    parts = sorted((p for p in product(range(4), repeat=4) if sum(p) == 3), reverse=True)
    np.testing.assert_array_equal(X, (np.array(parts, dtype=np.float64) / 3)[:, :-1])
    X = _full_blend(simplex_centroid(_mixture_factors(3)))
    assert X.shape == (7, 3)
    np.testing.assert_array_equal(X[:3], np.eye(3))


@pytest.mark.parametrize(
    "design",
    [
        full_factorial,
        lambda f: fractional_factorial(f, generators={"C": ["A", "B"]}),
        box_behnken,
        central_composite,
        definitive_screening,
        doehlert,
    ],
    ids=[
        "full_factorial",
        "fractional_factorial",
        "box_behnken",
        "central_composite",
        "definitive_screening",
        "doehlert",
    ],
)
def test_mixture_factor_raises_in_non_mixture_design(design):
    """Factorial-type and response surface designs ignore that the components sum to 1."""
    factors = [Factor("A", type="mixture"), Factor("B"), Factor("C")]
    with pytest.raises(AssertionError, match="Mixture factor A cannot be used.*simplex_lattice, simplex_centroid"):
        design(factors)
