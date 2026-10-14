from math import comb

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, simplex_centroid, simplex_lattice


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

    def test_constrained_component_raises(self, design):
        factors = [Factor("A", type="mixture"), Factor("B", type="mixture", levels=[0.2, 0.5])]
        with pytest.raises(AssertionError, match="must range from 0 to 1"):
            design(factors)
