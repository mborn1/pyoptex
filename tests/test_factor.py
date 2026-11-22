import numpy as np
import pytest

from pyoptex.doe.cost_optimal import Factor as CostOptimalFactor
from pyoptex.doe.fixed_structure import Factor as FixedStructureFactor
from pyoptex.utils.factor import Factor as BaseFactor

factor_classes = pytest.mark.parametrize("Factor", [FixedStructureFactor, CostOptimalFactor, BaseFactor])


@factor_classes
def test_range_from_levels(Factor):
    """Without a minimum and maximum, a continuous factor spans its levels."""
    f = Factor("x", levels=[20, 30, 40])
    np.testing.assert_allclose(f.normalize(np.array(f.levels)), [-1, 0, 1])
    np.testing.assert_allclose(f.coords_[:, 0], [-1, 0, 1])


@factor_classes
def test_explicit_range_with_levels(Factor):
    f = Factor("x", min=10, max=50, levels=[20, 30, 40])
    assert (f.min, f.max) == (10, 50)
    np.testing.assert_allclose(f.normalize(np.array(f.levels)), [-0.5, 0, 0.5])


@factor_classes
def test_levels_outside_range(Factor):
    with pytest.raises(AssertionError, match="must lie between its minimum 0 and maximum 10"):
        Factor("x", min=0, max=10, levels=[5, 20])
