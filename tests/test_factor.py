import pickle

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
@pytest.mark.parametrize("kind", ["cont_mixture", "mixture"])
def test_mixture_not_rescaled(Factor, kind):
    """
    A mixture component keeps its fractions: its levels are not rescaled to their range. Its min and max are
    fixed at -1 and 1 internally: explicit other bounds raise, and unpickling (which passes -1 and 1) works.
    """
    f = Factor("x", type=kind, levels=[0, 0.5, 1])
    np.testing.assert_allclose(f.normalize(np.array(f.levels)), [0, 0.5, 1])

    # A pickled mixture component unpickles, with min -1 and max 1
    restored = pickle.loads(pickle.dumps(f))
    assert (restored.type, restored.min, restored.max) == ("cont_mixture", -1, 1)
    np.testing.assert_allclose(restored.normalize(np.array(restored.levels)), [0, 0.5, 1])

    # Explicit bounds other than -1 and 1 raise
    with pytest.raises(AssertionError, match="min and max are fixed at -1 and 1 internally"):
        Factor("x", type="cont_mixture", min=0, max=1, levels=[0, 0.5, 1])