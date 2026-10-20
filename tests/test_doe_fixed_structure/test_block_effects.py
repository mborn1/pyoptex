import numpy as np
import pytest

from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_parameters, default_fn
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

NRUNS = 12


def _groups(size):
    """Grouping of the 12 runs into consecutive groups of the given size."""
    return np.repeat(np.arange(NRUNS // size), size).astype(np.int64)


def _expected_V(effects):
    """I + sum(ratio_i * Z_i Z_i^T), with Z_i the one-hot matrix of each grouping."""
    V = np.eye(NRUNS)
    for effect in effects:
        Z = np.eye(effect.Z.max() + 1)[effect.Z]
        V += effect.ratio * Z @ Z.T
    return V


@pytest.mark.parametrize(
    "factor_effects, block_effects",
    [
        # 1 factor random effect, 2 block effects
        (
            [RandomEffect(_groups(4), ratio=2.0)],
            [RandomEffect(_groups(6), ratio=3.0), RandomEffect(_groups(3), ratio=5.0)],
        ),
        # 2 factor random effects, 1 block effect
        (
            [RandomEffect(_groups(4), ratio=2.0), RandomEffect(_groups(2), ratio=7.0)],
            [RandomEffect(_groups(6), ratio=3.0)],
        ),
    ],
    ids=["1 factor, 2 blocks", "2 factors, 1 block"],
)
def test_block_effects_with_factor_random_effects(factor_effects, block_effects):
    """The covariance combines every factor random effect and block effect with its own ratio."""
    factors = [Factor(f"A{i}", re) for i, re in enumerate(factor_effects)] + [Factor("B")]
    model = partial_rsm_names({str(f.name): "lin" for f in factors})
    Y2X = model2Y2X(model, factors)
    params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), NRUNS, block_effects=block_effects)

    np.testing.assert_allclose(np.linalg.inv(params.Vinv[0]), _expected_V(factor_effects + block_effects), atol=1e-10)
