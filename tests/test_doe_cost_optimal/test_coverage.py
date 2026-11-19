import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Coverage
from pyoptex.doe.utils.coverage import coverage, coverage_features
from pyoptex.utils.design import encode_design


def _params(factors, budget=1000):
    """CODEX parameters with the coverage criterion, no model and a budget."""
    cost = parallel_worker_cost({str(f.name): 1 for f in factors}, factors, budget)
    return create_parameters(factors, default_fn(5, factors, cost, Coverage()))


def _features(params, Ynorm):
    """The coverage coordinates of a normalized, decoded design."""
    return coverage_features(encode_design(np.array(Ynorm, dtype=float), params.effect_types, params.coords), params)


def test_coverage():
    """Runs at -1 and 1 and samples at -1, 0 and 1: the distances to the nearest run are 0, 0.5 and 0."""
    params = _params([Factor("x", grouped=False)])
    runs = _features(params, [[-1], [1]])
    samples = _features(params, [[-1], [0], [1]])
    np.testing.assert_allclose(coverage(runs, samples, 2), -np.sqrt(0.25 / 3))


def test_coverage_features():
    """Another level of a categorical factor is at distance 1, the same level at 0; a continuous range counts as 1."""
    factors = [
        Factor("C", type="categorical", levels=["L1", "L2", "L3"], grouped=False),
        Factor("x", min=2, max=5, grouped=False),
    ]
    params = _params(factors)
    F = _features(params, [[0, -1], [1, -1], [0, -1], [0, 1]])  # (L1, 2), (L2, 2), (L1, 2), (L1, 5)

    def distance(i, j):
        return np.linalg.norm(F[i] - F[j])

    np.testing.assert_allclose([distance(0, 1), distance(0, 2), distance(0, 3)], [1, 0, 1])


def test_codex_uses_the_budget():
    """The score improves with every run added, so CODEX spreads about 6 runs over the range (maximin would stop at 2)."""
    set_seed(42)
    params = _params([Factor("x", grouped=False, levels=list(np.linspace(-1, 1, 21)))], budget=12)
    Y, _ = create_cost_optimal_codex_design(params, nsims=5, nreps=1)
    x = np.sort(Y["x"].to_numpy())
    assert len(x) >= 5
    assert np.diff(x).max() < 0.5
