import itertools

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe import fixed_structure as fs
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Coverage
from pyoptex.doe.fixed_structure import metric as fs_metric
from pyoptex.doe.utils.coverage import coverage, coverage_features
from pyoptex.utils.design import encode_design

LEVELS21 = list(np.linspace(-1, 1, 21))


def _params(factors, budget=1000, nsims=5):
    """CODEX parameters with the coverage criterion, no model and a budget."""
    cost = parallel_worker_cost({str(f.name): 1 for f in factors}, factors, budget)
    return create_parameters(factors, default_fn(nsims, factors, cost, Coverage(), None))


def _features(params, Ynorm):
    """The coverage coordinates of a normalized, decoded design."""
    Yenc = encode_design(np.array(Ynorm, dtype=float), params.effect_types, params.coords)
    return coverage_features(Yenc, params.factors, params.effect_types, params.coords)


def _fixed_design(metric, nruns, prior=None):
    """A fixed structure design of one continuous factor with 21 levels, without a model."""
    set_seed(42)
    factors = [fs.Factor("x", levels=LEVELS21)]
    params = fs.create_parameters(factors, fs.default_fn(factors, metric, None), nruns, prior=prior)
    Y, _ = fs.create_fixed_structure_design(params, n_tries=5)
    return Y["x"].to_numpy()


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
    params = _params([Factor("x", grouped=False, levels=LEVELS21)], budget=12)
    Y, _ = create_cost_optimal_codex_design(params, nsims=5, nreps=1)
    x = np.sort(Y["x"].to_numpy())
    assert len(x) >= 5
    assert np.diff(x).max() < 0.5


def test_codex_categorical_factors():
    """Only categorical factors: a budget for all 9 combinations covers each of them (a fully covered region)."""
    set_seed(42)
    factors = [Factor(name, type="categorical", levels=["L1", "L2", "L3"], grouped=False) for name in ("A", "B")]
    Y, _ = create_cost_optimal_codex_design(_params(factors, budget=20, nsims=20), nsims=20, nreps=1)
    assert set(zip(Y["A"], Y["B"], strict=True)) == set(itertools.product(["L1", "L2", "L3"], repeat=2))


def test_coverage_with_prior():
    """A fixed structure design with 3 prior runs spread over [-1, 0]: the 3 new runs fill the other half."""
    x = _fixed_design(fs_metric.Coverage(), 6, prior=pd.DataFrame({"x": [-0.8, -0.4, 0.0]}))
    assert np.all(x[3:] > 0)


def test_maximin():
    """Maximin puts 3 runs at -1, 0 and 1; with a prior run at 0, the 2 new runs at -1 and 1."""
    np.testing.assert_allclose(np.sort(_fixed_design(fs_metric.Maximin(), 3)), [-1, 0, 1])
    x = _fixed_design(fs_metric.Maximin(), 3, prior=pd.DataFrame({"x": [0.0]}))
    np.testing.assert_allclose(np.sort(x[1:]), [-1, 1])
