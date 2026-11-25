import numpy as np
import pandas as pd
import pytest

from pyoptex._seed import set_seed

pytestmark = pytest.mark.filterwarnings("ignore:Metric is .*:UserWarning")
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.init import init_warm_feasible
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.utils.design import encode_design
from pyoptex.utils.model import model2Y2X, partial_rsm_names


def test_warm_start():
    set_seed(42)

    factors = [
        Factor("A", type="categorical", levels=["L1", "L2"]),
        Factor("B", type="continuous", grouped=False),
        Factor("C", type="continuous", grouped=False, min=10, max=40),
    ]

    model = partial_rsm_names({"A": "tfi", "B": "tfi", "C": "tfi"})
    Y2X = model2Y2X(model, factors)

    metric = Iopt()
    cost_fn = parallel_worker_cost({"A": 60, "B": 1, "C": 1}, factors, 1500, execution_cost=5)
    constraints = parse_constraints_script("`B` > 0.5", factors, exclude=True)

    # Warm start in real units: 2x4x3 grid spread round-robin over 8 days of 3 runs;
    # the 6 runs with B == 1 violate the constraint
    grid = [(a, b, c) for a in ("L1", "L2") for b in (-1, -1 / 3, 1 / 3, 1) for c in (10, 25, 40)]
    df = pd.DataFrame(grid, columns=["A", "B", "C"])
    df["day"] = df.index % 8

    # Prior within the constraints
    prior = pd.DataFrame([["L2", 1 / 3, 40], ["L1", -1, 10]], columns=["A", "B", "C"])

    fn = default_fn(1, factors, cost_fn, metric, Y2X, constraints=constraints)
    params = create_parameters(factors, fn, prior=prior)

    # No n_keep, so the default path is used
    Y = init_warm_feasible(params, df, group_col="day")

    # Encode df as the function does
    Ydf = df.copy()
    for f in factors:
        Ydf[str(f.name)] = f.normalize(Ydf[str(f.name)])
    Yenc = encode_design(Ydf[[str(f.name) for f in factors]].to_numpy(), params.effect_types, coords=params.coords)

    # Prior first
    nprior = len(params.prior)
    np.testing.assert_array_equal(Y[:nprior], params.prior)

    # Every other row is a row of df
    matches = [np.flatnonzero(np.all(Yenc == row, axis=1)) for row in Y[nprior:]]
    assert all(m.size > 0 for m in matches)

    # No constraint violations
    assert not np.any(params.fn.constraints(Y))

    # Exactly the allowed runs of whole days
    rows = sorted(m[0] for m in matches)
    allowed = ~params.fn.constraints(Yenc)
    days = df["day"].to_numpy()
    expected = np.flatnonzero(np.isin(days, days[rows]) & allowed).tolist()
    assert rows == expected

    # Full rank
    X = Y2X(Y)
    assert np.linalg.matrix_rank(X) == X.shape[1]
