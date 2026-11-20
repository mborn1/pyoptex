import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe import fixed_structure as fs
from pyoptex.doe.fixed_structure import metric as fs_metric
from pyoptex.doe.fixed_structure.splitk_plot import Plot, create_parameters, create_splitk_plot_design, default_fn
from pyoptex.doe.fixed_structure.splitk_plot.metric import Coverage
from pyoptex.doe.fixed_structure.splitk_plot.metric import Maximin as SplitkMaximin
from tests._helpers import assert_frame_equal, load_reference

LEVELS21 = list(np.linspace(-1, 1, 21))


def _fixed_design(metric, nruns, prior=None):
    """A fixed structure design of one continuous factor with 21 levels, without a model."""
    set_seed(42)
    factors = [fs.Factor("x", levels=LEVELS21)]
    params = fs.create_parameters(factors, fs.default_fn(factors, metric, None), nruns, prior=prior)
    Y, _ = fs.create_fixed_structure_design(params, n_tries=5)
    return Y["x"].to_numpy()


def test_coverage_with_prior():
    """A fixed structure design with 3 prior runs spread over [-1, 0]: the 3 new runs fill the other half."""
    x = _fixed_design(fs_metric.Coverage(), 6, prior=pd.DataFrame({"x": [-0.8, -0.4, 0.0]}))
    assert np.all(x[3:] > 0)


def test_maximin():
    """Maximin puts 3 runs at -1, 0 and 1; with a prior run at 0, the 2 new runs at -1 and 1."""
    np.testing.assert_allclose(np.sort(_fixed_design(fs_metric.Maximin(), 3)), [-1, 0, 1])
    x = _fixed_design(fs_metric.Maximin(), 3, prior=pd.DataFrame({"x": [0.0]}))
    np.testing.assert_allclose(np.sort(x[1:]), [-1, 1])


def test_maximin_replicated_prior():
    """A prior with two runs at 0 does not hide the new runs: the 2 new runs go to -1 and 1."""
    x = _fixed_design(fs_metric.Maximin(), 4, prior=pd.DataFrame({"x": [0.0, 0.0]}))
    np.testing.assert_allclose(np.sort(x[2:]), [-1, 1])


def test_splitk_maximin_replicated_prior():
    """
    A split-plot prior with a replicate in each of its 3 whole plots, extended from 2 to 4 runs per whole plot:
    the replicates, spread over the plots, do not hide the new runs, which move away from them.
    """
    set_seed(42)
    etc, htc = Plot(level=0, size=4), Plot(level=1, size=3)
    factors = [fs.Factor("W", htc, levels=LEVELS21), fs.Factor("S", etc, levels=LEVELS21)]
    prior = pd.DataFrame({"W": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], "S": [0.0, 0.0, 0.5, 0.5, -0.5, -0.5]})
    prior_plots = [Plot(level=0, size=2), Plot(level=1, size=3)]
    params = create_parameters(factors, default_fn(factors, SplitkMaximin(), None), prior=(prior, prior_plots))
    Y, state = create_splitk_plot_design(params, n_tries=5)

    assert state.metric > -1e5
    S = Y["S"].to_numpy().reshape(3, 4)  # one row per whole plot, the prior's two runs first
    np.testing.assert_array_equal(S[:, :2], prior["S"].to_numpy().reshape(3, 2))
    assert np.all(S[:, 2:] != S[:, :1])


def test_splitk_coverage():
    """The split-plot coverage example: one hard-to-change factor over 5 whole plots, without a model."""
    ref = load_reference("doe_fixed_structure/splitk_coverage")
    set_seed(42)

    etc = Plot(level=0, size=4)
    htc = Plot(level=1, size=5)
    factors = [
        fs.Factor("A", htc, type="continuous", levels=np.linspace(-1, 1, 11)),
        fs.Factor("B", etc, type="continuous", levels=np.linspace(-1, 1, 11)),
        fs.Factor("C", etc, type="continuous", min=2, max=5, levels=np.linspace(2, 5, 11)),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert etc.size * htc.size == ref["nruns"]

    fn = default_fn(factors, Coverage(), None)
    params = create_parameters(factors, fn)

    Y, state = create_splitk_plot_design(params, n_tries=5, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
