import numpy as np
import pytest

from pyoptex.utils.runtime import set_nb_cores

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:multiprocessing.popen_fork")

set_nb_cores(1)

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.splitk_plot import Plot, create_parameters, create_splitk_plot_design, default_fn
from pyoptex.doe.fixed_structure.splitk_plot.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from pyoptex.utils.runtime import parallel_generation
from tests._helpers import assert_frame_equal, load_reference


def test_splitplot_multiprocessing():
    ref = load_reference("doe_quickstart/splitplot_multiprocessing")
    set_seed(42)

    etc = Plot(level=0, size=4)
    htc = Plot(level=1, size=5, ratio=0.1)
    plots = [etc, htc]
    nruns = np.prod([p.size for p in plots])

    factors = [
        Factor("A", htc, type="categorical", levels=["L1", "L2", "L3"], coords=[[1, 0], [0, 1], [0, 0]]),
        Factor("B", etc, type="continuous"),
        Factor("C", etc, type="continuous", min=2, max=5),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert int(nruns) == ref["nruns"]

    model = partial_rsm_names({"A": "tfi", "B": "quad", "C": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Dopt()
    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn)

    Y, state = parallel_generation(create_splitk_plot_design, params, n_tries=n_tries, ncores=2)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
