import numba
from pyoptex.doe.splitk_plot.utils import FunctionSet

@numba.njit
def test(a):
    return a.metric, a.constraints

a = FunctionSet(1, 2)
r = test(a)
print(r)
