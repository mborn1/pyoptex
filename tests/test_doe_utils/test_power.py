import numpy as np
import pytest
from scipy import stats

from pyoptex.doe.utils.power import t_test_power


def test_matches_noncentral_f():
    """For one parameter, the two-sided t-test power equals the noncentral F(1, df) power."""
    var = np.array([0.125, 0.5])
    df = 6.0
    expected = stats.ncf.sf(stats.f.ppf(0.95, 1, df), 1, df, 1.0 / var)
    np.testing.assert_allclose(t_test_power(var, df), expected, rtol=1e-8)


def test_per_parameter_degrees_of_freedom():
    """Non-integer, per-parameter degrees of freedom (as from Kenward-Roger): fewer df means less power."""
    var = np.full(3, 0.2)
    pw = t_test_power(var, np.array([2.5, 7.3, 30.0]))
    assert pw[0] < pw[1] < pw[2]
    np.testing.assert_allclose(pw[1], t_test_power(var[:1], 7.3)[0])


def test_zero_effect_gives_alpha():
    np.testing.assert_allclose(t_test_power(np.array([0.1, 1.0]), 5, effect_size=0.0, alpha=0.1), 0.1, rtol=1e-8)


def test_not_estimable_is_zero():
    assert t_test_power(np.array([np.inf]), 5)[0] == 0.0


def test_invalid_inputs_raise():
    with pytest.raises(AssertionError, match="alpha"):
        t_test_power(np.array([0.1]), 5, alpha=1.5)
    with pytest.raises(AssertionError, match="degrees of freedom"):
        t_test_power(np.array([0.1]), 0)
