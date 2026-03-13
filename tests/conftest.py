import pytest

from pyoptex._seed import set_seed


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fix all random seeds (numpy + numba + cython) for reproducible tests."""
    set_seed(42)
    yield
