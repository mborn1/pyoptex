import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fix numpy random seed for reproducible tests."""
    np.random.seed(42)
    yield
