import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def colour():
    """colour-science, used as an independent reference implementation (optional)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pytest.importorskip("colour")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
