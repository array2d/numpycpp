"""
Pytest fixtures and hooks for numpycpp precision alignment tests.

Usage:
  pytest tests/                          # run all tests
  pytest tests/ --cpp-module=my_module   # custom C++ module name
  pytest tests/ --rtol=1e-10 --atol=1e-10  # custom tolerances
"""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cpp-module", action="store", default=None,
        help="Python module name for the compiled C++ numpycpp library (default: numpycpp)",
    )
    parser.addoption(
        "--rtol", action="store", type=float, default=1e-12,
        help="Relative tolerance for numerical comparisons (default: 1e-12)",
    )
    parser.addoption(
        "--atol", action="store", type=float, default=1e-12,
        help="Absolute tolerance for numerical comparisons (default: 1e-12)",
    )


# -- Lazy C++ module import -----------------------------------------------

_cpp_module = None
_import_error = None


def _resolve_module_name(config) -> str:
    import os
    cli = config.getoption("--cpp-module", default=None)
    if cli:
        return cli
    env = os.environ.get("NUMPYCPP_MODULE")
    if env:
        return env
    return "numpycpp"


def get_cpp_module(request=None):
    """Return the compiled numpycpp C++ module (lazy, cached)."""
    global _cpp_module, _import_error

    if _cpp_module is not None:
        return _cpp_module
    if _import_error is not None:
        raise _import_error

    if request is not None:
        modname = _resolve_module_name(request.config)
    else:
        import os
        modname = os.environ.get("NUMPYCPP_MODULE", "numpycpp")

    import importlib
    try:
        _cpp_module = importlib.import_module(modname)
    except ImportError as e:
        _import_error = e
        raise
    return _cpp_module


# -- Fixtures --------------------------------------------------------------


@pytest.fixture(scope="session")
def cpp():
    """Session-scoped C++ module fixture."""
    return get_cpp_module()


@pytest.fixture
def rtol(request):
    return request.config.getoption("--rtol", default=1e-12)


@pytest.fixture
def atol(request):
    return request.config.getoption("--atol", default=1e-12)


@pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request):
    """Parametrized fixture: each test runs once with float64 and once with float32."""
    return request.param


def tolerance_for(dtype, rtol, atol):
    """Return (rtol, atol) appropriate for the given dtype."""
    if dtype == np.float32:
        return max(rtol, 1e-6), max(atol, 1e-6)
    return rtol, atol
