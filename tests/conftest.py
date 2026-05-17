"""
Precision alignment test framework for numpcpp vs Python numpy.

Design:
  Each test calls a C++ function (via the compiled numpcpp module) and the
  equivalent Python numpy function with identical inputs, then compares results
  using configurable tolerances.

  The C++ module name is configurable via --cpp-module or NUMPYCPP_MODULE env var.
  Default: "numpycpp".

Usage:
  pytest tests/                          # run all tests
  pytest tests/ --cpp-module=my_module   # custom C++ module name
  pytest tests/ -v --tb=short            # verbose, short tracebacks
  pytest tests/ --rtol=1e-10 --atol=1e-10  # custom tolerances
"""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cpp-module",
        action="store",
        default=None,
        help="Python module name for the compiled C++ numpcpp library (default: numpcpp)",
    )
    parser.addoption(
        "--rtol",
        action="store",
        type=float,
        default=1e-12,
        help="Relative tolerance for numerical comparisons (default: 1e-12)",
    )
    parser.addoption(
        "--atol",
        action="store",
        type=float,
        default=1e-12,
        help="Absolute tolerance for numerical comparisons (default: 1e-12)",
    )


# ---------------------------------------------------------------------------
# Shared state: lazily-imported C++ module
# ---------------------------------------------------------------------------
_cpp_module = None
_import_error = None


def _resolve_module_name(config) -> str:
    """Resolve C++ module name from CLI, env, or default."""
    import os

    cli = config.getoption("--cpp-module", default=None)
    if cli:
        return cli
    env = os.environ.get("NUMPYCPP_MODULE")
    if env:
        return env
    return "numpycpp"


def get_cpp_module(request=None) -> "module":
    """
    Return the compiled numpcpp C++ module.  Import is attempted once and
    cached; if it fails, all dependent tests are skipped with a clear message.
    """
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


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


def compare(
    cpp_result,
    py_result,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    label: str = "",
):
    """
    Compare C++ result against Python (ground-truth) result.

    Returns a dict with keys: pass, max_abs_diff, max_rel_diff, shape_match,
    cpp_dtype, py_dtype, label.

    Does NOT raise on mismatch — returns structured diagnostics.
    """
    cpp = np.asarray(cpp_result, dtype=np.float64)
    py = np.asarray(py_result, dtype=np.float64)

    info = {
        "label": label,
        "shape_match": cpp.shape == py.shape,
        "cpp_shape": cpp.shape,
        "py_shape": py.shape,
        "cpp_dtype": str(cpp.dtype),
        "py_dtype": str(py.dtype),
    }

    if not info["shape_match"]:
        info["pass"] = False
        info["max_abs_diff"] = float("nan")
        info["max_rel_diff"] = float("nan")
        info["error"] = f"shape mismatch: C++ {cpp.shape} vs Python {py.shape}"
        return info

    abs_diff = np.abs(cpp - py)
    max_abs = float(np.max(abs_diff))

    # Relative diff: avoid division by zero
    py_abs = np.abs(py)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(py_abs > 0, abs_diff / py_abs, abs_diff)
    max_rel = float(np.max(rel_diff))

    passed = bool(np.allclose(cpp, py, rtol=rtol, atol=atol))

    info["pass"] = passed
    info["max_abs_diff"] = max_abs
    info["max_rel_diff"] = max_rel

    if not passed:
        # Find worst offenders for diagnostics
        worst_idx = int(np.argmax(abs_diff))
        info["error"] = (
            f"numerical mismatch: max_abs_diff={max_abs:.2e}, "
            f"max_rel_diff={max_rel:.2e} at idx {worst_idx}\n"
            f"  C++ value: {cpp.flat[worst_idx]:.16e}\n"
            f"  Py  value: {py.flat[worst_idx]:.16e}"
        )

    return info


def assert_match(
    cpp_result,
    py_result,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    label: str = "",
):
    """
    Like compare(), but raises AssertionError on mismatch (for use in plain
    unittest-style tests).
    """
    info = compare(cpp_result, py_result, rtol=rtol, atol=atol, label=label)
    if not info["pass"]:
        raise AssertionError(info.get("error", "mismatch"))
    return info


# ---------------------------------------------------------------------------
# Shared test-data generators
# ---------------------------------------------------------------------------


def random_array(shape, dtype=np.float64, seed: int = 42):
    """Deterministic random array with controlled seed per shape."""
    rng = np.random.RandomState(seed + hash(shape) % (2**31))
    if np.issubdtype(dtype, np.floating):
        return rng.randn(*shape).astype(dtype)
    elif dtype == bool:
        return rng.rand(*shape) > 0.5
    else:
        return rng.randint(0, 100, size=shape).astype(dtype)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
