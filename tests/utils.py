"""
Shared test utilities: comparison engine, data generators.

Import from here in test files; conftest.py imports these for fixtures.
"""

import numpy as np


def compare(
    cpp_result,
    py_result,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    label: str = "",
):
    """
    Compare C++ result against Python (ground-truth) numpy result.

    Returns a dict with: pass, max_abs_diff, max_rel_diff, shape_match,
    cpp_shape, py_shape, cpp_dtype, py_dtype, label.
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

    py_abs = np.abs(py)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(py_abs > 0, abs_diff / py_abs, abs_diff)
    max_rel = float(np.max(rel_diff))

    passed = bool(np.allclose(cpp, py, rtol=rtol, atol=atol))

    info["pass"] = passed
    info["max_abs_diff"] = max_abs
    info["max_rel_diff"] = max_rel

    if not passed:
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
    """Like compare() but raises AssertionError on mismatch."""
    info = compare(cpp_result, py_result, rtol=rtol, atol=atol, label=label)
    if not info["pass"]:
        raise AssertionError(info.get("error", "mismatch"))
    return info


def tolerance_for(dtype, rtol, atol):
    """Return (rtol, atol) appropriate for the given dtype."""
    if dtype == np.float32:
        return max(rtol, 1e-6), max(atol, 1e-6)
    return rtol, atol


def random_array(shape, dtype=np.float64, seed: int = 42):
    """Deterministic random array with controlled seed per shape."""
    rng = np.random.RandomState(seed + hash(shape) % (2**31))
    if np.issubdtype(dtype, np.floating):
        return rng.randn(*shape).astype(dtype)
    elif dtype == bool:
        return rng.rand(*shape) > 0.5
    else:
        return rng.randint(0, 100, size=shape).astype(dtype)
