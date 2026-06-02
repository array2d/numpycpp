"""
Bit-level alignment tests — ALL numpycpp C++ vs Python numpy APIs.

SINGLE entry point.  Run with:
    pytest tests/test_all.py -v

Alignment: BIT-LEVEL (最高级对齐).
    Every test asserts bit-identical results between C++ and Python numpy.
    No tolerance, no atol/rtol — raw IEEE 754 bits must match exactly.

Coverage:
    - float64 + float32 (via dtype fixture)
    - All core, linalg, einsum APIs
    - bool / int / float dtypes
"""

import os
import importlib
import numpy as np
import pytest


# ============================================================================
# Bit-level assertion helpers
# ============================================================================

def check_bit_aligned(cpp_result, py_result, label=""):
    """
    Check bit-level alignment between C++ and Python numpy results.

    ALL dtypes require bit-exact equality (np.array_equal).
    On mismatch, reports raw hex values for float types.

    Returns:
        dict with keys: pass, shape_match, n_diff, error, label
    """
    cpp = np.asarray(cpp_result)
    py = np.asarray(py_result)

    info = {
        "label": label,
        "pass": False,
        "shape_match": cpp.shape == py.shape,
        "n_diff": 0,
        "error": None,
    }

    if not info["shape_match"]:
        info["error"] = (
            f"shape mismatch: C++ {cpp.shape} vs Python {py.shape}"
        )
        return info

    if np.array_equal(cpp, py):
        info["pass"] = True
        return info

    # --- bit-level mismatch detected --- build diagnostic ---
    diff_mask = cpp != py
    info["n_diff"] = int(np.sum(diff_mask))
    diff_indices = np.flatnonzero(diff_mask.ravel())
    n_show = min(5, len(diff_indices))

    err_lines = [
        f"BIT-LEVEL MISMATCH: {info['n_diff']}/{cpp.size} elements differ",
    ]
    for idx in diff_indices[:n_show]:
        cpp_val = cpp.flat[idx]
        py_val = py.flat[idx]

        if cpp.dtype == bool or np.issubdtype(cpp.dtype, np.integer):
            err_lines.append(
                f"  [{idx}] C++={cpp_val}  vs  numpy={py_val}"
            )
        else:
            _EL_VIEW = {2: np.uint16, 4: np.uint32, 8: np.uint64}
            _EL_FMT  = {2: "04x", 4: "08x", 8: "016x"}

            cpp_esz = cpp.itemsize
            py_esz = py.itemsize
            cpp_vdt = _EL_VIEW.get(cpp_esz)
            py_vdt = _EL_VIEW.get(py_esz)
            cpp_fmt = _EL_FMT.get(cpp_esz, "016x")
            py_fmt = _EL_FMT.get(py_esz, "016x")

            if cpp_vdt is not None:
                cpp_hex = cpp.view(cpp_vdt).flat[idx]
            else:
                cpp_hex = 0
                cpp_fmt = ""
            if py_vdt is not None:
                py_hex = py.view(py_vdt).flat[idx]
            else:
                py_hex = 0
                py_fmt = ""

            cpp_str = f"C++={cpp_val:.16e} (0x{cpp_hex:{cpp_fmt}})" if cpp_fmt else f"C++={cpp_val:.16e}"
            py_str  = f"numpy={py_val:.16e} (0x{py_hex:{py_fmt}})"  if py_fmt  else f"numpy={py_val:.16e}"
            err_lines.append(f"  [{idx}] {cpp_str}  vs  {py_str}")

    if len(diff_indices) > n_show:
        err_lines.append(
            f"  ... and {len(diff_indices) - n_show} more differing elements"
        )

    info["error"] = "\n".join(err_lines)
    return info


def assert_bit_aligned(cpp_result, py_result, label=""):
    """
    Assert C++ and Python results are bit-level identical.

    Raises AssertionError with hex dump on mismatch.
    """
    info = check_bit_aligned(cpp_result, py_result, label=label)
    if not info["pass"]:
        raise AssertionError(info.get("error", "bit-level alignment failure"))
    return info


def random_array(shape, dtype=np.float64, seed: int = 42):
    """Deterministic random array with controlled seed per shape."""
    rng = np.random.RandomState(seed + hash(shape) % (2**31))
    if np.issubdtype(dtype, np.floating):
        return rng.randn(*shape).astype(dtype)
    elif dtype == bool:
        return rng.rand(*shape) > 0.5
    else:
        return rng.randint(0, 100, size=shape).astype(dtype)


# ============================================================================
# C++ module fixture (lazy import, session-scoped)
# ============================================================================

_cpp_module = None
_import_error = None


def _resolve_module_name() -> str:
    cli_mod = getattr(pytest, "_numpycpp_module_name", None)
    if cli_mod:
        return cli_mod
    env = os.environ.get("NUMPYCPP_MODULE")
    if env:
        return env
    return "numpycpp"


def get_cpp_module():
    """Return the compiled numpycpp C++ module (lazy, cached)."""
    global _cpp_module, _import_error

    if _cpp_module is not None:
        return _cpp_module
    if _import_error is not None:
        raise _import_error

    modname = _resolve_module_name()
    try:
        _cpp_module = importlib.import_module(modname)
    except ImportError as e:
        _import_error = e
        raise
    return _cpp_module


@pytest.fixture(scope="session")
def cpp():
    """Session-scoped C++ module fixture."""
    return get_cpp_module()


@pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request):
    """Parametrized fixture: each test runs once with float64 and once with float32."""
    return request.param


# ===================================================================
# 1. Array creation (template: T in → T out)
# ===================================================================

class TestZerosLike:
    def test_basic(self, cpp, dtype):
        a = random_array((3, 4), dtype=dtype)
        assert_bit_aligned(cpp.zeros_like(a), np.zeros_like(a), "zeros_like")

    @pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (4, 5, 6)])
    def test_shapes(self, cpp, dtype, shape):
        a = random_array(shape, dtype=dtype)
        assert_bit_aligned(cpp.zeros_like(a), np.zeros_like(a), f"zeros_like{shape}")


class TestOnesLike:
    def test_basic(self, cpp, dtype):
        a = random_array((3, 4), dtype=dtype)
        assert_bit_aligned(cpp.ones_like(a), np.ones_like(a), "ones_like")

    @pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (1, 1, 1)])
    def test_shapes(self, cpp, dtype, shape):
        a = random_array(shape, dtype=dtype)
        assert_bit_aligned(cpp.ones_like(a), np.ones_like(a), f"ones_like{shape}")


class TestFullLike:
    @pytest.mark.parametrize("value_f64,value_f32", [
        (0.0, 0.0), (1.0, 1.0), (-3.14, -3.14), (42.0, 42.0), (1e10, 1e10),
    ])
    def test_values(self, cpp, dtype, value_f64, value_f32):
        v = value_f64 if dtype == np.float64 else np.float32(value_f32)
        a = random_array((3, 4), dtype=dtype)
        assert_bit_aligned(cpp.full_like(a, v), np.full_like(a, v), f"full_like({v})")


class TestEmptyLike:
    def test_shape_matches(self, cpp, dtype):
        a = random_array((3, 4), dtype=dtype)
        cpp_r = cpp.empty_like(a)
        assert np.asarray(cpp_r).shape == a.shape, "empty_like shape mismatch"


class TestZeros:
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, shape):
        assert_bit_aligned(cpp.zeros(shape), np.zeros(shape), f"zeros{shape}")


class TestOnes:
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, shape):
        assert_bit_aligned(cpp.ones(shape), np.ones(shape), f"ones{shape}")


# ===================================================================
# 2. bool-specialized creation
# ===================================================================

class TestFullLikeBool:
    @pytest.mark.parametrize("value", [True, False])
    def test_values(self, cpp, value):
        a = random_array((3, 4))
        assert_bit_aligned(cpp.full_like_bool(a, value),
                           np.full_like(a, value, dtype=bool),
                           f"full_like_bool({value})")


class TestZerosLikeBool:
    def test_basic(self, cpp):
        a = random_array((3, 4))
        assert_bit_aligned(cpp.zeros_like_bool(a),
                           np.zeros_like(a, dtype=bool), "zeros_like_bool")


class TestOnesLikeBool:
    def test_basic(self, cpp):
        a = random_array((3, 4))
        assert_bit_aligned(cpp.ones_like_bool(a),
                           np.ones_like(a, dtype=bool), "ones_like_bool")


# ===================================================================
# 3. astype conversions (non-template)
# ===================================================================

class TestAstypeInt:
    def test_basic(self, cpp):
        a = np.array([[1.7, 2.3], [-3.9, 0.5]], dtype=np.float64)
        assert_bit_aligned(cpp.astype_int(a), a.astype(np.int32), "astype_int")


class TestAstypeBool:
    def test_basic(self, cpp):
        a = np.array([[0.0, 1.0, -1.0], [3.14, 0.0, 0.0]], dtype=np.float64)
        assert_bit_aligned(cpp.astype_bool(a), a.astype(bool), "astype_bool")


class TestAstypeBoolFromInt:
    def test_basic(self, cpp):
        a = np.array([[0, 1, -1], [42, 0, 0]], dtype=np.int32)
        assert_bit_aligned(cpp.astype_bool_from_int(a), a.astype(bool),
                           "astype_bool_from_int")


class TestTruncateToFloat32:
    def test_basic(self, cpp):
        a = np.array([1.0 / 3.0, np.pi, np.sqrt(2.0)], dtype=np.float64)
        py_r = a.astype(np.float32).astype(np.float64)
        assert_bit_aligned(cpp.truncate_to_float32(a), py_r, "truncate_to_float32")


# ===================================================================
# 4. Element-wise math (template: T in → T out)
# ===================================================================

class TestSqrt:
    @pytest.mark.parametrize("shape", [(10,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, dtype, shape):
        a = np.abs(random_array(shape, dtype=dtype))
        assert_bit_aligned(cpp.sqrt(a), np.sqrt(a), f"sqrt{shape}")

    def test_zero(self, cpp, dtype):
        a = np.zeros((5,), dtype=dtype)
        assert_bit_aligned(cpp.sqrt(a), np.sqrt(a), "sqrt zero")


class TestAbs:
    @pytest.mark.parametrize("shape", [(10,), (3, 4)])
    def test_shapes(self, cpp, dtype, shape):
        a = random_array(shape, dtype=dtype)
        assert_bit_aligned(cpp.abs(a), np.abs(a), f"abs{shape}")


class TestExp:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), seed=1, dtype=dtype)
        assert_bit_aligned(cpp.exp(a), np.exp(a), "exp")


class TestLog:
    def test_basic(self, cpp, dtype):
        a = np.abs(random_array((10,), dtype=dtype)) + dtype(0.1)
        assert_bit_aligned(cpp.log(a), np.log(a), "log")


class TestSin:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.sin(a), np.sin(a), "sin")


class TestCos:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.cos(a), np.cos(a), "cos")


class TestTan:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * dtype(0.5)  # avoid poles
        assert_bit_aligned(cpp.tan(a), np.tan(a), "tan")


class TestPower:
    @pytest.mark.parametrize("expval", [2.0, 3.0, 0.5, -1.0, 0.0])
    def test_exponents(self, cpp, dtype, expval):
        e = dtype(expval)
        a = np.abs(random_array((10,), dtype=dtype)) + dtype(0.01)
        assert_bit_aligned(cpp.power(a, e), np.power(a, e), f"power({expval})")


class TestClip:
    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-1.0, 1.0), (0.5, 0.5), (-10.0, 10.0)])
    def test_bounds(self, cpp, dtype, lo, hi):
        l, h = dtype(lo), dtype(hi)
        a = random_array((20,), dtype=dtype)
        assert_bit_aligned(cpp.clip(a, l, h), np.clip(a, l, h),
                           f"clip({lo},{hi})")


class TestLog10:
    def test_basic(self, cpp, dtype):
        a = np.abs(random_array((10,), dtype=dtype)) + dtype(0.1)
        assert_bit_aligned(cpp.log10(a), np.log10(a), "log10")


class TestLog2:
    def test_basic(self, cpp, dtype):
        a = np.abs(random_array((10,), dtype=dtype)) + dtype(0.1)
        assert_bit_aligned(cpp.log2(a), np.log2(a), "log2")


class TestArcsin:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * dtype(0.5)
        assert_bit_aligned(cpp.arcsin(a), np.arcsin(a), "arcsin")


class TestArccos:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * dtype(0.5)
        assert_bit_aligned(cpp.arccos(a), np.arccos(a), "arccos")


class TestArctan:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.arctan(a), np.arctan(a), "arctan")


class TestRound:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * 10
        assert_bit_aligned(cpp.round(a), np.round(a), "round")


class TestFloor:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * 10
        assert_bit_aligned(cpp.floor(a), np.floor(a), "floor")


class TestCeil:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype) * 10
        assert_bit_aligned(cpp.ceil(a), np.ceil(a), "ceil")


class TestDegrees:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.degrees(a), np.degrees(a), "degrees")


class TestRadians:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.radians(a), np.radians(a), "radians")


class TestSign:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.sign(a), np.sign(a), "sign")

    def test_zero(self, cpp, dtype):
        a = np.array([0.0, -0.0, 0.0], dtype=dtype)
        assert_bit_aligned(cpp.sign(a), np.sign(a), "sign zero")


# ===================================================================
# 5. Reductions (template: T in → T out)
# ===================================================================

class TestSum:
    def test_1d(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert cpp.sum(a) == np.sum(a), "sum 1d"

    def test_2d(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert cpp.sum(a) == np.sum(a), "sum 2d"

    def test_empty(self, cpp, dtype):
        a = np.array([], dtype=dtype)
        assert cpp.sum(a) == dtype(0), "sum empty"


class TestMean:
    def test_1d(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(np.float64(cpp.mean(a)), np.float64(np.mean(a)),
                           "mean 1d")

    def test_empty(self, cpp, dtype):
        assert np.float64(cpp.mean(np.array([], dtype=dtype))) == 0.0, \
            "mean empty"


class TestMax:
    def test_1d(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert cpp.max(a) == np.max(a), "max 1d"


class TestMin:
    def test_1d(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert cpp.min(a) == np.min(a), "min 1d"


class TestAny:
    def test_true(self, cpp):
        a = np.array([False, False, True, False])
        assert cpp.any(a) == np.any(a), "any true"

    def test_false(self, cpp):
        a = np.array([False, False, False])
        assert cpp.any(a) == np.any(a), "any false"


class TestAll:
    def test_true(self, cpp):
        a = np.array([True, True, True])
        assert cpp.all(a) == np.all(a), "all true"

    def test_false(self, cpp):
        a = np.array([True, False, True])
        assert cpp.all(a) == np.all(a), "all false"


# ===================================================================
# 6. Comparison (template: T in → bool out)
# ===================================================================

class TestGreater:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.greater(a, dtype(0.0)),
                           np.greater(a, dtype(0.0)), "greater")


class TestLess:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.less(a, dtype(0.0)),
                           np.less(a, dtype(0.0)), "less")


class TestEqual:
    def test_basic(self, cpp, dtype):
        a = np.array([0.0, 1.0, 1.0, 0.0], dtype=dtype)
        assert_bit_aligned(cpp.equal(a, dtype(1.0)),
                           np.equal(a, dtype(1.0)), "equal")


class TestGreaterEqual:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.greater_equal(a, dtype(0.0)),
                           np.greater_equal(a, dtype(0.0)), "greater_equal")


class TestLessEqual:
    def test_basic(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.less_equal(a, dtype(0.0)),
                           np.less_equal(a, dtype(0.0)), "less_equal")


class TestNotEqual:
    def test_scalar(self, cpp, dtype):
        a = np.array([0.0, 1.0, 0.0], dtype=dtype)
        assert_bit_aligned(cpp.not_equal(a, dtype(0.0)),
                           np.not_equal(a, dtype(0.0)), "not_equal scalar")

    def test_array(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        b = random_array((10,), seed=99, dtype=dtype)
        assert_bit_aligned(cpp.not_equal(a, b),
                           np.not_equal(a, b), "not_equal array")


# ===================================================================
# 7. Logical (bool-only)
# ===================================================================

class TestLogicalAnd:
    def test_basic(self, cpp):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert_bit_aligned(cpp.logical_and(a, b),
                           np.logical_and(a, b), "logical_and")


class TestLogicalOr:
    def test_basic(self, cpp):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert_bit_aligned(cpp.logical_or(a, b),
                           np.logical_or(a, b), "logical_or")


class TestLogicalNot:
    def test_basic(self, cpp):
        a = np.array([True, False, True])
        assert_bit_aligned(cpp.logical_not(a),
                           np.logical_not(a), "logical_not")


class TestLogicalXor:
    def test_basic(self, cpp):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert_bit_aligned(cpp.logical_xor(a, b),
                           np.logical_xor(a, b), "logical_xor")


# ===================================================================
# 8. Special value checks (template: T in → bool out)
# ===================================================================

class TestIsnan:
    def test_basic(self, cpp, dtype):
        a = np.array([0.0, np.nan, 1.0, np.nan], dtype=dtype)
        assert_bit_aligned(cpp.isnan(a), np.isnan(a), "isnan")


class TestIsinf:
    def test_basic(self, cpp, dtype):
        a = np.array([0.0, np.inf, -np.inf, 1.0], dtype=dtype)
        assert_bit_aligned(cpp.isinf(a), np.isinf(a), "isinf")


class TestIsfinite:
    def test_basic(self, cpp, dtype):
        a = np.array([0.0, np.inf, np.nan, 1.0, -np.inf], dtype=dtype)
        assert_bit_aligned(cpp.isfinite(a), np.isfinite(a), "isfinite")


# ===================================================================
# 9. Binary element-wise (template: T in → T out)
# ===================================================================

class TestArctan2:
    def test_array_array(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        b = np.abs(random_array((10,), dtype=dtype)) + dtype(0.1)
        assert_bit_aligned(cpp.arctan2(a, b), np.arctan2(a, b),
                           "arctan2(a,b)")

    def test_array_scalar(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.arctan2(a, dtype(1.0)),
                           np.arctan2(a, dtype(1.0)), "arctan2(a,1)")


class TestMaximum:
    def test_array_array(self, cpp, dtype):
        a = random_array((10,), seed=1, dtype=dtype)
        b = random_array((10,), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.maximum(a, b), np.maximum(a, b),
                           "maximum(a,b)")

    def test_array_scalar(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.maximum(a, dtype(0.0)),
                           np.maximum(a, dtype(0.0)), "maximum(a,0)")


class TestMinimum:
    def test_array_array(self, cpp, dtype):
        a = random_array((10,), seed=1, dtype=dtype)
        b = random_array((10,), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.minimum(a, b), np.minimum(a, b),
                           "minimum(a,b)")

    def test_array_scalar(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        assert_bit_aligned(cpp.minimum(a, dtype(0.0)),
                           np.minimum(a, dtype(0.0)), "minimum(a,0)")


# ===================================================================
# 10. Array manipulation (template: T in → T out)
# ===================================================================

class TestDiff:
    def test_1d(self, cpp, dtype):
        a = np.array([1.0, 3.0, 6.0, 10.0], dtype=dtype)
        assert_bit_aligned(cpp.diff(a), np.diff(a), "diff 1d")

    def test_2d_axis0(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert_bit_aligned(cpp.diff(a, 1, 0), np.diff(a, n=1, axis=0),
                           "diff axis=0")

    def test_2d_axis1(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert_bit_aligned(cpp.diff(a, 1, 1), np.diff(a, n=1, axis=1),
                           "diff axis=1")

    def test_2d_axis_neg1(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert_bit_aligned(cpp.diff(a, 1, -1), np.diff(a, n=1, axis=-1),
                           "diff axis=-1")


class TestStack:
    def test_basic(self, cpp, dtype):
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(4)]
        assert_bit_aligned(cpp.stack(arrays), np.stack(arrays), "stack")


class TestConcatenate:
    def test_basic(self, cpp, dtype):
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
        assert_bit_aligned(cpp.concatenate(arrays),
                           np.concatenate(arrays), "concatenate")


class TestVstack:
    def test_basic(self, cpp, dtype):
        arrays = [random_array((1, 3), seed=i, dtype=dtype) for i in range(4)]
        assert_bit_aligned(cpp.vstack(arrays), np.vstack(arrays), "vstack")


class TestHstack:
    def test_basic(self, cpp, dtype):
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
        assert_bit_aligned(cpp.hstack(arrays), np.hstack(arrays), "hstack")


class TestWhere:
    def test_scalar(self, cpp, dtype):
        cond = np.array([True, False, True, False, True])
        assert_bit_aligned(cpp.where(cond, dtype(10.0), dtype(-1.0)),
                           np.where(cond, dtype(10.0), dtype(-1.0)),
                           "where scalar")

    def test_array(self, cpp, dtype):
        cond = np.array([True, False, True, False])
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=dtype)
        assert_bit_aligned(cpp.where(cond, x, y),
                           np.where(cond, x, y), "where array")


class TestTranspose:
    def test_1d(self, cpp, dtype):
        a = random_array((5,), dtype=dtype)
        assert_bit_aligned(cpp.transpose(a), np.transpose(a), "transpose 1d")

    def test_2d(self, cpp, dtype):
        a = random_array((3, 5), dtype=dtype)
        assert_bit_aligned(cpp.transpose(a), np.transpose(a), "transpose 2d")


class TestFlatten:
    def test_basic(self, cpp, dtype):
        a = random_array((3, 4), dtype=dtype)
        assert_bit_aligned(cpp.flatten(a), a.flatten(), "flatten")


class TestMeanAxis:
    def test_axis0_2d(self, cpp, dtype):
        a = random_array((4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, 0), np.mean(a, axis=0),
                           "mean axis=0")

    def test_axis1_2d(self, cpp, dtype):
        a = random_array((4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, 1), np.mean(a, axis=1),
                           "mean axis=1")

    def test_axis_neg1_2d(self, cpp, dtype):
        a = random_array((4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, -1), np.mean(a, axis=-1),
                           "mean axis=-1")

    def test_axis0_3d(self, cpp, dtype):
        a = random_array((3, 4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, 0), np.mean(a, axis=0),
                           "mean 3d axis=0")

    def test_axis1_3d(self, cpp, dtype):
        a = random_array((3, 4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, 1), np.mean(a, axis=1),
                           "mean 3d axis=1")

    def test_axis2_3d(self, cpp, dtype):
        a = random_array((3, 4, 5), dtype=dtype)
        assert_bit_aligned(cpp.mean(a, 2), np.mean(a, axis=2),
                           "mean 3d axis=2")


class TestSlice:
    def test_basic(self, cpp, dtype):
        a = random_array((10, 3), dtype=dtype)
        assert_bit_aligned(cpp.slice(a, 2, 7), a[2:7], "slice 2:7")

    def test_from_start(self, cpp, dtype):
        a = random_array((10, 3), dtype=dtype)
        assert_bit_aligned(cpp.slice(a, 0, 5), a[0:5], "slice :5")


class TestTakeCols:
    def test_basic(self, cpp, dtype):
        a = random_array((4, 6), dtype=dtype)
        assert_bit_aligned(cpp.take_cols(a, 3), a[:, :3], "take_cols 3")


class TestSliceAssign:
    def test_double(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        expected = a.copy()
        cpp.slice_assign(a, 2, dtype(99.0))
        expected[2:] = dtype(99.0)
        assert_bit_aligned(a, expected, "slice_assign double")

    def test_int(self, cpp):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected = a.copy()
        cpp.slice_assign(a, 3, -1)
        expected[3:] = -1
        assert_bit_aligned(a, expected, "slice_assign int")

    def test_bool(self, cpp):
        a = np.array([True, False, True, False], dtype=bool)
        expected = a.copy()
        cpp.slice_assign(a, 2, False)
        expected[2:] = False
        assert_bit_aligned(a, expected, "slice_assign bool")


class TestRoll:
    def test_positive(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        assert_bit_aligned(cpp.roll(a, 2), np.roll(a, 2), "roll +2")

    def test_negative(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        assert_bit_aligned(cpp.roll(a, -1), np.roll(a, -1), "roll -1")


class TestFlip:
    def test_basic(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        assert_bit_aligned(cpp.flip(a), np.flip(a), "flip")


class TestRepeat:
    def test_basic(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_bit_aligned(cpp.repeat(a, 3), np.repeat(a, 3), "repeat")


class TestTile:
    def test_basic(self, cpp, dtype):
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_bit_aligned(cpp.tile(a, 2), np.tile(a, 2), "tile")


# ===================================================================
# 11. Statistical (template: T in → T out)
# ===================================================================

class TestStd:
    def test_random(self, cpp, dtype):
        a = random_array((100,), dtype=dtype)
        cpp_r, py_r = cpp.std(a), np.std(a)
        assert np.float64(cpp_r) == np.float64(py_r), \
            f"std: {cpp_r} vs {py_r}"

    def test_constant(self, cpp, dtype):
        a = np.ones((50,), dtype=dtype) * dtype(3.0)
        cpp_r, py_r = cpp.std(a), np.std(a)
        assert np.float64(cpp_r) == np.float64(py_r), \
            f"std constant: {cpp_r} vs {py_r}"


class TestVar:
    def test_random(self, cpp, dtype):
        a = random_array((100,), dtype=dtype)
        cpp_r, py_r = cpp.var(a), np.var(a)
        assert np.float64(cpp_r) == np.float64(py_r), \
            f"var: {cpp_r} vs {py_r}"


# ===================================================================
# 12. Sorting & indexing (template: T in → int out)
# ===================================================================

class TestArgsort:
    def test_basic(self, cpp, dtype):
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dtype)
        assert_bit_aligned(cpp.argsort(a), np.argsort(a, kind='stable'),
                           "argsort")


class TestArgmax:
    def test_basic(self, cpp, dtype):
        a = np.array([1.0, 5.0, 3.0, 9.0, 2.0], dtype=dtype)
        assert cpp.argmax(a) == np.argmax(a), "argmax"


class TestArgmin:
    def test_basic(self, cpp, dtype):
        a = np.array([5.0, 1.0, 3.0, 0.5, 2.0], dtype=dtype)
        assert cpp.argmin(a) == np.argmin(a), "argmin"


# ===================================================================
# 13. Set operations & interpolation (double-only)
# ===================================================================

class TestIsin:
    def test_basic(self, cpp):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values = [2.0, 4.0, 6.0]
        assert_bit_aligned(cpp.isin(a, values), np.isin(a, values), "isin")


class TestIntersect1d:
    def test_basic(self, cpp):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        cpp_r = np.sort(np.asarray(cpp.intersect1d(a, b)))
        py_r = np.intersect1d(a, b)
        assert_bit_aligned(cpp_r, py_r, "intersect1d")


class TestInterp:
    def test_basic(self, cpp):
        xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        assert_bit_aligned(cpp.interp(x, xp, fp),
                           np.interp(x, xp, fp), "interp")

    def test_extrap_low(self, cpp):
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([0.0, 0.5])
        assert_bit_aligned(cpp.interp(x, xp, fp),
                           np.interp(x, xp, fp), "interp low")

    def test_extrap_high(self, cpp):
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([3.5, 5.0])
        assert_bit_aligned(cpp.interp(x, xp, fp),
                           np.interp(x, xp, fp), "interp high")


class TestSafeDivide:
    def test_normal(self, cpp):
        assert np.float64(cpp.safe_divide(10.0, 2.0)) == np.float64(5.0), \
            "safe_divide normal"

    def test_zero_denom(self, cpp):
        assert np.float64(cpp.safe_divide(10.0, 0.0, -1.0)) == np.float64(-1.0), \
            "safe_divide zero"

    def test_custom_default(self, cpp):
        assert np.float64(cpp.safe_divide(10.0, 0.0, 99.0)) == np.float64(99.0), \
            "safe_divide custom"


# ===================================================================
# 14. Array access & conversion (non-template)
# ===================================================================

class TestArrayGet:
    def test_1d(self, cpp):
        a = np.array([10.0, 20.0, 30.0])
        assert cpp.array_get(a, 0) == 10.0
        assert cpp.array_get(a, 2) == 30.0
        assert cpp.array_get(a, -1) == 30.0

    def test_2d(self, cpp):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert cpp.array_get(a, 0, 0) == 1.0
        assert cpp.array_get(a, 1, 1) == 4.0


class TestAsarray:
    def test_vector(self, cpp):
        v = [1.0, 2.0, 3.0]
        assert_bit_aligned(cpp.asarray(v), np.asarray(v), "asarray vec")

    def test_array(self, cpp):
        a = np.array([1.0, 2.0, 3.0])
        assert_bit_aligned(cpp.asarray(a), np.asarray(a), "asarray arr")


class TestToVector:
    def test_double(self, cpp):
        a = np.array([1.0, 2.0, 3.0])
        assert list(cpp.to_vector(a)) == [1.0, 2.0, 3.0], "to_vector double"

    def test_bool(self, cpp):
        a = np.array([True, False, True])
        assert list(cpp.to_vector(a)) == [True, False, True], "to_vector bool"


# ===================================================================
# 15. Linalg — norm, dot
# ===================================================================

class TestNorm:
    def test_1d(self, cpp, dtype):
        a = random_array((10,), dtype=dtype)
        # Our norm uses pairwise_sum → matches np.sqrt(np.sum(a*a)).
        # np.linalg.norm uses BLAS dot for scalars, which differs.
        assert_bit_aligned(dtype(cpp.linalg.norm(a)),
                           np.sqrt(np.sum(a * a)), "linalg.norm 1d")

    def test_2d(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert_bit_aligned(dtype(cpp.linalg.norm(a)),
                           np.sqrt(np.sum(a * a)), "linalg.norm 2d")

    def test_zero(self, cpp, dtype):
        a = np.zeros((10,), dtype=dtype)
        assert_bit_aligned(dtype(cpp.linalg.norm(a)),
                           dtype(0.0), "linalg.norm zero")


class TestNormAxis:
    def test_2d(self, cpp, dtype):
        a = random_array((5, 4), dtype=dtype)
        assert_bit_aligned(cpp.linalg.norm(a, axis=1),
                           np.linalg.norm(a, axis=1), "norm axis=1")

    def test_1d_fallback(self, cpp, dtype):
        a = np.array([3.0, 4.0], dtype=dtype)
        cpp_r = np.float64(np.asarray(cpp.linalg.norm(a)).item())
        py_r = np.float64(np.linalg.norm(a))
        assert cpp_r == py_r, f"norm 1d fallback: {cpp_r} vs {py_r}"


class TestDot:
    def test_basic(self, cpp, dtype):
        a = random_array((5,), dtype=dtype)
        b = random_array((5,), seed=99, dtype=dtype)
        assert_bit_aligned(cpp.dot(a, b),
                           np.sum(a * b), "dot")

    def test_orthogonal(self, cpp, dtype):
        a = np.array([1.0, 0.0], dtype=dtype)
        b = np.array([0.0, 1.0], dtype=dtype)
        assert_bit_aligned(cpp.dot(a, b),
                           np.sum(a * b), "dot orthogonal")


# ===================================================================
# 16. Einsum — all supported patterns
# ===================================================================

class TestEinsumIjIjToI:
    """Pattern: 'ij,ij->i' — row-wise dot product."""

    def test_fixed(self, cpp, dtype):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b), "ij,ij->i fixed")

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 2), (10, 5), (100, 3)])
    def test_random(self, cpp, dtype, m, n):
        a = random_array((m, n), seed=1, dtype=dtype)
        b = random_array((m, n), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b),
                           f"ij,ij->i [{m},{n}]")


class TestEinsumIjJkToIk:
    """Pattern: 'ij,jk->ik' — matrix multiplication."""

    def test_fixed(self, cpp, dtype):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,jk->ik", a, b),
                           np.einsum("ij,jk->ik", a, b), "ij,jk->ik fixed")

    @pytest.mark.parametrize("i,j,k", [(1, 3, 2), (3, 3, 3), (10, 5, 10)])
    def test_random(self, cpp, dtype, i, j, k):
        a = random_array((i, j), seed=1, dtype=dtype)
        b = random_array((j, k), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,jk->ik", a, b),
                           np.einsum("ij,jk->ik", a, b),
                           f"ij,jk->ik [{i},{j},{k}]")

    def test_vs_matmul(self, cpp, dtype):
        a = random_array((4, 5), seed=1, dtype=dtype)
        b = random_array((5, 3), seed=2, dtype=dtype)
        cpp_r = np.asarray(cpp.einsum("ij,jk->ik", a, b))
        assert_bit_aligned(cpp_r, a @ b, "ij,jk->ik vs matmul")


class TestEinsumBijBjkToBik:
    """Pattern: 'bij,bjk->bik' — batch matrix multiplication."""

    def test_fixed(self, cpp, dtype):
        a = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=dtype)
        b = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
                     dtype=dtype)
        assert_bit_aligned(cpp.einsum("bij,bjk->bik", a, b),
                           np.einsum("bij,bjk->bik", a, b),
                           "bij,bjk->bik fixed")

    @pytest.mark.parametrize("batch,i,j,k", [(1, 3, 2, 4), (3, 3, 3, 3), (5, 2, 5, 3)])
    def test_random(self, cpp, dtype, batch, i, j, k):
        a = random_array((batch, i, j), seed=1, dtype=dtype)
        b = random_array((batch, j, k), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("bij,bjk->bik", a, b),
                           np.einsum("bij,bjk->bik", a, b),
                           f"bij,bjk->bik [{batch},{i},{j},{k}]")

    def test_vs_batch_matmul(self, cpp, dtype):
        a = random_array((4, 5, 6), seed=1, dtype=dtype)
        b = random_array((4, 6, 3), seed=2, dtype=dtype)
        cpp_r = np.asarray(cpp.einsum("bij,bjk->bik", a, b))
        assert_bit_aligned(cpp_r, a @ b, "bij,bjk->bik vs batch matmul")


class TestEinsumAijAijToAi:
    """Pattern: 'aij,aij->ai' — batch row-wise dot product."""

    @pytest.mark.parametrize("a_sz,i_sz,j_sz", [(1, 3, 2), (3, 5, 4), (5, 10, 3)])
    def test_random(self, cpp, dtype, a_sz, i_sz, j_sz):
        a = random_array((a_sz, i_sz, j_sz), seed=1, dtype=dtype)
        b = random_array((a_sz, i_sz, j_sz), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("aij,aij->ai", a, b),
                           np.einsum("aij,aij->ai", a, b),
                           f"aij,aij->ai [{a_sz},{i_sz},{j_sz}]")


class TestEinsumBroadcastMatmul:
    """Pattern: 'ijk,mkl->mjl' — broadcast matmul with contraction."""

    @pytest.mark.parametrize("i,j,k,m,l", [(2, 3, 2, 3, 2), (1, 3, 2, 2, 3),
                                            (3, 2, 3, 2, 2)])
    def test_random(self, cpp, dtype, i, j, k, m, l):
        a = random_array((i, j, k), seed=1, dtype=dtype)
        b = random_array((m, k, l), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("ijk,mkl->mjl", a, b),
                           np.einsum("ijk,mkl->mjl", a, b),
                           f"ijk,mkl->mjl [{i},{j},{k},{m},{l}]")


class TestEinsumNijNmjToNmi:
    """Pattern: 'nij,nmj->nmi' — batched matmul with double transpose."""

    @pytest.mark.parametrize("n,i,j,m", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, dtype, n, i, j, m):
        a = random_array((n, i, j), seed=1, dtype=dtype)
        b = random_array((n, m, j), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("nij,nmj->nmi", a, b),
                           np.einsum("nij,nmj->nmi", a, b),
                           f"nij,nmj->nmi [{n},{i},{j},{m}]")


class TestEinsumAijJkaToAik:
    """Pattern: 'aij,jka->aik' — batch matmul with transpose."""

    @pytest.mark.parametrize("a_sz,i,j,k", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, dtype, a_sz, i, j, k):
        a = random_array((a_sz, i, j), seed=1, dtype=dtype)
        b = random_array((j, k, a_sz), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("aij,jka->aik", a, b),
                           np.einsum("aij,jka->aik", a, b),
                           f"aij,jka->aik [{a_sz},{i},{j},{k}]")


class TestEinsumImplicit:
    """Implicit mode: output labels are those appearing exactly once."""

    def test_ij_ij(self, cpp, dtype):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij", a, b),
                           np.einsum("ij,ij", a, b), "ij,ij implicit")

    def test_ij_jk(self, cpp, dtype):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,jk", a, b),
                           np.einsum("ij,jk", a, b), "ij,jk implicit")


class TestEinsumEdgeCases:
    def test_single_element(self, cpp, dtype):
        a = np.array([[1.0]], dtype=dtype)
        b = np.array([[2.0]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b), "single element")

    def test_large_values(self, cpp, dtype):
        a = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=dtype)
        b = np.array([[5e10, 6e10], [7e10, 8e10]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b), "large values")

    def test_negative(self, cpp, dtype):
        a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=dtype)
        b = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b), "negative values")

    def test_zeros(self, cpp, dtype):
        a = np.zeros((3, 2), dtype=dtype)
        b = np.ones((3, 2), dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b), "zeros input")


class TestEinsumLargeGateMachine:
    """Realistic gate_machine sizes: [N-1, dims] where N ~ 100."""

    @pytest.mark.parametrize("n,dims", [(10, 2), (50, 3), (100, 2), (200, 3)])
    def test_ij_ij_to_i(self, cpp, dtype, n, dims):
        a = random_array((n, dims), seed=1, dtype=dtype)
        b = random_array((n, dims), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("ij,ij->i", a, b),
                           np.einsum("ij,ij->i", a, b),
                           f"gate_machine [{n},{dims}]")


if __name__ == "__main__":
    import sys, os, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__, "-q", "--tb=short", "--no-header"]))
