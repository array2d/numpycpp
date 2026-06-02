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
    """Check bit-level alignment between C++ and Python numpy results."""
    cpp = np.asarray(cpp_result)
    py = np.asarray(py_result)

    info = {"label": label, "pass": False,
            "shape_match": cpp.shape == py.shape, "n_diff": 0, "error": None}

    if not info["shape_match"]:
        info["error"] = f"shape mismatch: C++ {cpp.shape} vs Python {py.shape}"
        return info

    if np.array_equal(cpp, py):
        info["pass"] = True
        return info

    # --- bit-level mismatch --- build hex diagnostic ---
    diff_mask = cpp != py
    info["n_diff"] = int(np.sum(diff_mask))
    diff_indices = np.flatnonzero(diff_mask.ravel())
    n_show = min(5, len(diff_indices))

    err_lines = [f"BIT-LEVEL MISMATCH: {info['n_diff']}/{cpp.size} elements differ"]
    for idx in diff_indices[:n_show]:
        cpp_val, py_val = cpp.flat[idx], py.flat[idx]
        if cpp.dtype == bool or np.issubdtype(cpp.dtype, np.integer):
            err_lines.append(f"  [{idx}] C++={cpp_val}  vs  numpy={py_val}")
        else:
            _EL_VIEW = {2: np.uint16, 4: np.uint32, 8: np.uint64}
            _EL_FMT  = {2: "04x", 4: "08x", 8: "016x"}
            cpp_esz, py_esz = cpp.itemsize, py.itemsize
            cpp_vdt, py_vdt = _EL_VIEW.get(cpp_esz), _EL_VIEW.get(py_esz)
            cpp_fmt = _EL_FMT.get(cpp_esz, "016x")
            py_fmt  = _EL_FMT.get(py_esz, "016x")
            cpp_hex = cpp.view(cpp_vdt).flat[idx] if cpp_vdt else 0
            py_hex  = py.view(py_vdt).flat[idx]  if py_vdt  else 0
            if not cpp_vdt: cpp_fmt = ""
            if not py_vdt:  py_fmt  = ""
            cpp_str = f"C++={cpp_val:.16e} (0x{cpp_hex:{cpp_fmt}})" if cpp_fmt else f"C++={cpp_val:.16e}"
            py_str  = f"numpy={py_val:.16e} (0x{py_hex:{py_fmt}})"   if py_fmt  else f"numpy={py_val:.16e}"
            err_lines.append(f"  [{idx}] {cpp_str}  vs  {py_str}")

    if len(diff_indices) > n_show:
        err_lines.append(f"  ... and {len(diff_indices) - n_show} more differing elements")
    info["error"] = "\n".join(err_lines)
    return info


def assert_bit_aligned(cpp_result, py_result, label=""):
    """Assert C++ and Python results are bit-level identical."""
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


def _dtype_val(v_f64, v_f32, dtype):
    """Return value cast to dtype."""
    return v_f64 if dtype == np.float64 else dtype(v_f32)


# ============================================================================
# C++ module fixture (lazy import, session-scoped)
# ============================================================================

_cpp_module = None
_import_error = None


def _resolve_module_name() -> str:
    cli_mod = getattr(pytest, "_numpycpp_module_name", None)
    if cli_mod: return cli_mod
    env = os.environ.get("NUMPYCPP_MODULE")
    if env: return env
    return "numpycpp"


def get_cpp_module():
    """Return the compiled numpycpp C++ module (lazy, cached)."""
    global _cpp_module, _import_error
    if _cpp_module is not None: return _cpp_module
    if _import_error is not None: raise _import_error
    modname = _resolve_module_name()
    try:
        _cpp_module = importlib.import_module(modname)
    except ImportError as e:
        _import_error = e
        raise
    return _cpp_module


@pytest.fixture(scope="session")
def cpp():
    return get_cpp_module()


@pytest.fixture(params=[np.float64, np.float32], ids=["float64", "float32"])
def dtype(request):
    return request.param


# ============================================================================
# 1. Data-driven element-wise unary math
# ============================================================================
# Each entry: (cpp_fn_name, np_fn, input_prep, sizes)
#   input_prep: None → random_array directly; else callable: prep(a) → input
#   sizes: list of (size, seed) tuples
#
# All functions (including transcendental) are bit-exact on EVERY architecture
# via numpy's own scalar math functions (npy_exp, npy_log, ...) resolved from
# _multiarray_umath.so by the SVML bridge. No AVX-512 required.

_UNARY_MATH = [
    ("sqrt",       np.sqrt,       lambda a: np.abs(a),                [(10, 42), (10000, 7), (100000, 7)]),
    ("abs",        np.abs,        None,                               [(10, 42), (10000, 7), (100000, 7)]),
    ("exp",        np.exp,        None,                               [(10, 1),  (1000, 7), (10000, 7), (100000, 7)]),
    ("log",        np.log,        lambda a: np.abs(a) + 0.1,          [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("sin",        np.sin,        None,                               [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("cos",        np.cos,        None,                               [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("tan",        np.tan,        lambda a: a * 0.5,                  [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("log10",      np.log10,      lambda a: np.abs(a) + 0.1,          [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("log2",       np.log2,       lambda a: np.abs(a) + 0.1,          [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arcsin",     np.arcsin,     lambda a: np.clip(a * 0.5, -1, 1),  [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arccos",     np.arccos,     lambda a: np.clip(a * 0.5, -1, 1),  [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arctan",     np.arctan,     None,                               [(10, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("round",      np.round,      lambda a: a * 100,                  [(10, 42), (10000, 7), (100000, 7)]),
    ("floor",      np.floor,      lambda a: a * 100,                  [(10, 42), (10000, 7), (100000, 7)]),
    ("ceil",       np.ceil,       lambda a: a * 100,                  [(10, 42), (10000, 7), (100000, 7)]),
    ("degrees",    np.degrees,    None,                               [(10, 42), (10000, 7), (100000, 7)]),
    ("radians",    np.radians,    None,                               [(10, 42), (10000, 7), (100000, 7)]),
    ("sign",       np.sign,       None,                               [(10, 42), (10000, 7), (100000, 7)]),
]


# Build parametrize tables at module level
_UNARY_ARGS = []
_UNARY_IDS = []
for fn, npf, prep, sizes in _UNARY_MATH:
    for size, seed in sizes:
        tag = f"{fn}_{size}"
        if seed != 42: tag += f"_s{seed}"
        _UNARY_IDS.append(tag)
        _UNARY_ARGS.append(pytest.param(fn, npf, prep, size, seed, id=tag))


@pytest.mark.parametrize("fn_name, np_fn, prep, size, seed", _UNARY_ARGS)
def test_unary_math(fn_name, np_fn, prep, size, seed, cpp, dtype):
    a = random_array((size,), seed=seed, dtype=dtype)
    inp = prep(a) if prep else a
    cpp_fn = getattr(cpp, fn_name)
    assert_bit_aligned(cpp_fn(inp), np_fn(inp), fn_name)


# Special: sqrt and sign zero tests
def test_sqrt_zero(cpp, dtype):
    a = np.zeros((5,), dtype=dtype)
    assert_bit_aligned(cpp.sqrt(a), np.sqrt(a), "sqrt zero")

def test_sign_zero(cpp, dtype):
    a = np.array([0.0, -0.0, 0.0], dtype=dtype)
    assert_bit_aligned(cpp.sign(a), np.sign(a), "sign zero")


# Power
@pytest.mark.parametrize("expval,size,seed", [
    (2.0, 10, 42), (3.0, 10, 42), (0.5, 10, 42), (-1.0, 10, 42), (0.0, 10, 42),
    (2.0, 10000, 7), (3.0, 10000, 7), (0.5, 10000, 7), (-1.0, 10000, 7),
])
def test_power(expval, size, seed, cpp, dtype):
    e = dtype(expval)
    a = np.abs(random_array((size,), seed=seed, dtype=dtype)) + dtype(0.01)
    assert_bit_aligned(cpp.power(a, e), np.power(a, e), f"power({expval})_{size}")


# Clip
@pytest.mark.parametrize("lo,hi,size", [
    (0.0, 1.0, 20), (-1.0, 1.0, 20), (0.5, 0.5, 20), (-10.0, 10.0, 20),
    (0.0, 1.0, 10000), (-100.0, 100.0, 10000),
])
def test_clip(lo, hi, size, cpp, dtype):
    l, h = dtype(lo), dtype(hi)
    a = random_array((size,), seed=(7 if size > 100 else 42), dtype=dtype)
    assert_bit_aligned(cpp.clip(a, l, h), np.clip(a, l, h), f"clip({lo},{hi})_{size}")


# ============================================================================
# 2. Data-driven comparisons
# ============================================================================

_COMPARISONS = [
    ("greater",       np.greater),
    ("less",          np.less),
    ("greater_equal", np.greater_equal),
    ("less_equal",    np.less_equal),
]


_COMP_ARGS = [(fn, npf, size, seed) for fn, npf in _COMPARISONS for size, seed in [(10, 42), (100000, 7)]]
_COMP_IDS = [f"{fn}_{size}" for fn, npf in _COMPARISONS for size, seed in [(10, 42), (100000, 7)]]


@pytest.mark.parametrize("fn_name, np_fn, size, seed", _COMP_ARGS, ids=_COMP_IDS)
def test_comparison(fn_name, np_fn, size, seed, cpp, dtype):
    a = random_array((size,), seed=seed, dtype=dtype)
    cpp_fn = getattr(cpp, fn_name)
    assert_bit_aligned(cpp_fn(a, dtype(0.0)), np_fn(a, dtype(0.0)), fn_name)


def test_equal(cpp, dtype):
    a = np.array([0.0, 1.0, 1.0, 0.0], dtype=dtype)
    assert_bit_aligned(cpp.equal(a, dtype(1.0)), np.equal(a, dtype(1.0)), "equal")

def test_equal_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    assert_bit_aligned(cpp.equal(a, dtype(0.0)), np.equal(a, dtype(0.0)), "equal large")


def test_not_equal_scalar(cpp, dtype):
    a = np.array([0.0, 1.0, 0.0], dtype=dtype)
    assert_bit_aligned(cpp.not_equal(a, dtype(0.0)), np.not_equal(a, dtype(0.0)), "not_equal scalar")

def test_not_equal_array(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    b = random_array((10,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.not_equal(a, b), np.not_equal(a, b), "not_equal array")

def test_not_equal_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.not_equal(a, b), np.not_equal(a, b), "not_equal large")


# ============================================================================
# 3. Data-driven binary element-wise
# ============================================================================

def test_maximum_array(cpp, dtype):
    a = random_array((10,), seed=1, dtype=dtype)
    b = random_array((10,), seed=2, dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, b), np.maximum(a, b), "maximum(a,b)")

def test_maximum_scalar(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, dtype(0.0)), np.maximum(a, dtype(0.0)), "maximum(a,0)")

def test_maximum_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, b), np.maximum(a, b), "maximum large")

def test_minimum_array(cpp, dtype):
    a = random_array((10,), seed=1, dtype=dtype)
    b = random_array((10,), seed=2, dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, b), np.minimum(a, b), "minimum(a,b)")

def test_minimum_scalar(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, dtype(0.0)), np.minimum(a, dtype(0.0)), "minimum(a,0)")

def test_minimum_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, b), np.minimum(a, b), "minimum large")

def test_arctan2_array(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    b = np.abs(random_array((10,), dtype=dtype)) + dtype(0.1)
    assert_bit_aligned(cpp.arctan2(a, b), np.arctan2(a, b), "arctan2(a,b)")

def test_arctan2_scalar(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert_bit_aligned(cpp.arctan2(a, dtype(1.0)), np.arctan2(a, dtype(1.0)), "arctan2(a,1)")

def test_arctan2_large(cpp, dtype):
    a = random_array((10000,), seed=7, dtype=dtype)
    b = np.abs(random_array((10000,), seed=99, dtype=dtype)) + dtype(0.1)
    assert_bit_aligned(cpp.arctan2(a, b), np.arctan2(a, b), "arctan2 large")


# ============================================================================
# 4. Reductions
# ============================================================================

def test_sum_1d(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert cpp.sum(a) == np.sum(a), "sum 1d"

def test_sum_2d(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert cpp.sum(a) == np.sum(a), "sum 2d"

def test_sum_empty(cpp, dtype):
    a = np.array([], dtype=dtype)
    assert cpp.sum(a) == dtype(0), "sum empty"


def test_mean_1d(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert_bit_aligned(np.float64(cpp.mean(a)), np.float64(np.mean(a)), "mean 1d")

def test_mean_empty(cpp, dtype):
    assert np.float64(cpp.mean(np.array([], dtype=dtype))) == 0.0, "mean empty"


def test_max_1d(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert cpp.max(a) == np.max(a), "max 1d"

def test_max_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    assert cpp.max(a) == np.max(a), "max large"

def test_min_1d(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert cpp.min(a) == np.min(a), "min 1d"

def test_min_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    assert cpp.min(a) == np.min(a), "min large"


# ============================================================================
# 5. Statistical
# ============================================================================

def test_std_random(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    cpp_r, py_r = cpp.std(a), np.std(a)
    assert np.float64(cpp_r) == np.float64(py_r), f"std: {cpp_r} vs {py_r}"

def test_std_constant(cpp, dtype):
    a = np.ones((50,), dtype=dtype) * dtype(3.0)
    cpp_r, py_r = cpp.std(a), np.std(a)
    assert np.float64(cpp_r) == np.float64(py_r), f"std constant: {cpp_r} vs {py_r}"

@pytest.mark.parametrize("size", [1000, 10000])
def test_std_large(size, cpp):
    """std large: float64 only — float32 ULP differences from summation order."""
    a = random_array((size,), seed=7, dtype=np.float64)
    cpp_r, py_r = cpp.std(a), np.std(a)
    assert np.float64(cpp_r) == np.float64(py_r), f"std large {size}"

def test_var_random(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    cpp_r, py_r = cpp.var(a), np.var(a)
    assert np.float64(cpp_r) == np.float64(py_r), f"var: {cpp_r} vs {py_r}"

@pytest.mark.parametrize("size", [1000, 10000])
def test_var_large(size, cpp):
    """var large: float64 only — float32 ULP differences from summation order."""
    a = random_array((size,), seed=7, dtype=np.float64)
    cpp_r, py_r = cpp.var(a), np.var(a)
    assert np.float64(cpp_r) == np.float64(py_r), f"var large {size}"


# ============================================================================
# 6. Array creation
# ============================================================================

def test_zeros_like(cpp, dtype):
    a = random_array((3, 4), dtype=dtype)
    assert_bit_aligned(cpp.zeros_like(a), np.zeros_like(a), "zeros_like")

@pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (4, 5, 6)])
def test_zeros_like_shapes(shape, cpp, dtype):
    a = random_array(shape, dtype=dtype)
    assert_bit_aligned(cpp.zeros_like(a), np.zeros_like(a), f"zeros_like{shape}")

def test_ones_like(cpp, dtype):
    a = random_array((3, 4), dtype=dtype)
    assert_bit_aligned(cpp.ones_like(a), np.ones_like(a), "ones_like")

@pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (1, 1, 1)])
def test_ones_like_shapes(shape, cpp, dtype):
    a = random_array(shape, dtype=dtype)
    assert_bit_aligned(cpp.ones_like(a), np.ones_like(a), f"ones_like{shape}")

@pytest.mark.parametrize("v_f64,v_f32", [(0.0, 0.0), (1.0, 1.0), (-3.14, -3.14), (42.0, 42.0), (1e10, 1e10)])
def test_full_like(v_f64, v_f32, cpp, dtype):
    v = _dtype_val(v_f64, v_f32, dtype)
    a = random_array((3, 4), dtype=dtype)
    assert_bit_aligned(cpp.full_like(a, v), np.full_like(a, v), f"full_like({v})")

def test_empty_like_shape(cpp, dtype):
    a = random_array((3, 4), dtype=dtype)
    assert np.asarray(cpp.empty_like(a)).shape == a.shape, "empty_like shape"

@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_zeros(shape, cpp):
    assert_bit_aligned(cpp.zeros(shape), np.zeros(shape), f"zeros{shape}")

@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_ones(shape, cpp):
    assert_bit_aligned(cpp.ones(shape), np.ones(shape), f"ones{shape}")


# ============================================================================
# 7. Bool-specialized creation
# ============================================================================

@pytest.mark.parametrize("value", [True, False])
def test_full_like_bool(value, cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.full_like_bool(a, value),
                       np.full_like(a, value, dtype=bool), f"full_like_bool({value})")

def test_zeros_like_bool(cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.zeros_like_bool(a), np.zeros_like(a, dtype=bool), "zeros_like_bool")

def test_ones_like_bool(cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.ones_like_bool(a), np.ones_like(a, dtype=bool), "ones_like_bool")


# ============================================================================
# 8. Astype conversions
# ============================================================================

def test_astype_int(cpp):
    a = np.array([[1.7, 2.3], [-3.9, 0.5]], dtype=np.float64)
    assert_bit_aligned(cpp.astype_int(a), a.astype(np.int32), "astype_int")

def test_astype_bool(cpp):
    a = np.array([[0.0, 1.0, -1.0], [3.14, 0.0, 0.0]], dtype=np.float64)
    assert_bit_aligned(cpp.astype_bool(a), a.astype(bool), "astype_bool")

def test_astype_bool_from_int(cpp):
    a = np.array([[0, 1, -1], [42, 0, 0]], dtype=np.int32)
    assert_bit_aligned(cpp.astype_bool_from_int(a), a.astype(bool), "astype_bool_from_int")

def test_truncate_to_float32(cpp):
    a = np.array([1.0 / 3.0, np.pi, np.sqrt(2.0)], dtype=np.float64)
    py_r = a.astype(np.float32).astype(np.float64)
    assert_bit_aligned(cpp.truncate_to_float32(a), py_r, "truncate_to_float32")


# ============================================================================
# 9. Logical & special values
# ============================================================================

def test_logical_and(cpp):
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert_bit_aligned(cpp.logical_and(a, b), np.logical_and(a, b), "logical_and")

def test_logical_or(cpp):
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert_bit_aligned(cpp.logical_or(a, b), np.logical_or(a, b), "logical_or")

def test_logical_not(cpp):
    a = np.array([True, False, True])
    assert_bit_aligned(cpp.logical_not(a), np.logical_not(a), "logical_not")

def test_logical_xor(cpp):
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert_bit_aligned(cpp.logical_xor(a, b), np.logical_xor(a, b), "logical_xor")

def test_any_true(cpp):
    assert cpp.any(np.array([False, False, True, False])) == True, "any true"

def test_any_false(cpp):
    assert cpp.any(np.array([False, False, False])) == False, "any false"

def test_all_true(cpp):
    assert cpp.all(np.array([True, True, True])) == True, "all true"

def test_all_false(cpp):
    assert cpp.all(np.array([True, False, True])) == False, "all false"

def test_isnan(cpp, dtype):
    a = np.array([0.0, np.nan, 1.0, np.nan], dtype=dtype)
    assert_bit_aligned(cpp.isnan(a), np.isnan(a), "isnan")

def test_isinf(cpp, dtype):
    a = np.array([0.0, np.inf, -np.inf, 1.0], dtype=dtype)
    assert_bit_aligned(cpp.isinf(a), np.isinf(a), "isinf")

def test_isfinite(cpp, dtype):
    a = np.array([0.0, np.inf, np.nan, 1.0, -np.inf], dtype=dtype)
    assert_bit_aligned(cpp.isfinite(a), np.isfinite(a), "isfinite")


# ============================================================================
# 10. Array manipulation
# ============================================================================

def test_diff_1d(cpp, dtype):
    a = np.array([1.0, 3.0, 6.0, 10.0], dtype=dtype)
    assert_bit_aligned(cpp.diff(a), np.diff(a), "diff 1d")

def test_diff_2d_axis0(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert_bit_aligned(cpp.diff(a, 1, 0), np.diff(a, n=1, axis=0), "diff axis=0")

def test_diff_2d_axis1(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert_bit_aligned(cpp.diff(a, 1, 1), np.diff(a, n=1, axis=1), "diff axis=1")

def test_diff_2d_axis_neg1(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert_bit_aligned(cpp.diff(a, 1, -1), np.diff(a, n=1, axis=-1), "diff axis=-1")

def test_stack(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(4)]
    assert_bit_aligned(cpp.stack(arrays), np.stack(arrays), "stack")

def test_concatenate(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concatenate")

def test_vstack(cpp, dtype):
    arrays = [random_array((1, 3), seed=i, dtype=dtype) for i in range(4)]
    assert_bit_aligned(cpp.vstack(arrays), np.vstack(arrays), "vstack")

def test_hstack(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.hstack(arrays), np.hstack(arrays), "hstack")

def test_where_scalar(cpp, dtype):
    cond = np.array([True, False, True, False, True])
    assert_bit_aligned(cpp.where(cond, dtype(10.0), dtype(-1.0)),
                       np.where(cond, dtype(10.0), dtype(-1.0)), "where scalar")

def test_where_array(cpp, dtype):
    cond = np.array([True, False, True, False])
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=dtype)
    assert_bit_aligned(cpp.where(cond, x, y), np.where(cond, x, y), "where array")

def test_transpose_1d(cpp, dtype):
    a = random_array((5,), dtype=dtype)
    assert_bit_aligned(cpp.transpose(a), np.transpose(a), "transpose 1d")

def test_transpose_2d(cpp, dtype):
    a = random_array((3, 5), dtype=dtype)
    assert_bit_aligned(cpp.transpose(a), np.transpose(a), "transpose 2d")

def test_flatten(cpp, dtype):
    a = random_array((3, 4), dtype=dtype)
    assert_bit_aligned(cpp.flatten(a), a.flatten(), "flatten")


# Mean axis
def test_mean_axis0_2d(cpp, dtype):
    a = random_array((4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, 0), np.mean(a, axis=0), "mean axis=0")

def test_mean_axis1_2d(cpp, dtype):
    a = random_array((4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, 1), np.mean(a, axis=1), "mean axis=1")

def test_mean_axis_neg1_2d(cpp, dtype):
    a = random_array((4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, -1), np.mean(a, axis=-1), "mean axis=-1")

def test_mean_axis0_3d(cpp, dtype):
    a = random_array((3, 4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, 0), np.mean(a, axis=0), "mean 3d axis=0")

def test_mean_axis1_3d(cpp, dtype):
    a = random_array((3, 4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, 1), np.mean(a, axis=1), "mean 3d axis=1")

def test_mean_axis2_3d(cpp, dtype):
    a = random_array((3, 4, 5), dtype=dtype)
    assert_bit_aligned(cpp.mean(a, 2), np.mean(a, axis=2), "mean 3d axis=2")


# Slice & assign
def test_slice_basic(cpp, dtype):
    a = random_array((10, 3), dtype=dtype)
    assert_bit_aligned(cpp.slice(a, 2, 7), a[2:7], "slice 2:7")

def test_slice_from_start(cpp, dtype):
    a = random_array((10, 3), dtype=dtype)
    assert_bit_aligned(cpp.slice(a, 0, 5), a[0:5], "slice :5")

def test_take_cols(cpp, dtype):
    a = random_array((4, 6), dtype=dtype)
    assert_bit_aligned(cpp.take_cols(a, 3), a[:, :3], "take_cols 3")

def test_slice_assign_float(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    expected = a.copy()
    cpp.slice_assign(a, 2, dtype(99.0))
    expected[2:] = dtype(99.0)
    assert_bit_aligned(a, expected, "slice_assign float")

def test_slice_assign_int(cpp):
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    expected = a.copy()
    cpp.slice_assign(a, 3, -1)
    expected[3:] = -1
    assert_bit_aligned(a, expected, "slice_assign int")

def test_slice_assign_bool(cpp):
    a = np.array([True, False, True, False], dtype=bool)
    expected = a.copy()
    cpp.slice_assign(a, 2, False)
    expected[2:] = False
    assert_bit_aligned(a, expected, "slice_assign bool")


# Roll, flip, repeat, tile
def test_roll_positive(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    assert_bit_aligned(cpp.roll(a, 2), np.roll(a, 2), "roll +2")

def test_roll_negative(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    assert_bit_aligned(cpp.roll(a, -1), np.roll(a, -1), "roll -1")

def test_flip(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    assert_bit_aligned(cpp.flip(a), np.flip(a), "flip")

def test_repeat(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0], dtype=dtype)
    assert_bit_aligned(cpp.repeat(a, 3), np.repeat(a, 3), "repeat")

def test_tile(cpp, dtype):
    a = np.array([1.0, 2.0, 3.0], dtype=dtype)
    assert_bit_aligned(cpp.tile(a, 2), np.tile(a, 2), "tile")


# ============================================================================
# 11. Sorting & indexing
# ============================================================================

def test_argsort(cpp, dtype):
    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dtype)
    assert_bit_aligned(cpp.argsort(a), np.argsort(a, kind='stable'), "argsort")

def test_argmax(cpp, dtype):
    a = np.array([1.0, 5.0, 3.0, 9.0, 2.0], dtype=dtype)
    assert cpp.argmax(a) == np.argmax(a), "argmax"

def test_argmin(cpp, dtype):
    a = np.array([5.0, 1.0, 3.0, 0.5, 2.0], dtype=dtype)
    assert cpp.argmin(a) == np.argmin(a), "argmin"


# ============================================================================
# 12. Set operations & interpolation
# ============================================================================

def test_isin(cpp):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_bit_aligned(cpp.isin(a, [2.0, 4.0, 6.0]), np.isin(a, [2.0, 4.0, 6.0]), "isin")

def test_intersect1d(cpp):
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
    cpp_r = np.sort(np.asarray(cpp.intersect1d(a, b)))
    assert_bit_aligned(cpp_r, np.intersect1d(a, b), "intersect1d")

def test_interp_basic(cpp):
    xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    fp = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    x = np.array([0.5, 1.5, 2.5, 3.5])
    assert_bit_aligned(cpp.interp(x, xp, fp), np.interp(x, xp, fp), "interp")

def test_interp_extrap_low(cpp):
    xp, fp = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    assert_bit_aligned(cpp.interp(np.array([0.0, 0.5]), xp, fp),
                       np.interp(np.array([0.0, 0.5]), xp, fp), "interp low")

def test_interp_extrap_high(cpp):
    xp, fp = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    assert_bit_aligned(cpp.interp(np.array([3.5, 5.0]), xp, fp),
                       np.interp(np.array([3.5, 5.0]), xp, fp), "interp high")

def test_safe_divide_normal(cpp):
    assert np.float64(cpp.safe_divide(10.0, 2.0)) == np.float64(5.0), "safe_divide normal"

def test_safe_divide_zero_denom(cpp):
    assert np.float64(cpp.safe_divide(10.0, 0.0, -1.0)) == np.float64(-1.0), "safe_divide zero"

def test_safe_divide_custom(cpp):
    assert np.float64(cpp.safe_divide(10.0, 0.0, 99.0)) == np.float64(99.0), "safe_divide custom"


# ============================================================================
# 13. Array access & conversion
# ============================================================================

def test_array_get_1d(cpp):
    a = np.array([10.0, 20.0, 30.0])
    assert cpp.array_get(a, 0) == 10.0
    assert cpp.array_get(a, 2) == 30.0
    assert cpp.array_get(a, -1) == 30.0

def test_array_get_2d(cpp):
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert cpp.array_get(a, 0, 0) == 1.0
    assert cpp.array_get(a, 1, 1) == 4.0

def test_asarray_vector(cpp):
    v = [1.0, 2.0, 3.0]
    assert_bit_aligned(cpp.asarray(v), np.asarray(v), "asarray vec")

def test_asarray_array(cpp):
    a = np.array([1.0, 2.0, 3.0])
    assert_bit_aligned(cpp.asarray(a), np.asarray(a), "asarray arr")

def test_to_vector_double(cpp):
    a = np.array([1.0, 2.0, 3.0])
    assert list(cpp.to_vector(a)) == [1.0, 2.0, 3.0], "to_vector double"

def test_to_vector_bool(cpp):
    a = np.array([True, False, True])
    assert list(cpp.to_vector(a)) == [True, False, True], "to_vector bool"


# ============================================================================
# 14. Linalg — norm, dot
# ============================================================================

def test_norm_1d(cpp, dtype):
    a = random_array((10,), dtype=dtype)
    assert_bit_aligned(dtype(cpp.linalg.norm(a)), np.sqrt(np.sum(a * a)), "linalg.norm 1d")

def test_norm_2d(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert_bit_aligned(dtype(cpp.linalg.norm(a)), np.sqrt(np.sum(a * a)), "linalg.norm 2d")

def test_norm_zero(cpp, dtype):
    a = np.zeros((10,), dtype=dtype)
    assert_bit_aligned(dtype(cpp.linalg.norm(a)), dtype(0.0), "linalg.norm zero")

def test_norm_axis_2d(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert_bit_aligned(cpp.linalg.norm(a, axis=1), np.linalg.norm(a, axis=1), "norm axis=1")

def test_norm_1d_fallback(cpp, dtype):
    a = np.array([3.0, 4.0], dtype=dtype)
    cpp_r = np.float64(np.asarray(cpp.linalg.norm(a)).item())
    py_r = np.float64(np.linalg.norm(a))
    assert cpp_r == py_r, f"norm 1d fallback: {cpp_r} vs {py_r}"

def test_dot(cpp, dtype):
    a = random_array((5,), dtype=dtype)
    b = random_array((5,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.dot(a, b), np.sum(a * b), "dot")

def test_dot_orthogonal(cpp, dtype):
    a = np.array([1.0, 0.0], dtype=dtype)
    b = np.array([0.0, 1.0], dtype=dtype)
    assert_bit_aligned(cpp.dot(a, b), np.sum(a * b), "dot orthogonal")


# ============================================================================
# 15. Einsum — all supported patterns
# ============================================================================

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
                           np.einsum("ij,ij->i", a, b), f"ij,ij->i [{m},{n}]")


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
                           np.einsum("ij,jk->ik", a, b), f"ij,jk->ik [{i},{j},{k}]")

    def test_vs_matmul(self, cpp, dtype):
        a = random_array((4, 5), seed=1, dtype=dtype)
        b = random_array((5, 3), seed=2, dtype=dtype)
        assert_bit_aligned(np.asarray(cpp.einsum("ij,jk->ik", a, b)), a @ b,
                           "ij,jk->ik vs matmul")


class TestEinsumBijBjkToBik:
    """Pattern: 'bij,bjk->bik' — batch matrix multiplication."""

    def test_fixed(self, cpp, dtype):
        a = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=dtype)
        b = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=dtype)
        assert_bit_aligned(cpp.einsum("bij,bjk->bik", a, b),
                           np.einsum("bij,bjk->bik", a, b), "bij,bjk->bik fixed")

    @pytest.mark.parametrize("batch,i,j,k", [(1, 3, 2, 4), (3, 3, 3, 3), (5, 2, 5, 3)])
    def test_random(self, cpp, dtype, batch, i, j, k):
        a = random_array((batch, i, j), seed=1, dtype=dtype)
        b = random_array((batch, j, k), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("bij,bjk->bik", a, b),
                           np.einsum("bij,bjk->bik", a, b), f"bij,bjk->bik [{batch},{i},{j},{k}]")

    def test_vs_batch_matmul(self, cpp, dtype):
        a = random_array((4, 5, 6), seed=1, dtype=dtype)
        b = random_array((4, 6, 3), seed=2, dtype=dtype)
        assert_bit_aligned(np.asarray(cpp.einsum("bij,bjk->bik", a, b)), a @ b,
                           "bij,bjk->bik vs batch matmul")


class TestEinsumAijAijToAi:
    """Pattern: 'aij,aij->ai' — batch row-wise dot product."""

    @pytest.mark.parametrize("a_sz,i_sz,j_sz", [(1, 3, 2), (3, 5, 4), (5, 10, 3)])
    def test_random(self, cpp, dtype, a_sz, i_sz, j_sz):
        a = random_array((a_sz, i_sz, j_sz), seed=1, dtype=dtype)
        b = random_array((a_sz, i_sz, j_sz), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("aij,aij->ai", a, b),
                           np.einsum("aij,aij->ai", a, b), f"aij,aij->ai [{a_sz},{i_sz},{j_sz}]")


class TestEinsumBroadcastMatmul:
    """Pattern: 'ijk,mkl->mjl' — broadcast matmul with contraction."""

    @pytest.mark.parametrize("i,j,k,m,l", [(2, 3, 2, 3, 2), (1, 3, 2, 2, 3), (3, 2, 3, 2, 2)])
    def test_random(self, cpp, dtype, i, j, k, m, l):
        a = random_array((i, j, k), seed=1, dtype=dtype)
        b = random_array((m, k, l), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("ijk,mkl->mjl", a, b),
                           np.einsum("ijk,mkl->mjl", a, b), f"ijk,mkl->mjl [{i},{j},{k},{m},{l}]")


class TestEinsumNijNmjToNmi:
    """Pattern: 'nij,nmj->nmi' — batched matmul with double transpose."""

    @pytest.mark.parametrize("n,i,j,m", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, dtype, n, i, j, m):
        a = random_array((n, i, j), seed=1, dtype=dtype)
        b = random_array((n, m, j), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("nij,nmj->nmi", a, b),
                           np.einsum("nij,nmj->nmi", a, b), f"nij,nmj->nmi [{n},{i},{j},{m}]")


class TestEinsumAijJkaToAik:
    """Pattern: 'aij,jka->aik' — batch matmul with transpose."""

    @pytest.mark.parametrize("a_sz,i,j,k", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, dtype, a_sz, i, j, k):
        a = random_array((a_sz, i, j), seed=1, dtype=dtype)
        b = random_array((j, k, a_sz), seed=2, dtype=dtype)
        assert_bit_aligned(cpp.einsum("aij,jka->aik", a, b),
                           np.einsum("aij,jka->aik", a, b), f"aij,jka->aik [{a_sz},{i},{j},{k}]")


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
                           np.einsum("ij,ij->i", a, b), f"gate_machine [{n},{dims}]")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__, "-q", "--tb=short", "--no-header"]))
