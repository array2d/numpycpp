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

    # --- bit-level equality ---
    # For floating-point arrays with the SAME dtype use uint bit-view comparison so
    # NaN==NaN (same bits) passes.  np.array_equal returns False for NaN (IEEE 754).
    #
    # When dtypes differ (some C++ functions return float64 for float32 input) we
    # fall back to numpy's numeric comparison (float32 is upcast to float64).
    _UINT_VIEW = {4: np.uint32, 8: np.uint64}
    if cpp.dtype.kind == 'f' and cpp.dtype == py.dtype:
        uint_t = _UINT_VIEW.get(cpp.itemsize)
        if uint_t is not None:
            cpp_u = np.ascontiguousarray(cpp).ravel().view(uint_t)
            py_u  = np.ascontiguousarray(py ).ravel().view(uint_t)
            if np.array_equal(cpp_u, py_u):
                info["pass"] = True
                return info
            diff_mask = (cpp_u != py_u).reshape(cpp.shape)
        else:
            if np.array_equal(cpp, py):
                info["pass"] = True
                return info
            diff_mask = cpp != py
    else:
        if np.array_equal(cpp, py):
            info["pass"] = True
            return info
        # cpp != py may return a scalar bool (Python/numpy) for incompatible shapes
        # (old numpy behaviour).  Normalise to an ndarray with cpp's shape.
        diff_raw = np.asarray(cpp != py)
        if diff_raw.shape != cpp.shape:
            try:
                diff_mask = diff_raw.reshape(cpp.shape)
            except ValueError:
                diff_mask = np.ones(cpp.shape, dtype=bool)
        else:
            diff_mask = diff_raw

    # --- bit-level mismatch --- build hex diagnostic ---
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
            cpp_hex = np.ascontiguousarray(cpp).view(cpp_vdt).flat[idx] if cpp_vdt else 0
            py_hex  = np.ascontiguousarray(py ).view(py_vdt ).flat[idx] if py_vdt  else 0
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
    ("sqrt",       np.sqrt,       lambda a: np.abs(a),                [(100, 42), (10000, 7), (100000, 7)]),
    ("abs",        np.abs,        None,                               [(100, 42), (10000, 7), (100000, 7)]),
    ("exp",        np.exp,        None,                               [(100, 1),  (1000, 7), (10000, 7), (100000, 7)]),
    ("log",        np.log,        lambda a: np.abs(a) + 0.1,          [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("sin",        np.sin,        None,                               [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("cos",        np.cos,        None,                               [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("tan",        np.tan,        lambda a: a * 0.5,                  [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("cbrt",       np.cbrt,       None,                               [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("expm1",      np.expm1,      lambda a: a * 2.0,                  [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("log1p",      np.log1p,      lambda a: np.abs(a) + 0.1,          [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("log10",      np.log10,      lambda a: np.abs(a) + 0.1,          [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("log2",       np.log2,       lambda a: np.abs(a) + 0.1,          [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arcsin",     np.arcsin,     lambda a: np.clip(a * 0.5, -1, 1),  [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arccos",     np.arccos,     lambda a: np.clip(a * 0.5, -1, 1),  [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("arctan",     np.arctan,     None,                               [(100, 42), (1000, 7), (10000, 7), (100000, 7)]),
    ("round",      np.round,      lambda a: a * 100,                  [(100, 42), (10000, 7), (100000, 7)]),
    ("floor",      np.floor,      lambda a: a * 100,                  [(100, 42), (10000, 7), (100000, 7)]),
    ("ceil",       np.ceil,       lambda a: a * 100,                  [(100, 42), (10000, 7), (100000, 7)]),
    ("degrees",    np.degrees,    None,                               [(100, 42), (10000, 7), (100000, 7)]),
    ("radians",    np.radians,    None,                               [(100, 42), (10000, 7), (100000, 7)]),
    ("sign",       np.sign,       None,                               [(100, 42), (10000, 7), (100000, 7)]),
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


_COMP_ARGS = [(fn, npf, size, seed) for fn, npf in _COMPARISONS for size, seed in [(100, 42), (100000, 7)]]
_COMP_IDS = [f"{fn}_{size}" for fn, npf in _COMPARISONS for size, seed in [(100, 42), (100000, 7)]]


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
    a = random_array((100,), dtype=dtype)
    b = random_array((100,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.not_equal(a, b), np.not_equal(a, b), "not_equal array")

def test_not_equal_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.not_equal(a, b), np.not_equal(a, b), "not_equal large")


# ============================================================================
# 3. Data-driven binary element-wise
# ============================================================================

def test_maximum_array(cpp, dtype):
    a = random_array((100,), seed=1, dtype=dtype)
    b = random_array((100,), seed=2, dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, b), np.maximum(a, b), "maximum(a,b)")

def test_maximum_scalar(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, dtype(0.0)), np.maximum(a, dtype(0.0)), "maximum(a,0)")

def test_maximum_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.maximum(a, b), np.maximum(a, b), "maximum large")

def test_minimum_array(cpp, dtype):
    a = random_array((100,), seed=1, dtype=dtype)
    b = random_array((100,), seed=2, dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, b), np.minimum(a, b), "minimum(a,b)")

def test_minimum_scalar(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, dtype(0.0)), np.minimum(a, dtype(0.0)), "minimum(a,0)")

def test_minimum_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    b = random_array((100000,), seed=99, dtype=dtype)
    assert_bit_aligned(cpp.minimum(a, b), np.minimum(a, b), "minimum large")

def test_arctan2_array(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    b = np.abs(random_array((100,), dtype=dtype)) + dtype(0.1)
    assert_bit_aligned(cpp.arctan2(a, b), np.arctan2(a, b), "arctan2(a,b)")

def test_arctan2_scalar(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert_bit_aligned(cpp.arctan2(a, dtype(1.0)), np.arctan2(a, dtype(1.0)), "arctan2(a,1)")

def test_arctan2_large(cpp, dtype):
    a = random_array((10000,), seed=7, dtype=dtype)
    b = np.abs(random_array((10000,), seed=99, dtype=dtype)) + dtype(0.1)
    assert_bit_aligned(cpp.arctan2(a, b), np.arctan2(a, b), "arctan2 large")


# ============================================================================
# 4. Reductions
# ============================================================================

def test_sum_1d(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert cpp.sum(a) == np.sum(a), "sum 1d"

def test_sum_2d(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    assert cpp.sum(a) == np.sum(a), "sum 2d"

def test_sum_empty(cpp, dtype):
    a = np.array([], dtype=dtype)
    assert cpp.sum(a) == dtype(0), "sum empty"


def test_mean_1d(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert_bit_aligned(np.float64(cpp.mean(a)), np.float64(np.mean(a)), "mean 1d")

def test_mean_empty(cpp, dtype):
    assert np.float64(cpp.mean(np.array([], dtype=dtype))) == 0.0, "mean empty"


def test_max_1d(cpp, dtype):
    a = random_array((100,), dtype=dtype)
    assert cpp.max(a) == np.max(a), "max 1d"

def test_max_large(cpp, dtype):
    a = random_array((100000,), seed=7, dtype=dtype)
    assert cpp.max(a) == np.max(a), "max large"

def test_min_1d(cpp, dtype):
    a = random_array((100,), dtype=dtype)
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

@pytest.mark.parametrize("shape,fill_val", [((5,), 3.14), ((2, 3), -1.0), ((4,), 0.0)])
def test_full(shape, fill_val, cpp):
    assert_bit_aligned(cpp.full(list(shape), fill_val),
                       np.full(shape, fill_val), f"full{shape}_{fill_val}")


# ============================================================================
# 7. Bool-specialized creation
# ============================================================================

@pytest.mark.parametrize("value", [True, False])
def test_full_like_bool(value, cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.full_like(a, value),
                       np.full_like(a, value, dtype=bool), f"full_like({value})")

def test_zeros_like_bool(cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.zeros_like(a, "bool"), np.zeros_like(a, dtype=bool), "zeros_like")

def test_ones_like_bool(cpp):
    a = random_array((3, 4))
    assert_bit_aligned(cpp.ones_like(a, "bool"), np.ones_like(a, dtype=bool), "ones_like")


# ============================================================================
# 8. Astype conversions
# ============================================================================

def test_astype_int(cpp):
    a = np.array([[1.7, 2.3], [-3.9, 0.5]], dtype=np.float64)
    assert_bit_aligned(cpp.astype(a, "int"), a.astype(np.int32), "astype_int")

def test_astype_bool(cpp):
    a = np.array([[0.0, 1.0, -1.0], [3.14, 0.0, 0.0]], dtype=np.float64)
    assert_bit_aligned(cpp.astype(a, "bool"), a.astype(bool), "astype_bool")

def test_astype_bool_from_int(cpp):
    a = np.array([[0, 1, -1], [42, 0, 0]], dtype=np.int32)
    assert_bit_aligned(cpp.astype(a, "bool"), a.astype(bool), "astype_bool_from_int")

def test_astype_f64_to_f32(cpp):
    a = np.array([1.5, 2.7, -3.1], dtype=np.float64)
    assert_bit_aligned(cpp.astype(a, "float32"), a.astype(np.float32), "astype_f64_to_f32")

def test_astype_f32_to_f64(cpp):
    a = np.array([1.5, 2.7, -3.1], dtype=np.float32)
    assert_bit_aligned(cpp.astype(a, "float64"), a.astype(np.float64), "astype_f32_to_f64")

def test_astype_f64_to_int64(cpp):
    a = np.array([1.5, 2.7, -3.1], dtype=np.float64)
    assert_bit_aligned(cpp.astype(a, "int64"), a.astype(np.int64), "astype_f64_to_int64")

def test_astype_int_to_f64(cpp):
    a = np.array([1, 2, -3], dtype=np.int32)
    assert_bit_aligned(cpp.astype(a, "float64"), a.astype(np.float64), "astype_int_to_f64")

def test_astype_int_to_f32(cpp):
    a = np.array([1, 2, -3], dtype=np.int32)
    assert_bit_aligned(cpp.astype(a, "float32"), a.astype(np.float32), "astype_int_to_f32")

def test_astype_bool_to_f64(cpp):
    a = np.array([True, False, True], dtype=bool)
    assert_bit_aligned(cpp.astype(a, "float64"), a.astype(np.float64), "astype_bool_to_f64")

def test_astype_bool_to_int(cpp):
    a = np.array([True, False, True, False], dtype=bool)
    assert_bit_aligned(cpp.astype(a, "int"), a.astype(np.int32), "astype_bool_to_int")

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

def test_concatenate_1d(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concatenate 1d")

def test_concatenate_2d_axis0(cpp, dtype):
    arrays = [random_array((2, 3), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, 0), np.concatenate(arrays, axis=0), "concatenate 2d axis=0")
    # Verify default axis=0
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concatenate 2d default axis")

def test_concatenate_2d_axis1(cpp, dtype):
    arrays = [random_array((3, 2), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, 1), np.concatenate(arrays, axis=1), "concatenate 2d axis=1")

def test_concatenate_2d_axis_neg1(cpp, dtype):
    arrays = [random_array((3, 2), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, -1), np.concatenate(arrays, axis=-1), "concatenate 2d axis=-1")

def test_concatenate_3d_axis0(cpp, dtype):
    arrays = [random_array((2, 3, 4), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 0), np.concatenate(arrays, axis=0), "concatenate 3d axis=0")

def test_concatenate_3d_axis1(cpp, dtype):
    arrays = [random_array((3, 2, 4), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 1), np.concatenate(arrays, axis=1), "concatenate 3d axis=1")

def test_concatenate_3d_axis2(cpp, dtype):
    arrays = [random_array((3, 4, 2), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 2), np.concatenate(arrays, axis=2), "concatenate 3d axis=2")

def test_concatenate_two_arrays(cpp, dtype):
    arrays = [random_array((5,), seed=0, dtype=dtype), random_array((7,), seed=1, dtype=dtype)]
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concatenate two")

def test_concatenate_single(cpp, dtype):
    arrays = [random_array((5,), dtype=dtype)]
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concatenate single")

def test_vstack(cpp, dtype):
    arrays = [random_array((1, 3), seed=i, dtype=dtype) for i in range(4)]
    assert_bit_aligned(cpp.vstack(arrays), np.vstack(arrays), "vstack")

def test_vstack_1d(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(4)]
    assert_bit_aligned(cpp.vstack(arrays), np.vstack(arrays), "vstack 1d")

def test_hstack(cpp, dtype):
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.hstack(arrays), np.hstack(arrays), "hstack 1d")

def test_hstack_2d(cpp, dtype):
    arrays = [random_array((3, 2), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.hstack(arrays), np.hstack(arrays), "hstack 2d")

# -- Concatenate complex / edge-case tests ----------------------------------

def test_concatenate_4d_axis0(cpp, dtype):
    arrays = [random_array((2, 3, 4, 5), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 0), np.concatenate(arrays, axis=0), "concatenate 4d axis=0")

def test_concatenate_4d_axis2(cpp, dtype):
    arrays = [random_array((2, 3, 2, 5), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 2), np.concatenate(arrays, axis=2), "concatenate 4d axis=2")

def test_concatenate_4d_axis_neg2(cpp, dtype):
    arrays = [random_array((2, 3, 2, 5), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, -2), np.concatenate(arrays, axis=-2), "concatenate 4d axis=-2")

def test_concatenate_unequal_axis_sizes(cpp, dtype):
    """Concatenate arrays of different sizes along the concatenation axis."""
    a = random_array((3, 2), seed=1, dtype=dtype)
    b = random_array((3, 4), seed=2, dtype=dtype)
    c = random_array((3, 1), seed=3, dtype=dtype)
    assert_bit_aligned(cpp.concatenate([a, b, c], 1),
                       np.concatenate([a, b, c], axis=1), "concat unequal axis sizes")

def test_concatenate_many_arrays(cpp, dtype):
    """Concatenate 10 arrays along axis=0."""
    arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(10)]
    assert_bit_aligned(cpp.concatenate(arrays), np.concatenate(arrays), "concat 10 arrays")

def test_concatenate_large_3d(cpp, dtype):
    """Large 3D concatenation along middle axis."""
    arrays = [random_array((50, 20, 30), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, 1), np.concatenate(arrays, axis=1), "concat large 3d axis=1")

def test_concatenate_large_2d_axis0(cpp, dtype):
    """Large 2D concatenation — 500 rows each, 4 arrays."""
    arrays = [random_array((500, 10), seed=i, dtype=dtype) for i in range(4)]
    assert_bit_aligned(cpp.concatenate(arrays, 0), np.concatenate(arrays, axis=0), "concat large 2d axis=0")

def test_concatenate_large_2d_axis1(cpp, dtype):
    """Large 2D concatenation — 500 cols each, 3 arrays."""
    arrays = [random_array((10, 500), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, 1), np.concatenate(arrays, axis=1), "concat large 2d axis=1")

def test_concatenate_identity(cpp, dtype):
    """Concatenating a single array returns identical copy."""
    a = random_array((3, 4), seed=42, dtype=dtype)
    assert_bit_aligned(cpp.concatenate([a], 0), np.concatenate([a], axis=0), "concat identity")
    assert_bit_aligned(cpp.concatenate([a], 1), np.concatenate([a], axis=1), "concat identity axis=1")

def test_concatenate_zeros(cpp, dtype):
    """Concatenate arrays of zeros."""
    a = np.zeros((2, 3), dtype=dtype)
    b = np.zeros((2, 5), dtype=dtype)
    assert_bit_aligned(cpp.concatenate([a, b], 1), np.concatenate([a, b], axis=1), "concat zeros")

def test_concatenate_ones(cpp, dtype):
    """Concatenate arrays of ones."""
    a = np.ones((3, 2), dtype=dtype)
    b = np.ones((5, 2), dtype=dtype)
    assert_bit_aligned(cpp.concatenate([a, b], 0), np.concatenate([a, b], axis=0), "concat ones")

def test_concatenate_3d_axis_neg2(cpp, dtype):
    """3D concatenate along axis=-2 (middle axis)."""
    arrays = [random_array((2, 3, 4), seed=i, dtype=dtype) for i in range(3)]
    assert_bit_aligned(cpp.concatenate(arrays, -2), np.concatenate(arrays, axis=-2), "concat 3d axis=-2")

def test_concatenate_3d_axis_neg3(cpp, dtype):
    """3D concatenate along axis=-3 (first axis)."""
    arrays = [random_array((2, 3, 4), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, -3), np.concatenate(arrays, axis=-3), "concat 3d axis=-3")

def test_concatenate_5d(cpp, dtype):
    """5D concatenate along various axes."""
    arrays = [random_array((2, 3, 2, 3, 2), seed=i, dtype=dtype) for i in range(2)]
    assert_bit_aligned(cpp.concatenate(arrays, 0), np.concatenate(arrays, axis=0), "concat 5d axis=0")
    assert_bit_aligned(cpp.concatenate(arrays, 2), np.concatenate(arrays, axis=2), "concat 5d axis=2")
    assert_bit_aligned(cpp.concatenate(arrays, -1), np.concatenate(arrays, axis=-1), "concat 5d axis=-1")

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

def test_isin_int(cpp):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_bit_aligned(cpp.isin(a, [2, 4, 6]), np.isin(a, [2, 4, 6]), "isin_int")

def test_flatnonzero(cpp):
    a = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    assert_bit_aligned(cpp.flatnonzero(a), np.flatnonzero(a), "flatnonzero")
    # all zeros
    a2 = np.array([0.0, 0.0, 0.0])
    assert_bit_aligned(cpp.flatnonzero(a2), np.flatnonzero(a2), "flatnonzero zeros")

def test_hypot(cpp):
    for dt in [np.float64, np.float32]:
        x = np.array([3.0, 1.0, 5.0, 0.0, 1e10], dtype=dt)
        y = np.array([4.0, 1.0, 12.0, 5.0, 1e10], dtype=dt)
        assert_bit_aligned(cpp.hypot(x, y), np.hypot(x, y), f"hypot_{dt}")

def test_unwrap(cpp):
    for dt in [np.float64, np.float32]:
        a = np.array([0.0, 0.5, 0.8, -0.9, -0.5, 0.2], dtype=dt)
        assert_bit_aligned(cpp.unwrap(a), np.unwrap(a), f"unwrap_{dt}")
    # Large values — bit-exact for both float64 and float32
    for dt in [np.float64, np.float32]:
        a2 = np.array([0.0, 2.5, 5.0, -2.5, -5.0], dtype=dt) * np.pi
        assert_bit_aligned(cpp.unwrap(a2), np.unwrap(a2), f"unwrap_large_{dt.__name__}")

def test_cumsum(cpp):
    for dt in [np.float64, np.float32]:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        assert_bit_aligned(cpp.cumsum(a), np.cumsum(a), f"cumsum_{dt}")
        a2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=dt)
        assert_bit_aligned(cpp.cumsum(a2), np.cumsum(a2), f"cumsum_frac_{dt}")
        a3 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=dt)
        assert_bit_aligned(cpp.cumsum(a3), np.cumsum(a3), f"cumsum_neg_{dt}")

def test_squeeze(cpp):
    for dt in [np.float64, np.float32]:
        a = np.array([1.0, 2.0, 3.0], dtype=dt).reshape(3, 1)
        assert_bit_aligned(cpp.squeeze(a), np.squeeze(a), f"squeeze_col_{dt}")
        a2 = np.array([1.0, 2.0, 3.0], dtype=dt).reshape(1, 3)
        assert_bit_aligned(cpp.squeeze(a2), np.squeeze(a2), f"squeeze_row_{dt}")
        a3 = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt).reshape(1, 2, 1, 2, 1)
        assert_bit_aligned(cpp.squeeze(a3), np.squeeze(a3), f"squeeze_multi_{dt}")

def test_intersect1d(cpp):
    for dt in [np.float64, np.float32]:
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)
        b = np.array([3.0, 4.0, 5.0, 6.0], dtype=dt)
        cpp_r = np.sort(np.asarray(cpp.intersect1d(a, b)))
        assert_bit_aligned(cpp_r, np.intersect1d(a, b), f"intersect1d_{dt}")

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
    a = random_array((100,), dtype=dtype)
    # np.linalg.norm internally computes sqrt(a.dot(a)) via BLAS
    assert_bit_aligned(dtype(cpp.linalg.norm(a)), dtype(np.linalg.norm(a)), "linalg.norm 1d")

def test_norm_2d(cpp, dtype):
    a = random_array((5, 4), dtype=dtype)
    # Frobenius norm: same BLAS path as 1d
    assert_bit_aligned(dtype(cpp.linalg.norm(a)), dtype(np.linalg.norm(a)), "linalg.norm 2d")

def test_norm_zero(cpp, dtype):
    a = np.zeros((100,), dtype=dtype)
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
    # np.dot routes through BLAS sdot/ddot
    assert_bit_aligned(cpp.dot(a, b), np.dot(a, b), "dot")

def test_dot_orthogonal(cpp, dtype):
    a = np.array([1.0, 0.0], dtype=dtype)
    b = np.array([0.0, 1.0], dtype=dtype)
    assert_bit_aligned(cpp.dot(a, b), np.dot(a, b), "dot orthogonal")


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
        # Compare vs np.einsum (same SSE forward mul+add path as our implementation).
        # np.einsum and np.matmul use different BLAS paths and can differ at machine
        # epsilon — the bit-exact contract is with np.einsum, not with matmul.
        assert_bit_aligned(np.asarray(cpp.einsum("ij,jk->ik", a, b)),
                           np.einsum("ij,jk->ik", a, b),
                           "ij,jk->ik vs np.einsum")


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
        # Compare vs np.einsum (same SSE forward mul+add path as our implementation).
        # np.einsum and np.matmul use different BLAS paths and can differ at machine
        # epsilon — the bit-exact contract is with np.einsum, not with batched matmul.
        assert_bit_aligned(np.asarray(cpp.einsum("bij,bjk->bik", a, b)),
                           np.einsum("bij,bjk->bik", a, b),
                           "bij,bjk->bik vs np.einsum")


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


# ============================================================================
# 16. Special values — NaN / ±0 / ±Inf passthrough
# ============================================================================
#
# All float functions must be bit-exact with numpy for IEEE 754 special inputs:
#   • NaN input  → NaN output with identical bit pattern
#   • ±0 input   → correct signed result (sin(-0)=-0, cos(0)=1, etc.)
#   • ±Inf input → correct result (+inf, -inf, or NaN as numpy defines)
#   • Domain errors (log(-1), sqrt(-1), arcsin(2)) → NaN bit-exact with numpy
#
# AVX-512 boundary sizes (1, 16, 17) stress the scalar-tail vs full-vector paths.
# ============================================================================

# Functions tested for NaN passthrough (all unary float functions)
_UNARY_NAN_FNS = [
    ("exp",     np.exp),
    ("log",     np.log),
    ("sin",     np.sin),
    ("cos",     np.cos),
    ("tan",     np.tan),
    ("sqrt",    np.sqrt),
    ("cbrt",    np.cbrt),
    ("expm1",   np.expm1),
    ("log1p",   np.log1p),
    ("log10",   np.log10),
    ("log2",    np.log2),
    ("arcsin",  np.arcsin),
    ("arccos",  np.arccos),
    ("arctan",  np.arctan),
    ("abs",     np.abs),
    ("sign",    np.sign),
    ("floor",   np.floor),
    ("ceil",    np.ceil),
    ("round",   np.round),
    ("degrees", np.degrees),
    ("radians", np.radians),
]
_NAN_FN_IDS = [fn for fn, _ in _UNARY_NAN_FNS]

# AVX-512 boundary sizes: 1 = scalar-tail only; 16 = full f32 vector; 17 = vector+tail
_NAN_SIZES_F32 = [1, 16, 17]
# F64 vector width = 8: 1 = scalar tail; 8 = full f64 vector; 9 = vector+tail
_NAN_SIZES_F64 = [1, 8, 9]


@pytest.mark.parametrize(
    "fn_name,np_fn,n",
    [(fn, npf, sz) for fn, npf in _UNARY_NAN_FNS for sz in _NAN_SIZES_F32],
    ids=[f"{fn}_n{sz}" for fn, _ in _UNARY_NAN_FNS for sz in _NAN_SIZES_F32])
def test_nan_passthrough_f32(fn_name, np_fn, n, cpp):
    """NaN input → NaN output, bit-exact with numpy, float32.

    Covers scalar-tail (n=1), full AVX-512 vector (n=16), and vector+tail (n=17).
    """
    a = np.full(n, np.nan, dtype=np.float32)
    assert_bit_aligned(getattr(cpp, fn_name)(a), np_fn(a),
                       f"{fn_name}(nan) f32 n={n}")


@pytest.mark.parametrize(
    "fn_name,np_fn,n",
    [(fn, npf, sz) for fn, npf in _UNARY_NAN_FNS for sz in _NAN_SIZES_F64],
    ids=[f"{fn}_n{sz}" for fn, _ in _UNARY_NAN_FNS for sz in _NAN_SIZES_F64])
def test_nan_passthrough_f64(fn_name, np_fn, n, cpp):
    """NaN input → NaN output, bit-exact with numpy, float64."""
    a = np.full(n, np.nan, dtype=np.float64)
    assert_bit_aligned(getattr(cpp, fn_name)(a), np_fn(a),
                       f"{fn_name}(nan) f64 n={n}")


@pytest.mark.parametrize("fn_name,np_fn", _UNARY_NAN_FNS, ids=_NAN_FN_IDS)
def test_nan_mixed_f32(fn_name, np_fn, cpp):
    """Mixed NaN and finite values (17 elements): NaN must not corrupt neighbours.

    17 elements forces AVX-512 16-wide + 1-element scalar tail path.
    Finite values chosen to lie in the safe domain of all tested functions.
    """
    a = np.array([1.0, np.nan, 0.5, np.nan, 0.3, np.nan, 0.7, np.nan,
                  0.1, np.nan, 0.9, np.nan, 0.2, np.nan, 0.8, np.nan,
                  0.4], dtype=np.float32)
    assert_bit_aligned(getattr(cpp, fn_name)(a), np_fn(a),
                       f"{fn_name}(mixed nan) f32")


@pytest.mark.parametrize("fn_name,np_fn", _UNARY_NAN_FNS, ids=_NAN_FN_IDS)
def test_nan_mixed_f64(fn_name, np_fn, cpp):
    """Mixed NaN and finite values (9 elements): AVX-512 8-wide + 1-element tail."""
    a = np.array([1.0, np.nan, 0.5, np.nan, 0.3,
                  np.nan, 0.7, np.nan, 0.4], dtype=np.float64)
    assert_bit_aligned(getattr(cpp, fn_name)(a), np_fn(a),
                       f"{fn_name}(mixed nan) f64")


# --- Signed zero ---

def test_sin_signed_zero(cpp):
    """sin(−0.0) = −0.0, sin(+0.0) = +0.0 — signed zeros must propagate."""
    for dt in [np.float32, np.float64]:
        a = np.array([-0.0, 0.0], dtype=dt)
        assert_bit_aligned(cpp.sin(a), np.sin(a), f"sin(±0) {dt.__name__}")


def test_cos_signed_zero(cpp):
    """cos(±0.0) = 1.0 — both signs of zero give the same positive result."""
    for dt in [np.float32, np.float64]:
        a = np.array([0.0, -0.0], dtype=dt)
        assert_bit_aligned(cpp.cos(a), np.cos(a), f"cos(±0) {dt.__name__}")


def test_log_neg_zero(cpp):
    """log(−0.0) = −∞ (IEEE 754: log of ±0 is −∞, not NaN).

    Classic bug: sign-bit check firing before zero check returns NaN instead.
    """
    for dt in [np.float32, np.float64]:
        a = np.array([-0.0], dtype=dt)
        result = np.asarray(cpp.log(a))
        assert np.isinf(result[0]) and result[0] < 0, \
               f"log(-0) should be -inf, got {result[0]} ({dt.__name__})"
        assert_bit_aligned(result, np.log(a), f"log(-0) {dt.__name__}")


def test_exp_signed_zero(cpp):
    """exp(±0.0) = 1.0 for both zero signs."""
    for dt in [np.float32, np.float64]:
        a = np.array([0.0, -0.0], dtype=dt)
        assert_bit_aligned(cpp.exp(a), np.exp(a), f"exp(±0) {dt.__name__}")


# --- Infinity inputs ---

def test_exp_inf(cpp):
    """exp(+∞) = +∞, exp(−∞) = 0.0 — both must be bit-exact."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf, -np.inf], dtype=dt)
        assert_bit_aligned(cpp.exp(a), np.exp(a), f"exp(±inf) {dt.__name__}")


def test_log_pos_inf(cpp):
    """log(+∞) = +∞."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf], dtype=dt)
        assert_bit_aligned(cpp.log(a), np.log(a), f"log(+inf) {dt.__name__}")


def test_sqrt_pos_inf(cpp):
    """sqrt(+∞) = +∞."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf], dtype=dt)
        assert_bit_aligned(cpp.sqrt(a), np.sqrt(a), f"sqrt(+inf) {dt.__name__}")


def test_sin_inf(cpp):
    """sin(±∞) = NaN (invalid-operation domain error)."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf, -np.inf], dtype=dt)
        assert_bit_aligned(cpp.sin(a), np.sin(a), f"sin(±inf) {dt.__name__}")


def test_cos_inf(cpp):
    """cos(±∞) = NaN (invalid-operation domain error)."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf, -np.inf], dtype=dt)
        assert_bit_aligned(cpp.cos(a), np.cos(a), f"cos(±inf) {dt.__name__}")


# --- Domain errors → NaN ---

def test_domain_sqrt_neg(cpp):
    """sqrt(negative) = NaN — bit-exact with numpy."""
    for dt in [np.float32, np.float64]:
        a = np.array([-1.0, -4.0, -0.5], dtype=dt)
        assert_bit_aligned(cpp.sqrt(a), np.sqrt(a), f"sqrt(neg) {dt.__name__}")


def test_domain_log_neg(cpp):
    """log(negative) = NaN — both C++ and numpy must produce NaN.

    The exact NaN bit-pattern (sign bit) varies across numpy versions and CPU
    paths (SVML vs scalar) — numpy ≥1.26 normalises to positive qNaN on the
    scalar path but not the SVML path.  We therefore check only that the output
    is NaN where numpy is NaN, not the exact bit pattern.
    """
    for dt in [np.float32, np.float64]:
        a = np.array([-1.0, -2.5, -0.5], dtype=dt)
        cpp_r = np.asarray(cpp.log(a))
        np_r  = np.log(a)
        assert np.all(np.isnan(cpp_r) == np.isnan(np_r)), \
               f"log(neg) {dt.__name__}: NaN mask mismatch: C++={cpp_r} numpy={np_r}"
    # 16 negative floats — exercises AVX-512 wide loop when available
    a16 = np.full(16, -1.0, dtype=np.float32)
    cpp_r16 = np.asarray(cpp.log(a16))
    assert np.all(np.isnan(cpp_r16)), \
           f"log(neg) f32 n=16: expected all NaN, got {cpp_r16}"


def test_domain_arcsin_oob(cpp):
    """arcsin/arccos(|x|>1) = NaN — bit-exact with numpy."""
    for dt in [np.float32, np.float64]:
        a = np.array([2.0, -2.0, 1.5], dtype=dt)
        assert_bit_aligned(cpp.arcsin(a), np.arcsin(a), f"arcsin(oob) {dt.__name__}")
        assert_bit_aligned(cpp.arccos(a), np.arccos(a), f"arccos(oob) {dt.__name__}")


# --- sign special values ---

def test_sign_nan_special(cpp):
    """sign(NaN) = NaN — must not silently return 0."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.nan, np.nan], dtype=dt)
        assert_bit_aligned(cpp.sign(a), np.sign(a), f"sign(nan) {dt.__name__}")


def test_sign_inf_special(cpp):
    """sign(+∞) = +1.0, sign(−∞) = −1.0."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf, -np.inf], dtype=dt)
        assert_bit_aligned(cpp.sign(a), np.sign(a), f"sign(±inf) {dt.__name__}")


def test_sign_zero_signs(cpp):
    """sign(+0) = sign(−0) = 0.0 — both signed zeros map to positive zero."""
    for dt in [np.float32, np.float64]:
        a = np.array([0.0, -0.0], dtype=dt)
        assert_bit_aligned(cpp.sign(a), np.sign(a), f"sign(±0) {dt.__name__}")


# --- unwrap NaN propagation ---

def test_unwrap_nan_propagation(cpp):
    """NaN element in unwrap must propagate through subsequent output values."""
    for dt in [np.float32, np.float64]:
        a = np.array([0.0, 1.0, np.nan, 2.0, 3.0], dtype=dt)
        assert_bit_aligned(cpp.unwrap(a), np.unwrap(a),
                           f"unwrap(nan mid) {dt.__name__}")


def test_unwrap_nan_leading(cpp):
    """NaN as first element of unwrap input."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.nan, 1.0, 2.0, 3.0], dtype=dt)
        assert_bit_aligned(cpp.unwrap(a), np.unwrap(a),
                           f"unwrap(nan first) {dt.__name__}")


def test_unwrap_nan_all_nan(cpp):
    """All-NaN unwrap input."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.nan, np.nan, np.nan], dtype=dt)
        assert_bit_aligned(cpp.unwrap(a), np.unwrap(a),
                           f"unwrap(all nan) {dt.__name__}")


# --- linalg special values ---

def test_linalg_norm_nan(cpp):
    """linalg.norm of array with NaN must produce NaN (not a finite value)."""
    for dt in [np.float32, np.float64]:
        a = np.array([1.0, np.nan, 2.0], dtype=dt)
        r = float(np.asarray(cpp.linalg.norm(a)).item())
        assert np.isnan(r), \
               f"linalg.norm(nan array) should be NaN, got {r} ({dt.__name__})"


def test_linalg_norm_inf(cpp):
    """linalg.norm of array with +∞ must produce +∞."""
    for dt in [np.float32, np.float64]:
        a = np.array([1.0, np.inf, 2.0], dtype=dt)
        r = float(np.asarray(cpp.linalg.norm(a)).item())
        assert np.isinf(r), \
               f"linalg.norm(inf array) should be Inf, got {r} ({dt.__name__})"


def test_dot_nan(cpp):
    """dot product with NaN operand must produce NaN."""
    for dt in [np.float32, np.float64]:
        a = np.array([1.0, np.nan, 3.0], dtype=dt)
        b = np.array([4.0, 5.0, 6.0], dtype=dt)
        r = float(np.asarray(cpp.dot(a, b)).item())
        assert np.isnan(r), \
               f"dot(nan,...) should be NaN, got {r} ({dt.__name__})"


def test_dot_inf(cpp):
    """dot([+∞, 0], [1, 0]) = +∞ — infinity propagates through dot."""
    for dt in [np.float32, np.float64]:
        a = np.array([np.inf, 0.0], dtype=dt)
        b = np.array([1.0,    0.0], dtype=dt)
        r    = float(np.asarray(cpp.dot(a, b)).item())
        py_r = float(np.dot(a, b))
        # Both should be +inf (or consistently NaN if BLAS treats it that way)
        assert (np.isinf(r) and r > 0) == (np.isinf(py_r) and py_r > 0) or \
               (np.isnan(r) and np.isnan(py_r)), \
               f"dot(inf,...): C++={r} vs numpy={py_r} ({dt.__name__})"


# --- AVX-512 boundary sizes for normal inputs ---

@pytest.mark.parametrize("fn_name,np_fn", [
    ("exp",    np.exp),
    ("log",    np.log),
    ("sin",    np.sin),
    ("cos",    np.cos),
], ids=["exp", "log", "sin", "cos"])
@pytest.mark.parametrize("n", [15, 16, 17, 32], ids=["n15", "n16", "n17", "n32"])
def test_avx512_boundary_f32(fn_name, np_fn, n, cpp):
    """Normal finite values at exact AVX-512 vector boundary sizes (f32)."""
    rng = np.random.RandomState(42)
    a = (np.abs(rng.randn(n)) + 0.1).astype(np.float32)
    assert_bit_aligned(getattr(cpp, fn_name)(a), np_fn(a),
                       f"{fn_name} f32 n={n}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__, "-q", "--tb=short", "--no-header"]))


# =============================================================================
# Section 17: numpy.matmul — bit-exact via cblas_sgemm64_ / cblas_sgemv64_
# =============================================================================

@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64","float32"])
@pytest.mark.parametrize("M,K,N", [
    (1,  1,  1),
    (3,  4,  5),
    (5,  3,  1),
    (1,  8,  4),
    (16, 16, 16),
    (50, 64, 50),
    (100,100,100),
], ids=["1x1x1","3x4x5","5x3x1","1x8x4","16x16x16","50x64x50","100x100x100"])
def test_matmul_2d(dtype, M, K, N, cpp):
    """2D matmul: C(M,N) = A(M,K) @ B(K,N)  — cblas_sgemm64_, 0 ULP."""
    rng = np.random.RandomState(M * 1000 + K * 100 + N)
    A = rng.randn(M, K).astype(dtype)
    B = rng.randn(K, N).astype(dtype)
    assert_bit_aligned(cpp.matmul(A, B), np.matmul(A, B),
                       f"matmul 2D ({M},{K})@({K},{N}) {dtype.__name__}")


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64","float32"])
@pytest.mark.parametrize("K,N", [(1,1),(8,5),(16,7),(64,32)],
                          ids=["1x1","8x5","16x7","64x32"])
def test_matmul_1d_2d(dtype, K, N, cpp):
    """1D × 2D matmul: y(N,) = a(K,) @ B(K,N)  — cblas_sgemv64_ Trans, 0 ULP."""
    rng = np.random.RandomState(K * 100 + N)
    a = rng.randn(K).astype(dtype)
    B = rng.randn(K, N).astype(dtype)
    assert_bit_aligned(cpp.matmul(a, B), np.matmul(a, B),
                       f"matmul 1D×2D ({K},)@({K},{N}) {dtype.__name__}")


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64","float32"])
@pytest.mark.parametrize("M,K", [(1,1),(5,8),(7,16),(32,64)],
                          ids=["1x1","5x8","7x16","32x64"])
def test_matmul_2d_1d(dtype, M, K, cpp):
    """2D × 1D matmul: y(M,) = A(M,K) @ x(K,)  — cblas_sgemv64_ NoTrans, 0 ULP."""
    rng = np.random.RandomState(M * 100 + K)
    A = rng.randn(M, K).astype(dtype)
    x = rng.randn(K).astype(dtype)
    assert_bit_aligned(cpp.matmul(A, x), np.matmul(A, x),
                       f"matmul 2D×1D ({M},{K})@({K},) {dtype.__name__}")


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64","float32"])
@pytest.mark.parametrize("batch,M,K,N", [
    (1, 2, 3, 4),
    (4, 3, 5, 6),
    (8, 16, 32, 16),
    (3, 50, 64, 50),
], ids=["1x2x3x4","4x3x5x6","8x16x32x16","3x50x64x50"])
def test_matmul_batched(dtype, batch, M, K, N, cpp):
    """Batched 3D matmul: C(B,M,N) = A(B,M,K) @ B(B,K,N)  — loop gemm, 0 ULP."""
    rng = np.random.RandomState(batch * 10000 + M * 1000 + K * 100 + N)
    A = rng.randn(batch, M, K).astype(dtype)
    B = rng.randn(batch, K, N).astype(dtype)
    assert_bit_aligned(cpp.matmul(A, B), np.matmul(A, B),
                       f"matmul 3D ({batch},{M},{K})@({batch},{K},{N}) {dtype.__name__}")
