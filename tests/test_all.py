"""
全量位级对齐测试 —— numpycpp C++ vs Python numpy 全部 API。

运行:  pytest tests/test_all.py -v

架构: 5 函数驱动全量测试。
  F3  compare()     — 位级比对
  F4  call_cpp_py() — 按名称反射调用 CPP / PY 同名函数
  F5  api_catalog() — 导出全部 API 元数据（按源码5大类组织）
  F1  极端数据生成   — 内嵌于各类目工厂函数
  F2  reshape适配   — 内嵌于各类目工厂函数
"""

import os
import importlib
import numpy as np
import pytest
from collections import namedtuple


# ═══════════════════════════════════════════════════════════════════════════════
# F3: compare — 位级比对
# ═══════════════════════════════════════════════════════════════════════════════

_UINT_VIEW = {4: np.uint32, 8: np.uint64}
_EL_VIEW  = {2: np.uint16, 4: np.uint32, 8: np.uint64}
_EL_FMT   = {2: "04x", 4: "08x", 8: "016x"}


def compare(cpp_result, py_result, strategy="bit_exact", label=""):
    """统一比对入口。strategy: bit_exact | scalar_eq | shape_only | nan_mask | none"""
    if strategy == "none":
        return
    if strategy == "scalar_eq":
        c = float(np.asarray(cpp_result).item())
        p = float(np.asarray(py_result).item())
        if c != p:
            raise AssertionError(f"[{label}] scalar mismatch: C++={c} vs numpy={p}")
        return
    if strategy == "shape_only":
        if np.asarray(cpp_result).shape != np.asarray(py_result).shape:
            raise AssertionError(f"[{label}] shape mismatch: "
                                 f"C++ {np.asarray(cpp_result).shape} vs numpy {np.asarray(py_result).shape}")
        return
    if strategy == "nan_mask":
        cpp = np.asarray(cpp_result)
        py  = np.asarray(py_result)
        if not np.array_equal(np.isnan(cpp), np.isnan(py)):
            raise AssertionError(f"[{label}] NaN mask mismatch")
        return
    # bit_exact
    _compare_bit_exact(cpp_result, py_result, label)


def _compare_bit_exact(cpp_result, py_result, label=""):
    """位级精确比对，含 hex dump 诊断。"""
    cpp = np.asarray(cpp_result)
    py  = np.asarray(py_result)

    if cpp.shape != py.shape:
        raise AssertionError(
            f"[{label}] shape mismatch: C++ {cpp.shape} vs numpy {py.shape}")

    if cpp.dtype.kind == 'f' and cpp.dtype == py.dtype:
        uint_t = _UINT_VIEW.get(cpp.itemsize)
        if uint_t is not None:
            cpp_u = np.ascontiguousarray(cpp).ravel().view(uint_t)
            py_u  = np.ascontiguousarray(py).ravel().view(uint_t)
            if np.array_equal(cpp_u, py_u):
                return
            diff_mask = (cpp_u != py_u).reshape(cpp.shape)
        else:
            if np.array_equal(cpp, py):
                return
            diff_mask = cpp != py
    else:
        if np.array_equal(cpp, py):
            return
        diff_mask = np.asarray(cpp != py)
        if diff_mask.shape != cpp.shape:
            try:
                diff_mask = diff_mask.reshape(cpp.shape)
            except ValueError:
                diff_mask = np.ones(cpp.shape, dtype=bool)

    n_diff = int(np.sum(diff_mask))
    diff_idx = np.flatnonzero(diff_mask.ravel())
    lines = [f"[{label}] BIT-LEVEL MISMATCH: {n_diff}/{cpp.size}"]
    for idx in diff_idx[:5]:
        cv, pv = cpp.flat[idx], py.flat[idx]
        if cpp.dtype == bool or np.issubdtype(cpp.dtype, np.integer):
            lines.append(f"  [{idx}] C++={cv}  vs  numpy={pv}")
        else:
            cvt = _EL_VIEW.get(cpp.itemsize)
            pvt = _EL_VIEW.get(py.itemsize)
            cf  = _EL_FMT.get(cpp.itemsize, "016x")
            pf  = _EL_FMT.get(py.itemsize, "016x")
            ch = np.ascontiguousarray(cpp).view(cvt).flat[idx] if cvt else 0
            ph = np.ascontiguousarray(py).view(pvt).flat[idx] if pvt else 0
            lines.append(f"  [{idx}] C++={cv:.16e} (0x{ch:{cf}})  vs  "
                         f"numpy={pv:.16e} (0x{ph:{pf}})")
    if len(diff_idx) > 5:
        lines.append(f"  ... 还有 {len(diff_idx) - 5} 个差异元素")
    raise AssertionError("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# F4: call_cpp_py — 按名称反射调用 C++ 与 Python 同名函数
# ═══════════════════════════════════════════════════════════════════════════════

_CPP_TO_NP_MAP = {}

def _register_np_alias(cpp_path, np_path=None):
    def deco(fn):
        _CPP_TO_NP_MAP[cpp_path] = (np_path, fn)
        return fn
    return deco

# 无 numpy 等效的 API
@_register_np_alias("putmask")
def _rewrite_putmask(_r, args, kwargs): return args, kwargs
_register_np_alias("safe_divide", None)(None)
_register_np_alias("truncate_to_float32", None)(None)
_register_np_alias("array_get", None)(None)
_register_np_alias("get_array", None)(None)
_register_np_alias("set_array", None)(None)
_register_np_alias("take_cols", None)(None)
_register_np_alias("slice_assign", None)(None)
_register_np_alias("to_vector", None)(None)


def call_cpp_py(api_name, cpp, *args, **kwargs):
    """按名称字符串反射调用 C++ 和 Python numpy 同名函数。"""
    parts = api_name.split(".")
    # 解析 C++ 函数
    cpp_fn = cpp
    for part in parts:
        try:
            cpp_fn = getattr(cpp_fn, part)
        except AttributeError:
            raise AttributeError(
                f"C++ 模块在路径 '{api_name}' 中不存在属性 '{part}'。"
                f"可用: {dir(cpp_fn)}")
    # 解析 numpy 函数
    np_fn = np
    np_parts = list(parts)
    map_entry = _CPP_TO_NP_MAP.get(api_name)
    if map_entry is not None:
        np_path, _rewrite = map_entry
        if np_path is None:
            np_fn = None
        else:
            np_parts = np_path.split(".")
    if np_fn is not None:
        for part in np_parts:
            try:
                np_fn = getattr(np_fn, part)
            except AttributeError:
                np_fn = None
                break
    cpp_result = cpp_fn(*args, **kwargs)
    py_result = np_fn(*args, **kwargs) if np_fn is not None else None
    return cpp_result, py_result


# ═══════════════════════════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════════════════════════

TestCase = namedtuple('TestCase', [
    'api_name',     # API 名称
    'args',         # 位置参数 tuple
    'kwargs',       # 关键字参数 dict
    'dtype_label',  # dtype 标签 (用于 pytest ID)
    'category',     # 极端数据类别
    'cmp_strategy', # 比较策略
    'check_py',     # 是否比对 numpy
    'setup_fn',     # 前置/后置处理函数 or None
    'group',        # 大类: elementwise/init/linalg/manipulation/reduce
], defaults=("",))


def _seed(shape, *extras):
    """返回非负确定性种子。h = hash(shape) 或 hash((shape, *extras))。"""
    h = hash((shape,) + extras) if extras else hash(shape)
    return h & 0x7FFFFFFF


def _random_array(shape, dtype=np.float64, seed=42):
    """确定性的随机数组。"""
    rng = np.random.RandomState((seed + _seed(shape)) % (2**31))
    if np.issubdtype(dtype, np.floating):
        return rng.randn(*shape).astype(dtype)
    elif dtype == bool:
        return rng.rand(*shape) > 0.5
    else:
        return rng.randint(0, 100, size=shape).astype(dtype)


def _make_extreme(shape, dtype, category, seed=42):
    """生成指定类别的极端数据。"""
    rng = np.random.RandomState((seed + _seed(shape, category)) % (2**31))
    n = int(np.prod(shape)) if shape else 0

    if category == "random":
        return _random_array(shape, dtype, seed)
    if category == "zeros":
        return np.zeros(shape, dtype=dtype)
    if category == "ones":
        return np.ones(shape, dtype=dtype)
    if category == "nan":
        return np.full(shape, np.nan, dtype=dtype)
    if category == "mixed_nan":
        a = _random_array(shape, dtype, seed)
        a.flat[::3] = np.nan
        return a
    if category == "inf":
        return np.array([np.inf, -np.inf] * ((n + 1) // 2), dtype=dtype)[:n].reshape(shape)
    if category == "mixed_inf":
        a = _random_array(shape, dtype, seed)
        a.flat[::4] = np.inf
        a.flat[1::4] = -np.inf
        return a
    if category == "signed_zero":
        return np.array([0.0, -0.0] * ((n + 1) // 2), dtype=dtype)[:n].reshape(shape)
    if category == "domain_edge":
        return (np.abs(_random_array(shape, dtype, seed)) * 0.1 + 0.01).astype(dtype)
    if category == "large":
        return (_random_array(shape, dtype, seed) * 1e150).astype(dtype)
    if category == "tiny":
        return (_random_array(shape, dtype, seed) * 1e-150).astype(dtype)
    if category == "empty":
        empty_shape = list(shape)
        empty_shape[0] = 0
        return np.empty(empty_shape, dtype=dtype)
    return _random_array(shape, dtype, seed)


# 一元浮点 API 的输入预处理（确保值在定义域内）
_PREP_INPUT = {
    "sqrt":   lambda a: np.abs(a),
    "log":    lambda a: np.abs(a) + 0.1,
    "log10":  lambda a: np.abs(a) + 0.1,
    "log2":   lambda a: np.abs(a) + 0.1,
    "log1p":  lambda a: np.abs(a) + 0.1,
    "arcsin": lambda a: np.clip(a * 0.5, -1, 1),
    "arccos": lambda a: np.clip(a * 0.5, -1, 1),
    "tan":    lambda a: a * 0.5,
    "expm1":  lambda a: a * 2.0,
    "round":  lambda a: a * 100,
    "floor":  lambda a: a * 100,
    "ceil":   lambda a: a * 100,
}


# ═══════════════════════════════════════════════════════════════════════════════
# F5: api_catalog() — 5大类工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

# ── 第1类: elementwise (elementwise_py.h, 43 APIs) ───────────────────────────

_UNARY_FLOAT = [
    "sqrt", "abs", "exp", "log", "sin", "cos", "tan",
    "cbrt", "expm1", "log1p", "log10", "log2",
    "arcsin", "arccos", "arctan",
    "round", "floor", "ceil", "degrees", "radians", "sign", "reciprocal",
]
_UNARY_SHAPES = [(100,), (10000,)]
_UNARY_CATS  = ["random", "nan", "mixed_nan", "inf", "signed_zero", "zeros", "domain_edge"]


def _catalog_elementwise():
    """elementwise_py.h — 一元浮点 / 二元 / 比较 / 逻辑 / 特殊值 / 类型转换"""
    # ── 一元浮点 (21 APIs) ──
    for name in _UNARY_FLOAT:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            prep = _PREP_INPUT.get(name, lambda a: a)
            # 常规尺寸 + 极端数据
            for shape in _UNARY_SHAPES:
                for cat in _UNARY_CATS:
                    raw = _make_extreme(shape, dt, cat, seed=hash(name))
                    yield TestCase(name, (prep(raw),), {}, dn, f"{cat}_{shape[0]}",
                                   "bit_exact", True, None)
            # AVX-512 边界尺寸
            for sz in ([16, 17] if dt == np.float32 else [8, 9]):
                raw = _make_extreme((sz,), dt, "random", seed=hash(name))
                yield TestCase(name, (prep(raw),), {}, dn, f"avx512_n{sz}",
                               "bit_exact", True, None)

    # ── 二元标量: power, maximum(s), minimum(s), arctan2(s) ──
    for name, sv in [("power", 2.0), ("power", 3.0), ("power", 0.5),
                     ("maximum", 0.0), ("minimum", 0.0), ("arctan2", 1.0)]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            for shape in _UNARY_SHAPES:
                a = _make_extreme(shape, dt, "random", seed=hash(name))
                s = dt(sv)
                yield TestCase(name, (a, s), {}, dn, f"random_{shape[0]}",
                               "bit_exact", True, None)

    # ── 三元: clip ──
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        for lo, hi, tag in [(0.0, 1.0, "01"), (-1.0, 1.0, "m11"), (-10.0, 10.0, "m1010")]:
            for shape in _UNARY_SHAPES:
                a = _make_extreme(shape, dt, "random", seed=42)
                yield TestCase("clip", (a, dt(lo), dt(hi)), {}, dn,
                               f"{tag}_{shape[0]}", "bit_exact", True, None)

    # ── 二元数组: hypot, arctan2(a), maximum(a), minimum(a) ──
    for name in ["hypot", "arctan2", "maximum", "minimum"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            for shape in [(100,), (10000,)]:
                a = _make_extreme(shape, dt, "random", seed=1)
                b = _make_extreme(shape, dt, "random", seed=2)
                if name == "hypot":
                    a, b = np.abs(a), np.abs(b)
                yield TestCase(name, (a, b), {}, dn, f"random_{shape[0]}",
                               "bit_exact", True, None)

    # ── 比较: greater, less, equal, greater_equal, less_equal ──
    for name in ["greater", "less", "equal", "greater_equal", "less_equal"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            for shape in [(100,), (100000,)]:
                a = _make_extreme(shape, dt, "random", seed=42)
                yield TestCase(name, (a, dt(0.0)), {}, dn, f"vs0_{shape[0]}",
                               "bit_exact", True, None)

    # not_equal: scalar + array
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        a = _make_extreme((100,), dt, "random", seed=42)
        yield TestCase("not_equal", (a, dt(0.0)), {}, dn, "scalar",
                       "bit_exact", True, None)
        b = _make_extreme((100,), dt, "random", seed=99)
        yield TestCase("not_equal", (a, b), {}, dn, "array",
                       "bit_exact", True, None)

    # ── 逻辑: logical_and/or/xor/not ──
    for name in ["logical_and", "logical_or", "logical_xor"]:
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        yield TestCase(name, (a, b), {}, "bool", "basic", "bit_exact", True, None)
    yield TestCase("logical_not", (np.array([True, False, True]),), {},
                   "bool", "basic", "bit_exact", True, None)

    # ── 特殊值检测: isnan, isinf, isfinite ──
    for name in ["isnan", "isinf", "isfinite"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            a = np.array([0.0, np.nan, np.inf, -np.inf, 1.0], dtype=dt)
            yield TestCase(name, (a,), {}, dn, "special", "bit_exact", True, None)



# ── 第2类: init (init_py.h, 15 APIs) ─────────────────────────────────────────

def _catalog_init():
    """init_py.h — like创建 / shape创建 / 范围生成 / 矩阵创建"""
    # zeros_like, ones_like, empty_like, full_like
    for name in ["zeros_like", "ones_like", "empty_like", "full_like"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            for shape in [(3, 4), (5,)]:
                a = _make_extreme(shape, dt, "random", seed=42)
                if name == "full_like":
                    yield TestCase(name, (a, dt(3.14)), {}, dn, f"{shape}",
                                   "bit_exact", True, None)
                else:
                    yield TestCase(name, (a,), {}, dn, f"{shape}",
                                   "bit_exact" if name != "empty_like" else "shape_only",
                                   True, None)

    # zeros, ones, full, empty
    for shape in [(5,), (3, 4), (2, 3, 4)]:
        yield TestCase("zeros", (list(shape),), {}, "f64", str(shape),
                       "bit_exact", True, None)
        yield TestCase("ones",  (list(shape),), {}, "f64", str(shape),
                       "bit_exact", True, None)
        yield TestCase("full",  (list(shape), 3.14), {}, "f64", str(shape),
                       "bit_exact", True, None)
        yield TestCase("empty", (list(shape),), {}, "f64", str(shape),
                       "shape_only", True, None)

    # arange
    for args, tag in [((10,), "10"), ((1, 10), "1_10"), ((0, 10, 2), "0_10_2"),
                       ((0.5, 5.5, 0.5), "step05"), ((-3, 3, 1), "neg")]:
        yield TestCase("arange", args, {}, "f64", tag, "bit_exact", True, None)

    # linspace
    for s, e, n, ep, tag in [(0.0, 1.0, 50, True, "50T"), (0.0, 1.0, 50, False, "50F"),
                              (0.0, 0.0, 3, True, "degen"), (0.0, 1.0, 0, True, "empty")]:
        yield TestCase("linspace", (s, e, n, ep), {}, "f64", tag, "bit_exact", True, None)

    # logspace
    for s, e, n, ep, base, tag in [(0.0, 2.0, 5, True, 10.0, "T10"),
                                     (0.0, 1.0, 4, True, 2.0, "T2")]:
        yield TestCase("logspace", (s, e, n, ep, base), {}, "f64", tag, "bit_exact", True, None)

    # geomspace
    for s, e, n, ep, tag in [(1.0, 1000.0, 4, True, "T"),
                               (1.0, 256.0, 9, True, "9")]:
        yield TestCase("geomspace", (s, e, n, ep), {}, "f64", tag, "bit_exact", True, None)

    # eye — M=None (不传) = 方阵，严格对齐 numpy API
    yield TestCase("eye", (3,), {"k": 0}, "f64", "3", "bit_exact", True, None)
    yield TestCase("eye", (3,), {"k": 1}, "f64", "3_k1", "bit_exact", True, None)
    for M, tag in [(5, "3x5"), (3, "5x3")]:
        N = 3 if M == 5 else 5
        yield TestCase("eye", (N, M, 0), {}, "f64", tag, "bit_exact", True, None)

    # identity
    for n in [1, 3, 5]:
        yield TestCase("identity", (n,), {}, "f64", str(n), "bit_exact", True, None)

    # diag
    v = np.array([1.0, 2.0, 3.0])
    for k in [-2, -1, 0, 1, 2]:
        yield TestCase("diag", (v, k), {}, "f64", f"vec_k{k}", "bit_exact", True, None)
    m = np.arange(16.0).reshape(4, 4)
    yield TestCase("diag", (m, 0), {}, "f64", "mat_k0", "bit_exact", True, None)


# ── 第3类: linalg (linalg_py.h, 6 APIs) ──────────────────────────────────────

def _catalog_linalg():
    """linalg_py.h — norm / inv / dot / matmul / einsum"""
    # linalg.norm — scalar
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        for shape in [(100,), (5, 4)]:
            a = _make_extreme(shape, dt, "random", seed=42)
            yield TestCase("linalg.norm", (a,), {}, dn, str(shape),
                           "bit_exact", True, None)
        yield TestCase("linalg.norm", (np.zeros(100, dtype=dt),), {}, dn, "zero",
                       "bit_exact", True, None)
        for cat in ["nan", "inf"]:
            a = _make_extreme((10,), dt, cat, seed=42)
            yield TestCase("linalg.norm", (a,), {}, dn, cat,
                           "nan_mask" if cat == "nan" else "bit_exact", True, None)

    # linalg.norm — axis
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        a = _make_extreme((5, 4), dt, "random", seed=42)
        yield TestCase("linalg.norm", (a,), {"axis": 1}, dn, "axis1",
                       "bit_exact", True, None)

    # linalg.inv
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        yield TestCase("linalg.inv", (np.eye(4, dtype=dt),), {}, dn, "eye",
                       "bit_exact", True, None)
        yield TestCase("linalg.inv", (_make_extreme((4, 4), dt, "random", seed=42),),
                       {}, dn, "random4x4", "bit_exact", True, None)
    # 3x3, 8x8 float64 only
    for sz in [3, 8]:
        a = _make_extreme((sz, sz), np.float64, "random", seed=42)
        yield TestCase("linalg.inv", (a,), {}, "float64", f"random{sz}x{sz}",
                       "bit_exact", True, None)

    # dot
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        a = _make_extreme((100,), dt, "random", seed=1)
        b = _make_extreme((100,), dt, "random", seed=2)
        yield TestCase("dot", (a, b), {}, dn, "random", "bit_exact", True, None)

    # matmul
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        for M, K, N in [(3, 4, 5), (16, 16, 16), (100, 100, 100)]:
            a = _make_extreme((M, K), dt, "random", seed=M*100+K)
            b = _make_extreme((K, N), dt, "random", seed=K*100+N)
            yield TestCase("matmul", (a, b), {}, dn, f"2d_{M}x{K}x{N}",
                           "bit_exact", True, None)

    # einsum — 6种下标模式
    _EINSUM = [
        ("ij,ij->i",   (3, 2), (3, 2)),
        ("ij,jk->ik",  (3, 4), (4, 5)),
        ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
        ("aij,aij->ai",  (3, 5, 4), (3, 5, 4)),
        ("ij,ij",     (3, 2), (3, 2)),
        ("ij,jk",     (3, 4), (4, 5)),
    ]
    for sub, sa, sb in _EINSUM:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            a = _make_extreme(sa, dt, "random", seed=1)
            b = _make_extreme(sb, dt, "random", seed=2)
            tag = sub.replace(",", "_").replace("->", "__")
            yield TestCase("einsum", (sub, a, b), {}, dn, tag,
                           "bit_exact", True, None)


# ── 第4类: manipulation (manipulation_py.h, 26 APIs) ─────────────────────────

def _catalog_manipulation():
    """manipulation_py.h — 形状变换 / 重排 / 排序 / 切片 / 高级索引"""
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # diff
        for ax, tag in [(0, "ax0"), (1, "ax1"), (-1, "ax-1")]:
            a = _make_extreme((5, 4), dt, "random", seed=42)
            yield TestCase("diff", (a,), {"n": 1, "axis": ax}, dn, tag,
                           "bit_exact", True, None)

        # transpose, flatten
        a = _make_extreme((3, 5), dt, "random", seed=42)
        yield TestCase("transpose", (a,), {}, dn, "2d", "bit_exact", True, None)
        yield TestCase("flatten", (a,), {}, dn, "2d", "bit_exact", True, None)

        # squeeze
        for sh, tag in [((3, 1), "col"), ((1, 3), "row"), ((1, 2, 1, 2, 1), "multi")]:
            yield TestCase("squeeze", (_make_extreme(sh, dt, "random", seed=42),),
                           {}, dn, tag, "bit_exact", True, None)

        # stack, concatenate
        arrays = [_make_extreme((3,), dt, "random", seed=i) for i in range(4)]
        yield TestCase("stack", (arrays,), {}, dn, "4x3", "bit_exact", True, None)
        for ax, tag in [(0, "ax0"), (1, "ax1"), (-1, "ax-1")]:
            arrays2 = [_make_extreme((3, 2), dt, "random", seed=i) for i in range(3)]
            yield TestCase("concatenate", (arrays2,), {"axis": ax}, dn, tag,
                           "bit_exact", True, None)

        # vstack, hstack
        arrays1d = [_make_extreme((3,), dt, "random", seed=i) for i in range(4)]
        yield TestCase("vstack", (arrays1d,), {}, dn, "1d", "bit_exact", True, None)
        yield TestCase("hstack", (arrays1d,), {}, dn, "1d", "bit_exact", True, None)

        # where
        cond = np.array([True, False, True, False, True])
        yield TestCase("where", (cond, dt(10.0), dt(-1.0)), {}, dn, "scalar",
                       "bit_exact", True, None)
        yield TestCase("where", (cond,
                                 _make_extreme((5,), dt, "random", seed=1),
                                 _make_extreme((5,), dt, "random", seed=2)),
                       {}, dn, "array", "bit_exact", True, None)

        # roll, flip, repeat, tile
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        yield TestCase("roll", (a, 2), {}, dn, "+2", "bit_exact", True, None)
        yield TestCase("flip", (a[:4],), {}, dn, "4", "bit_exact", True, None)
        yield TestCase("repeat", (a[:3], 3), {}, dn, "x3", "bit_exact", True, None)
        yield TestCase("tile", (a[:3], 2), {}, dn, "x2", "bit_exact", True, None)

        # argsort
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dt)
        yield TestCase("argsort", (a,), {}, dn, "basic", "bit_exact", True, None)

        # cumsum
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        yield TestCase("cumsum", (a,), {}, dn, "basic", "bit_exact", True, None)

        # slice 2-arg
        a = _make_extreme((10, 3), dt, "random", seed=42)
        yield TestCase("slice", (a, 2, 7), {}, dn, "2:7", "bit_exact", True, None)
        yield TestCase("slice", (a, 0, 5), {}, dn, ":5", "bit_exact", True, None)

        # slice N-D
        a = _make_extreme((6, 8), dt, "random", seed=42)
        yield TestCase("slice", (a, [1, 2], [3, 5], [1, 1]), {}, dn, "nd_1:3_2:5",
                       "bit_exact", True, None)

        # take
        a = _make_extreme((5, 7), dt, "random", seed=42)
        idx = np.array([0, 2, 4], dtype=np.int64)
        yield TestCase("take", (a, idx, 0), {}, dn, "rows", "bit_exact", True, None)
        yield TestCase("take", (a, idx, 1), {}, dn, "cols", "bit_exact", True, None)

        # compress — condition FIRST, matching numpy np.compress(condition, a)
        mask = np.array([True, False, True, False, True])
        yield TestCase("compress", (mask, _make_extreme((5,), dt, "random", seed=42)),
                       {}, dn, "basic", "bit_exact", True, None)

    # argmax, argmin
    for name in ["argmax", "argmin"]:
        for dt in (np.float64, np.float32):
            dn = dt.__name__
            yield TestCase(name, (_make_extreme((100,), dt, "random", seed=42),),
                           {}, dn, "random", "scalar_eq", True, None)


# ── 第5类: reduce (reduce_py.h, 10 APIs) ─────────────────────────────────────

def _catalog_reduce():
    """reduce_py.h — 标量归约 / 轴归约 / 布尔归约"""
    for dt in (np.float64, np.float32):
        dn = dt.__name__

        # sum, max, min
        for name in ["sum", "max", "min"]:
            a = _make_extreme((100,), dt, "random", seed=42)
            yield TestCase(name, (a,), {}, dn, "random", "scalar_eq", True, None)
            yield TestCase(name, (np.zeros(10, dtype=dt),), {}, dn, "zeros",
                           "scalar_eq", True, None)

        # std, var — large size float64 only
        for name in ["std", "var"]:
            a = _make_extreme((100,), dt, "random", seed=42)
            yield TestCase(name, (a,), {}, dn, "random", "scalar_eq", True, None)
            yield TestCase(name, (np.ones(50, dtype=dt) * dt(3.0),), {}, dn, "const",
                           "scalar_eq", True, None)
        if dt == np.float64:
            for name in ["std", "var"]:
                yield TestCase(name, (_make_extreme((10000,), dt, "random", seed=7),),
                               {}, "float64", "large", "scalar_eq", True, None)

        # mean — scalar
        yield TestCase("mean", (_make_extreme((100,), dt, "random", seed=42),),
                       {}, dn, "scalar", "scalar_eq", True, None)
        # mean — axis
        for ax, tag in [(0, "ax0"), (1, "ax1"), (-1, "ax-1")]:
            yield TestCase("mean", (_make_extreme((4, 5), dt, "random", seed=42),),
                           {"axis": ax}, dn, tag, "bit_exact", True, None)

    # any, all
    for name in ["any", "all"]:
        yield TestCase(name, (np.array([True, False, True, False]),), {}, "bool", "mixed",
                       "scalar_eq", True, None)
        yield TestCase(name, (np.array([True, True, True]),), {}, "bool", "all_true",
                       "scalar_eq", True, None)

    # cumsum
    for dt in (np.float64, np.float32):
        dn = dt.__name__
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        yield TestCase("cumsum", (a,), {}, dn, "basic", "bit_exact", True, None)


# ═══════════════════════════════════════════════════════════════════════════════
# api_catalog() — 汇总入口
# ═══════════════════════════════════════════════════════════════════════════════

def api_catalog():
    """导出 numpypcpp 全部 API 的测试用例目录。"""
    for tc in _catalog_elementwise():
        yield tc._replace(group="elementwise")
    for tc in _catalog_init():
        yield tc._replace(group="init")
    for tc in _catalog_linalg():
        yield tc._replace(group="linalg")
    for tc in _catalog_manipulation():
        yield tc._replace(group="manipulation")
    for tc in _catalog_reduce():
        yield tc._replace(group="reduce")


# ═══════════════════════════════════════════════════════════════════════════════
# 模块加载
# ═══════════════════════════════════════════════════════════════════════════════

_cpp_module = None
_import_error = None


def _resolve_module_name():
    return (getattr(pytest, "_numpycpp_module_name", None)
            or os.environ.get("NUMPYCPP_MODULE")
            or "numpycpp")


def get_cpp_module():
    global _cpp_module, _import_error
    if _cpp_module is not None:
        return _cpp_module
    if _import_error is not None:
        raise _import_error
    try:
        _cpp_module = importlib.import_module(_resolve_module_name())
    except ImportError as e:
        _import_error = e
        raise
    return _cpp_module


@pytest.fixture(scope="session")
def cpp():
    return get_cpp_module()


# ═══════════════════════════════════════════════════════════════════════════════
# build_all_tests — catalog → pytest.param
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_tests():
    """将 api_catalog() 展开为 pytest 参数化列表。"""
    for tc in api_catalog():
        test_id = f"{tc.api_name}[{tc.dtype_label}][{tc.category}]"
        yield pytest.param(tc, id=test_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 唯一的参数化测试函数 + report.csv 收集
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_ROWS = []  # 全局收集: [{category, api, mode, dtype, feature, result, ulp}]


def _ulp_diff(cpp_r, py_r):
    """返回 C++ 与 numpy 结果的最大 ULP 差异。无法比较时返回 -1。"""
    cpp = np.asarray(cpp_r)
    py  = np.asarray(py_r)
    if cpp.dtype.kind != 'f' or cpp.dtype != py.dtype:
        return -1
    if cpp.size == 0:
        return 0
    sz = cpp.itemsize
    if sz not in (4, 8):
        return -1
    uint_t = np.uint32 if sz == 4 else np.uint64
    cu = cpp.ravel().view(uint_t).astype(np.int64)
    pu = py.ravel().view(uint_t).astype(np.int64)
    return int(np.max(np.abs(cu - pu)))


def _compile_mode(cpp):
    """检测编译模式: 'bit_exact' 或 'std'。"""
    try:
        return cpp.compile_mode()
    except AttributeError:
        return "unknown"


@pytest.mark.parametrize("tc", list(build_all_tests()))
def test_api(tc, cpp):
    """全量测试: F5→F1→F2→F4→F3 流水线 + 记录 report 行。"""
    api_name  = tc.api_name
    args      = tc.args
    kwargs    = tc.kwargs
    strategy  = tc.cmp_strategy
    check_py  = tc.check_py

    # 原地修改类 API：先 copy，调用后再比对 copy
    if tc.setup_fn is not None:
        tc.setup_fn(args, kwargs)
        _REPORT_ROWS.append(dict(
            category=tc.group, api=api_name, mode=_compile_mode(cpp),
            dtype=tc.dtype_label, feature=tc.category,
            result="SKIP", ulp=-1))
        return

    cpp_r, py_r = call_cpp_py(api_name, cpp, *args, **kwargs)
    if check_py and py_r is None:
        _REPORT_ROWS.append(dict(
            category=tc.group, api=api_name, mode=_compile_mode(cpp),
            dtype=tc.dtype_label, feature=tc.category,
            result="SKIP", ulp=-1))
        pytest.skip(f"no numpy equivalent for {api_name}")

    ulp = _ulp_diff(cpp_r, py_r) if strategy == "bit_exact" else -1
    try:
        compare(cpp_r, py_r, strategy=strategy, label=f"{api_name}")
        _REPORT_ROWS.append(dict(
            category=tc.group, api=api_name, mode=_compile_mode(cpp),
            dtype=tc.dtype_label, feature=tc.category,
            result="PASS", ulp=ulp))
    except AssertionError:
        _REPORT_ROWS.append(dict(
            category=tc.group, api=api_name, mode=_compile_mode(cpp),
            dtype=tc.dtype_label, feature=tc.category,
            result="FAIL", ulp=ulp))
        raise


@pytest.fixture(scope="session", autouse=True)
def _write_report_csv():
    """全部测试结束后写 report.csv。"""
    yield
    import csv
    fname = os.environ.get("REPORT_CSV", "report.csv")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    if not _REPORT_ROWS:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "api", "mode", "dtype",
                                          "feature", "result", "ulp"])
        w.writeheader()
        w.writerows(_REPORT_ROWS)
    print(f"\nreport.csv: {len(_REPORT_ROWS)} rows → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# __main__
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(pytest.main([__file__, "-q", "--tb=short", "--no-header"]))
