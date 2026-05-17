"""
Precision alignment tests for numpy.* (core.h / core.cpp).

Each test compares the C++ implementation against Python numpy using
identical inputs and verifies results match within configured tolerance.
Tests are parametrized over float64 and float32 dtypes for all template functions.
"""

import numpy as np
import pytest

from .utils import compare, random_array, tolerance_for


# ===========================================================================
# Array creation (template: T in → T out)
# ===========================================================================


class TestZerosLike:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4), dtype=dtype)
        info = compare(cpp.zeros_like(a), np.zeros_like(a), _rtol, _atol, "zeros_like")
        assert info["pass"], info.get("error")

    @pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (4, 5, 6)])
    def test_shapes(self, cpp, rtol, atol, dtype, shape):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array(shape, dtype=dtype)
        info = compare(cpp.zeros_like(a), np.zeros_like(a), _rtol, _atol, f"zeros_like{shape}")
        assert info["pass"], info.get("error")


class TestOnesLike:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4), dtype=dtype)
        info = compare(cpp.ones_like(a), np.ones_like(a), _rtol, _atol, "ones_like")
        assert info["pass"], info.get("error")

    @pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (1, 1, 1)])
    def test_shapes(self, cpp, rtol, atol, dtype, shape):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array(shape, dtype=dtype)
        info = compare(cpp.ones_like(a), np.ones_like(a), _rtol, _atol, f"ones_like{shape}")
        assert info["pass"], info.get("error")


class TestFullLike:
    @pytest.mark.parametrize("value", [0.0, 1.0, -3.14, 42.0, 1e10])
    def test_values(self, cpp, rtol, atol, dtype, value):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4), dtype=dtype)
        info = compare(cpp.full_like(a, value), np.full_like(a, value), _rtol, _atol, f"full_like({value})")
        assert info["pass"], info.get("error")


class TestEmptyLike:
    def test_shape_matches(self, cpp, rtol, atol, dtype):
        a = random_array((3, 4), dtype=dtype)
        cpp_r = cpp.empty_like(a)
        assert np.asarray(cpp_r).shape == a.shape, "empty_like shape mismatch"


# ===========================================================================
# Array creation (non-template: bool / double-only)
# ===========================================================================


class TestFullLikeBool:
    @pytest.mark.parametrize("value", [True, False])
    def test_values(self, cpp, rtol, atol, value):
        a = random_array((3, 4))
        py_r = np.full_like(a, value, dtype=bool)
        cpp_r = cpp.full_like_bool(a, value)
        assert np.array_equal(cpp_r, py_r), f"full_like_bool({value}) mismatch"


class TestZerosLikeBool:
    def test_basic(self, cpp, rtol, atol):
        a = random_array((3, 4))
        cpp_r = cpp.zeros_like_bool(a)
        py_r = np.zeros_like(a, dtype=bool)
        assert np.array_equal(cpp_r, py_r), "zeros_like_bool mismatch"


class TestOnesLikeBool:
    def test_basic(self, cpp, rtol, atol):
        a = random_array((3, 4))
        cpp_r = cpp.ones_like_bool(a)
        py_r = np.ones_like(a, dtype=bool)
        assert np.array_equal(cpp_r, py_r), "ones_like_bool mismatch"


class TestZeros:
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, rtol, atol, shape):
        info = compare(cpp.zeros(shape), np.zeros(shape), rtol, atol, f"zeros{shape}")
        assert info["pass"], info.get("error")


class TestOnes:
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, rtol, atol, shape):
        info = compare(cpp.ones(shape), np.ones(shape), rtol, atol, f"ones{shape}")
        assert info["pass"], info.get("error")


# ===========================================================================
# astype (non-template: specific types)
# ===========================================================================


class TestAstypeInt:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([[1.7, 2.3], [-3.9, 0.5]], dtype=np.float64)
        cpp_r = np.asarray(cpp.astype_int(a))
        py_r = a.astype(np.int32)
        assert np.array_equal(cpp_r, py_r), "astype_int mismatch"


class TestAstypeBool:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([[0.0, 1.0, -1.0], [3.14, 0.0, 0.0]], dtype=np.float64)
        cpp_r = np.asarray(cpp.astype_bool(a))
        py_r = a.astype(bool)
        assert np.array_equal(cpp_r, py_r), "astype_bool mismatch"


class TestAstypeBoolFromInt:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([[0, 1, -1], [42, 0, 0]], dtype=np.int32)
        cpp_r = np.asarray(cpp.astype_bool_from_int(a))
        py_r = a.astype(bool)
        assert np.array_equal(cpp_r, py_r), "astype_bool_from_int mismatch"


class TestTruncateToFloat32:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([1.0 / 3.0, np.pi, np.sqrt(2.0)], dtype=np.float64)
        cpp_r = np.asarray(cpp.truncate_to_float32(a))
        py_r = a.astype(np.float32).astype(np.float64)
        info = compare(cpp_r, py_r, rtol, atol, "truncate_to_float32")
        assert info["pass"], info.get("error")


# ===========================================================================
# Element-wise math (template: T in → T out)
# ===========================================================================


class TestSqrt:
    @pytest.mark.parametrize("shape", [(10,), (3, 4), (2, 3, 4)])
    def test_shapes(self, cpp, rtol, atol, dtype, shape):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.abs(random_array(shape, dtype=dtype))
        info = compare(cpp.sqrt(a), np.sqrt(a), _rtol, _atol, f"sqrt{shape}")
        assert info["pass"], info.get("error")

    def test_zero(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.zeros((5,), dtype=dtype)
        info = compare(cpp.sqrt(a), np.sqrt(a), _rtol, _atol, "sqrt zero")
        assert info["pass"], info.get("error")


class TestAbs:
    @pytest.mark.parametrize("shape", [(10,), (3, 4)])
    def test_shapes(self, cpp, rtol, atol, dtype, shape):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array(shape, dtype=dtype)
        info = compare(cpp.abs(a), np.abs(a), _rtol, _atol, f"abs{shape}")
        assert info["pass"], info.get("error")


class TestExp:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), seed=1, dtype=dtype)
        info = compare(cpp.exp(a), np.exp(a), _rtol, _atol, label="exp")
        assert info["pass"], info.get("error")


class TestLog:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.abs(random_array((10,), dtype=dtype)) + 0.1
        info = compare(cpp.log(a), np.log(a), _rtol, _atol, label="log")
        assert info["pass"], info.get("error")


class TestSin:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.sin(a), np.sin(a), _rtol, _atol, label="sin")
        assert info["pass"], info.get("error")


class TestCos:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.cos(a), np.cos(a), _rtol, _atol, label="cos")
        assert info["pass"], info.get("error")


class TestTan:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        _rtol = max(_rtol, 1e-7)
        _atol = max(_atol, 1e-7)
        a = random_array((10,), dtype=dtype) * 0.5  # avoid poles
        info = compare(cpp.tan(a), np.tan(a), _rtol, _atol, label="tan")
        assert info["pass"], info.get("error")


class TestPower:
    @pytest.mark.parametrize("exp", [2.0, 3.0, 0.5, -1.0, 0.0])
    def test_exponents(self, cpp, rtol, atol, dtype, exp):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.abs(random_array((10,), dtype=dtype)) + 0.01
        info = compare(cpp.power(a, exp), np.power(a, exp), _rtol, _atol, label=f"power({exp})")
        assert info["pass"], info.get("error")


class TestClip:
    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-1.0, 1.0), (0.5, 0.5), (-10.0, 10.0)])
    def test_bounds(self, cpp, rtol, atol, dtype, lo, hi):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((20,), dtype=dtype)
        info = compare(cpp.clip(a, lo, hi), np.clip(a, lo, hi), _rtol, _atol, f"clip({lo},{hi})")
        assert info["pass"], info.get("error")


# ===========================================================================
# Reduction (template: T in → T out)
# ===========================================================================


class TestSum:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        cpp_r, py_r = cpp.sum(a), np.sum(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"sum: {cpp_r} vs {py_r}"

    def test_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        cpp_r, py_r = cpp.sum(a), np.sum(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"sum 2d: {cpp_r} vs {py_r}"


class TestMean:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        cpp_r, py_r = cpp.mean(a), np.mean(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"mean: {cpp_r} vs {py_r}"

    def test_empty(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([], dtype=dtype)
        cpp_r, py_r = cpp.mean(a), 0.0  # C++ returns 0 for empty
        assert np.allclose(cpp_r, py_r, _rtol, _atol)


class TestMax:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        cpp_r, py_r = cpp.max(a), np.max(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"max: {cpp_r} vs {py_r}"


class TestMin:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        cpp_r, py_r = cpp.min(a), np.min(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"min: {cpp_r} vs {py_r}"


# ===========================================================================
# Reduction (bool-only)
# ===========================================================================


class TestAny:
    def test_true(self, cpp, rtol, atol):
        a = np.array([False, False, True, False])
        assert cpp.any(a) == np.any(a)

    def test_false(self, cpp, rtol, atol):
        a = np.array([False, False, False])
        assert cpp.any(a) == np.any(a)


class TestAll:
    def test_true(self, cpp, rtol, atol):
        a = np.array([True, True, True])
        assert cpp.all(a) == np.all(a)

    def test_false(self, cpp, rtol, atol):
        a = np.array([True, False, True])
        assert cpp.all(a) == np.all(a)


# ===========================================================================
# Comparison (template: T in → bool out)
# ===========================================================================


class TestGreater:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = random_array((10,), dtype=dtype)
        b = dtype(0.0)
        assert np.array_equal(cpp.greater(a, b), np.greater(a, b))


class TestLess:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = random_array((10,), dtype=dtype)
        b = dtype(0.0)
        assert np.array_equal(cpp.less(a, b), np.less(a, b))


class TestEqual:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([0.0, 1.0, 1.0, 0.0], dtype=dtype)
        assert np.array_equal(cpp.equal(a, dtype(1.0)), np.equal(a, dtype(1.0)))


class TestGreaterEqual:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = random_array((10,), dtype=dtype)
        b = dtype(0.0)
        assert np.array_equal(cpp.greater_equal(a, b), np.greater_equal(a, b))


class TestLessEqual:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = random_array((10,), dtype=dtype)
        b = dtype(0.0)
        assert np.array_equal(cpp.less_equal(a, b), np.less_equal(a, b))


class TestNotEqual:
    def test_scalar(self, cpp, rtol, atol, dtype):
        a = np.array([0.0, 1.0, 0.0], dtype=dtype)
        assert np.array_equal(cpp.not_equal(a, dtype(0.0)), np.not_equal(a, dtype(0.0)))

    def test_array(self, cpp, rtol, atol, dtype):
        a = random_array((10,), dtype=dtype)
        b = random_array((10,), seed=99, dtype=dtype)
        assert np.array_equal(cpp.not_equal(a, b), np.not_equal(a, b))


# ===========================================================================
# Logical (bool-only)
# ===========================================================================


class TestLogicalAnd:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert np.array_equal(cpp.logical_and(a, b), np.logical_and(a, b))


class TestLogicalOr:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert np.array_equal(cpp.logical_or(a, b), np.logical_or(a, b))


class TestLogicalNot:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([True, False, True])
        assert np.array_equal(cpp.logical_not(a), np.logical_not(a))


class TestLogicalXor:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        assert np.array_equal(cpp.logical_xor(a, b), np.logical_xor(a, b))


# ===========================================================================
# Special value helpers (template: T in → bool out)
# ===========================================================================


class TestIsnan:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([0.0, np.nan, 1.0, np.nan], dtype=dtype)
        assert np.array_equal(cpp.isnan(a), np.isnan(a))


class TestIsinf:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([0.0, np.inf, -np.inf, 1.0], dtype=dtype)
        assert np.array_equal(cpp.isinf(a), np.isinf(a))


class TestIsfinite:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([0.0, np.inf, np.nan, 1.0, -np.inf], dtype=dtype)
        assert np.array_equal(cpp.isfinite(a), np.isfinite(a))


# ===========================================================================
# Additional math (template: T in → T out)
# ===========================================================================


class TestLog10:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.abs(random_array((10,), dtype=dtype)) + 0.1
        info = compare(cpp.log10(a), np.log10(a), _rtol, _atol, label="log10")
        assert info["pass"], info.get("error")


class TestLog2:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.abs(random_array((10,), dtype=dtype)) + 0.1
        info = compare(cpp.log2(a), np.log2(a), _rtol, _atol, label="log2")
        assert info["pass"], info.get("error")


class TestArcsin:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype) * 0.5  # domain [-0.5, 0.5]
        info = compare(cpp.arcsin(a), np.arcsin(a), _rtol, _atol, label="arcsin")
        assert info["pass"], info.get("error")


class TestArccos:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype) * 0.5  # domain [-0.5, 0.5]
        info = compare(cpp.arccos(a), np.arccos(a), _rtol, _atol, label="arccos")
        assert info["pass"], info.get("error")


class TestArctan:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.arctan(a), np.arctan(a), _rtol, _atol, label="arctan")
        assert info["pass"], info.get("error")


class TestRound:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype) * 10
        info = compare(cpp.round(a), np.round(a), _rtol, _atol, "round")
        assert info["pass"], info.get("error")


class TestFloor:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype) * 10
        info = compare(cpp.floor(a), np.floor(a), _rtol, _atol, "floor")
        assert info["pass"], info.get("error")


class TestCeil:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype) * 10
        info = compare(cpp.ceil(a), np.ceil(a), _rtol, _atol, "ceil")
        assert info["pass"], info.get("error")


class TestDegrees:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.degrees(a), np.degrees(a), _rtol, _atol, "degrees")
        assert info["pass"], info.get("error")


class TestRadians:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.radians(a), np.radians(a), _rtol, _atol, "radians")
        assert info["pass"], info.get("error")


class TestSign:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.sign(a), np.sign(a), _rtol, _atol, "sign")
        assert info["pass"], info.get("error")

    def test_zero(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([0.0, -0.0, 0.0], dtype=dtype)
        info = compare(cpp.sign(a), np.sign(a), _rtol, _atol, "sign zero")
        assert info["pass"], info.get("error")


# ===========================================================================
# Binary element-wise (template: T in → T out)
# ===========================================================================


class TestArctan2:
    def test_array_array(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        b = np.abs(random_array((10,), dtype=dtype)) + 0.1
        info = compare(cpp.arctan2(a, b), np.arctan2(a, b), _rtol, _atol, label="arctan2(a,b)")
        assert info["pass"], info.get("error")

    def test_array_scalar(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.arctan2(a, dtype(1.0)), np.arctan2(a, dtype(1.0)), _rtol, _atol, label="arctan2(a,1)")
        assert info["pass"], info.get("error")


class TestMaximum:
    def test_array_array(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), seed=1, dtype=dtype)
        b = random_array((10,), seed=2, dtype=dtype)
        info = compare(cpp.maximum(a, b), np.maximum(a, b), _rtol, _atol, "maximum(a,b)")
        assert info["pass"], info.get("error")

    def test_array_scalar(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.maximum(a, dtype(0.0)), np.maximum(a, dtype(0.0)), _rtol, _atol, "maximum(a,0)")
        assert info["pass"], info.get("error")


class TestMinimum:
    def test_array_array(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), seed=1, dtype=dtype)
        b = random_array((10,), seed=2, dtype=dtype)
        info = compare(cpp.minimum(a, b), np.minimum(a, b), _rtol, _atol, "minimum(a,b)")
        assert info["pass"], info.get("error")

    def test_array_scalar(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        info = compare(cpp.minimum(a, dtype(0.0)), np.minimum(a, dtype(0.0)), _rtol, _atol, "minimum(a,0)")
        assert info["pass"], info.get("error")


# ===========================================================================
# Array manipulation (template: T in → T out)
# ===========================================================================


class TestDiff:
    def test_1d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 3.0, 6.0, 10.0], dtype=dtype)
        info = compare(cpp.diff(a), np.diff(a), _rtol, _atol, "diff 1d")
        assert info["pass"], info.get("error")

    def test_2d_axis0(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        info = compare(cpp.diff(a, 1, 0), np.diff(a, n=1, axis=0), _rtol, _atol, "diff axis=0")
        assert info["pass"], info.get("error")

    def test_2d_axis1(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        info = compare(cpp.diff(a, 1, 1), np.diff(a, n=1, axis=1), _rtol, _atol, "diff axis=1")
        assert info["pass"], info.get("error")

    def test_2d_axis_neg1(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        info = compare(cpp.diff(a, 1, -1), np.diff(a, n=1, axis=-1), _rtol, _atol, "diff axis=-1")
        assert info["pass"], info.get("error")


class TestStack:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(4)]
        info = compare(cpp.stack(arrays), np.stack(arrays), _rtol, _atol, "stack")
        assert info["pass"], info.get("error")


class TestConcatenate:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
        info = compare(cpp.concatenate(arrays), np.concatenate(arrays), _rtol, _atol, "concatenate")
        assert info["pass"], info.get("error")


class TestVstack:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        arrays = [random_array((1, 3), seed=i, dtype=dtype) for i in range(4)]
        info = compare(cpp.vstack(arrays), np.vstack(arrays), _rtol, _atol, "vstack")
        assert info["pass"], info.get("error")


class TestHstack:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        arrays = [random_array((3,), seed=i, dtype=dtype) for i in range(3)]
        info = compare(cpp.hstack(arrays), np.hstack(arrays), _rtol, _atol, "hstack")
        assert info["pass"], info.get("error")


class TestWhere:
    def test_scalar(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        cond = np.array([True, False, True, False, True])
        info = compare(cpp.where(cond, dtype(10.0), dtype(-1.0)),
                       np.where(cond, dtype(10.0), dtype(-1.0)),
                       _rtol, _atol, "where scalar")
        assert info["pass"], info.get("error")

    def test_array(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        cond = np.array([True, False, True, False])
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        y = np.array([-1.0, -2.0, -3.0, -4.0], dtype=dtype)
        info = compare(cpp.where(cond, x, y), np.where(cond, x, y), _rtol, _atol, "where array")
        assert info["pass"], info.get("error")


class TestTranspose:
    def test_1d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5,), dtype=dtype)
        info = compare(cpp.transpose(a), np.transpose(a), _rtol, _atol, "transpose 1d")
        assert info["pass"], info.get("error")

    def test_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 5), dtype=dtype)
        info = compare(cpp.transpose(a), np.transpose(a), _rtol, _atol, "transpose 2d")
        assert info["pass"], info.get("error")


class TestFlatten:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4), dtype=dtype)
        info = compare(cpp.flatten(a), a.flatten(), _rtol, _atol, "flatten")
        assert info["pass"], info.get("error")


class TestMeanAxis:
    def test_axis0_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 5), dtype=dtype)
        info = compare(cpp.mean(a, 0), np.mean(a, axis=0), _rtol, _atol, "mean axis=0")
        assert info["pass"], info.get("error")

    def test_axis1_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 5), dtype=dtype)
        info = compare(cpp.mean(a, 1), np.mean(a, axis=1), _rtol, _atol, "mean axis=1")
        assert info["pass"], info.get("error")

    def test_axis_neg1_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 5), dtype=dtype)
        info = compare(cpp.mean(a, -1), np.mean(a, axis=-1), _rtol, _atol, "mean axis=-1")
        assert info["pass"], info.get("error")

    def test_axis0_3d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4, 5), dtype=dtype)
        info = compare(cpp.mean(a, 0), np.mean(a, axis=0), _rtol, _atol, "mean 3d axis=0")
        assert info["pass"], info.get("error")

    def test_axis1_3d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4, 5), dtype=dtype)
        info = compare(cpp.mean(a, 1), np.mean(a, axis=1), _rtol, _atol, "mean 3d axis=1")
        assert info["pass"], info.get("error")

    def test_axis2_3d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((3, 4, 5), dtype=dtype)
        info = compare(cpp.mean(a, 2), np.mean(a, axis=2), _rtol, _atol, "mean 3d axis=2")
        assert info["pass"], info.get("error")


class TestSlice:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10, 3), dtype=dtype)
        info = compare(cpp.slice(a, 2, 7), a[2:7], _rtol, _atol, "slice 2:7")
        assert info["pass"], info.get("error")

    def test_from_start(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10, 3), dtype=dtype)
        info = compare(cpp.slice(a, 0, 5), a[0:5], _rtol, _atol, "slice :5")
        assert info["pass"], info.get("error")


class TestTakeCols:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 6), dtype=dtype)
        info = compare(cpp.take_cols(a, 3), a[:, :3], _rtol, _atol, "take_cols 3")
        assert info["pass"], info.get("error")


class TestSliceAssign:
    def test_double(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        cpp.slice_assign(a, 2, float(99.0))
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        expected[2:] = float(99.0)
        info = compare(a, expected, _rtol, _atol, "slice_assign double")
        assert info["pass"], info.get("error")

    def test_int(self, cpp, rtol, atol):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        cpp.slice_assign(a, 3, -1)
        expected = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected[3:] = -1
        assert np.array_equal(a, expected), "slice_assign int mismatch"

    def test_bool(self, cpp, rtol, atol):
        a = np.array([True, False, True, False], dtype=bool)
        cpp.slice_assign(a, 2, False)
        expected = np.array([True, False, True, False], dtype=bool)
        expected[2:] = False
        assert np.array_equal(a, expected), "slice_assign bool mismatch"


class TestRoll:
    def test_positive(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        info = compare(cpp.roll(a, 2), np.roll(a, 2), _rtol, _atol, "roll +2")
        assert info["pass"], info.get("error")

    def test_negative(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        info = compare(cpp.roll(a, -1), np.roll(a, -1), _rtol, _atol, "roll -1")
        assert info["pass"], info.get("error")


class TestFlip:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        info = compare(cpp.flip(a), np.flip(a), _rtol, _atol, "flip")
        assert info["pass"], info.get("error")


class TestRepeat:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        info = compare(cpp.repeat(a, 3), np.repeat(a, 3), _rtol, _atol, "repeat")
        assert info["pass"], info.get("error")


class TestTile:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        info = compare(cpp.tile(a, 2), np.tile(a, 2), _rtol, _atol, "tile")
        assert info["pass"], info.get("error")


# ===========================================================================
# Statistical (template: T in → T out)
# ===========================================================================


class TestStd:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((100,), dtype=dtype)
        cpp_r, py_r = cpp.std(a), np.std(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"std: {cpp_r} vs {py_r}"

    def test_constant(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.ones((50,), dtype=dtype) * dtype(3.0)
        cpp_r, py_r = cpp.std(a), np.std(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"std constant: {cpp_r} vs {py_r}"


class TestVar:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((100,), dtype=dtype)
        cpp_r, py_r = cpp.var(a), np.var(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"var: {cpp_r} vs {py_r}"


# ===========================================================================
# Sorting and indexing (template: T in → ssize_t out)
# ===========================================================================


class TestArgsort:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dtype)
        cpp_r = np.asarray(cpp.argsort(a))
        py_r = np.argsort(a, kind='stable')
        assert np.array_equal(cpp_r, py_r), f"argsort: {cpp_r} vs {py_r}"


class TestArgmax:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([1.0, 5.0, 3.0, 9.0, 2.0], dtype=dtype)
        assert cpp.argmax(a) == np.argmax(a)


class TestArgmin:
    def test_basic(self, cpp, rtol, atol, dtype):
        a = np.array([5.0, 1.0, 3.0, 0.5, 2.0], dtype=dtype)
        assert cpp.argmin(a) == np.argmin(a)


# ===========================================================================
# Set operations / Interpolation (double-only)
# ===========================================================================


class TestIsin:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values = [2.0, 4.0, 6.0]
        assert np.array_equal(cpp.isin(a, values), np.isin(a, values))


class TestIntersect1d:
    def test_basic(self, cpp, rtol, atol):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        cpp_r = np.sort(np.asarray(cpp.intersect1d(a, b)))
        py_r = np.intersect1d(a, b)
        info = compare(cpp_r, py_r, rtol, atol, "intersect1d")
        assert info["pass"], info.get("error")


class TestInterp:
    def test_basic(self, cpp, rtol, atol):
        xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        info = compare(cpp.interp(x, xp, fp), np.interp(x, xp, fp), rtol, atol, "interp")
        assert info["pass"], info.get("error")

    def test_extrap_low(self, cpp, rtol, atol):
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([0.0, 0.5])
        info = compare(cpp.interp(x, xp, fp), np.interp(x, xp, fp), rtol, atol, "interp low")
        assert info["pass"], info.get("error")

    def test_extrap_high(self, cpp, rtol, atol):
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([3.5, 5.0])
        info = compare(cpp.interp(x, xp, fp), np.interp(x, xp, fp), rtol, atol, "interp high")
        assert info["pass"], info.get("error")


# ===========================================================================
# Safe division (double-only)
# ===========================================================================


class TestSafeDivide:
    def test_normal(self, cpp, rtol, atol):
        assert np.allclose(cpp.safe_divide(10.0, 2.0), 5.0)

    def test_zero_denom(self, cpp, rtol, atol):
        assert np.allclose(cpp.safe_divide(10.0, 0.0, -1.0), -1.0)

    def test_custom_default(self, cpp, rtol, atol):
        assert np.allclose(cpp.safe_divide(10.0, 0.0, 99.0), 99.0)


# ===========================================================================
# Array access (non-template: fixed types)
# ===========================================================================


class TestArrayGet:
    def test_1d(self, cpp, rtol, atol):
        a = np.array([10.0, 20.0, 30.0])
        assert cpp.array_get(a, 0) == 10.0
        assert cpp.array_get(a, 2) == 30.0
        assert cpp.array_get(a, -1) == 30.0

    def test_2d(self, cpp, rtol, atol):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert cpp.array_get(a, 0, 0) == 1.0
        assert cpp.array_get(a, 1, 1) == 4.0


# ===========================================================================
# Conversion (non-template: double-only)
# ===========================================================================


class TestAsarray:
    def test_vector(self, cpp, rtol, atol):
        v = [1.0, 2.0, 3.0]
        info = compare(cpp.asarray(v), np.asarray(v), rtol, atol, "asarray vec")
        assert info["pass"], info.get("error")

    def test_array(self, cpp, rtol, atol):
        a = np.array([1.0, 2.0, 3.0])
        info = compare(cpp.asarray(a), np.asarray(a), rtol, atol, "asarray arr")
        assert info["pass"], info.get("error")


# ===========================================================================
# to_vector (non-template: bool/double)
# ===========================================================================


class TestToVector:
    def test_double(self, cpp, rtol, atol):
        a = np.array([1.0, 2.0, 3.0])
        v = cpp.to_vector(a)
        assert list(v) == [1.0, 2.0, 3.0]

    def test_bool(self, cpp, rtol, atol):
        a = np.array([True, False, True])
        v = cpp.to_vector(a)
        assert list(v) == [True, False, True]
