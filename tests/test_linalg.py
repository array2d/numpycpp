"""
Precision alignment tests for numpy.linalg.* and numpy.dot (linalg.h / linalg.cpp).
Tests are parametrized over float64 and float32 dtypes.
"""

import numpy as np
import pytest

from .utils import compare, random_array, tolerance_for


class TestNorm:
    def test_1d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((10,), dtype=dtype)
        cpp_r = cpp.linalg.norm(a)
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"norm: {cpp_r} vs {py_r}"

    def test_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        cpp_r = cpp.linalg.norm(a)
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"norm 2d: {cpp_r} vs {py_r}"

    def test_zero(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.zeros((10,), dtype=dtype)
        cpp_r = cpp.linalg.norm(a)
        assert np.allclose(cpp_r, 0.0, _rtol, _atol)


class TestNormAxis:
    def test_2d(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5, 4), dtype=dtype)
        info = compare(cpp.linalg.norm(a, axis=1), np.linalg.norm(a, axis=1), _rtol, _atol, "norm axis=1")
        assert info["pass"], info.get("error")

    def test_1d_fallback(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([3.0, 4.0], dtype=dtype)  # norm = 5.0
        cpp_r = np.asarray(cpp.linalg.norm(a)).item()
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, _rtol, _atol)


class TestDot:
    def test_basic(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((5,), dtype=dtype)
        b = random_array((5,), seed=99, dtype=dtype)
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a, b)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"dot: {cpp_r} vs {py_r}"

    def test_different_sizes(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        b = np.array([0.5, 1.5], dtype=dtype)
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a[:2], b)  # C++ uses min(size_a, size_b)
        assert np.allclose(cpp_r, py_r, _rtol, _atol), f"dot diff sizes: {cpp_r} vs {py_r}"

    def test_orthogonal(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([1.0, 0.0], dtype=dtype)
        b = np.array([0.0, 1.0], dtype=dtype)
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a, b)
        assert np.allclose(cpp_r, py_r, _rtol, _atol)
