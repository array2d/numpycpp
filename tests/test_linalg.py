"""
Precision alignment tests for numpy.linalg.* and numpy.dot (linalg.h / linalg.cpp).
"""

import numpy as np
import pytest

from conftest import compare, random_array


class TestNorm:
    def test_1d_double(self, cpp, rtol, atol):
        a = random_array((10,))
        cpp_r = cpp.linalg.norm(a)
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, rtol, atol), f"norm: {cpp_r} vs {py_r}"

    def test_2d_double(self, cpp, rtol, atol):
        a = random_array((5, 4))
        cpp_r = cpp.linalg.norm(a)
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, rtol, atol), f"norm 2d: {cpp_r} vs {py_r}"

    def test_zero(self, cpp, rtol, atol):
        a = np.zeros((10,), dtype=np.float64)
        cpp_r = cpp.linalg.norm(a)
        assert np.allclose(cpp_r, 0.0, rtol, atol)

    def test_float32_input(self, cpp, rtol, atol):
        """norm with float32 input: C++ computes in float32 internally."""
        a = np.random.randn(10).astype(np.float32)
        cpp_r = cpp.linalg.norm(a)
        py_r = np.linalg.norm(a)
        # float32 computation has lower precision
        assert np.allclose(cpp_r, py_r, rtol=1e-6, atol=1e-6), f"norm f32: {cpp_r} vs {py_r}"


class TestNormAxis1:
    def test_2d(self, cpp, rtol, atol):
        a = random_array((5, 4))
        info = compare(cpp.linalg.norm_axis1(a), np.linalg.norm(a, axis=1), rtol, atol, "norm axis=1")
        assert info["pass"], info.get("error")

    def test_1d_fallback(self, cpp, rtol, atol):
        a = np.array([3.0, 4.0])  # norm = 5.0
        cpp_r = np.asarray(cpp.linalg.norm_axis1(a)).item()
        py_r = np.linalg.norm(a)
        assert np.allclose(cpp_r, py_r, rtol, atol)


class TestDot:
    def test_basic(self, cpp, rtol, atol):
        a = random_array((5,))
        b = random_array((5,), seed=99)
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a, b)
        assert np.allclose(cpp_r, py_r, rtol, atol), f"dot: {cpp_r} vs {py_r}"

    def test_different_sizes(self, cpp, rtol, atol):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.5])
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a[:2], b)  # C++ uses min(size_a, size_b)
        assert np.allclose(cpp_r, py_r, rtol, atol), f"dot diff sizes: {cpp_r} vs {py_r}"

    def test_orthogonal(self, cpp, rtol, atol):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        cpp_r = cpp.dot(a, b)
        py_r = np.dot(a, b)
        assert np.allclose(cpp_r, py_r, rtol, atol)
