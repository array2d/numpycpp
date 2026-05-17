"""
Precision alignment tests for numpy.einsum (einsum.h / einsum.cpp).

Covers all supported einsum patterns with both explicit ('->') and implicit modes.
Tests are parametrized over float64 and float32 dtypes.
"""

import numpy as np
import pytest

from .utils import compare, random_array, tolerance_for


# ---------------------------------------------------------------------------
# Explicit mode patterns
# ---------------------------------------------------------------------------


class TestEinsum_ij_ij_to_i:
    """Pattern: 'ij,ij->i' — row-wise dot product."""

    def test_fixed_small(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, "ij,ij->i fixed")
        assert info["pass"], info.get("error")

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 2), (10, 5), (100, 3)])
    def test_random(self, cpp, rtol, atol, dtype, m, n):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((m, n), seed=1, dtype=dtype)
        b = random_array((m, n), seed=2, dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, f"ij,ij->i [{m},{n}]")
        assert info["pass"], info.get("error")


class TestEinsum_ij_jk_to_ik:
    """Pattern: 'ij,jk->ik' — matrix multiplication."""

    def test_fixed(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        info = compare(cpp.einsum("ij,jk->ik", a, b), np.einsum("ij,jk->ik", a, b), _rtol, _atol, "ij,jk->ik fixed")
        assert info["pass"], info.get("error")

    @pytest.mark.parametrize("i,j,k", [(1, 3, 2), (3, 3, 3), (10, 5, 10)])
    def test_random(self, cpp, rtol, atol, dtype, i, j, k):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((i, j), seed=1, dtype=dtype)
        b = random_array((j, k), seed=2, dtype=dtype)
        info = compare(cpp.einsum("ij,jk->ik", a, b), np.einsum("ij,jk->ik", a, b), _rtol, _atol, f"ij,jk->ik [{i},{j},{k}]")
        assert info["pass"], info.get("error")

    def test_vs_matmul(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 5), seed=1, dtype=dtype)
        b = random_array((5, 3), seed=2, dtype=dtype)
        cpp_r = np.asarray(cpp.einsum("ij,jk->ik", a, b))
        np_r = a @ b
        info = compare(cpp_r, np_r, _rtol, _atol, "ij,jk->ik vs matmul")
        assert info["pass"], info.get("error")


class TestEinsum_bij_bjk_to_bik:
    """Pattern: 'bij,bjk->bik' — batch matrix multiplication."""

    def test_fixed(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtype)
        b = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=dtype)
        info = compare(cpp.einsum("bij,bjk->bik", a, b), np.einsum("bij,bjk->bik", a, b), _rtol, _atol, "bij,bjk->bik fixed")
        assert info["pass"], info.get("error")

    @pytest.mark.parametrize("batch,i,j,k", [(1, 3, 2, 4), (3, 3, 3, 3), (5, 2, 5, 3)])
    def test_random(self, cpp, rtol, atol, dtype, batch, i, j, k):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((batch, i, j), seed=1, dtype=dtype)
        b = random_array((batch, j, k), seed=2, dtype=dtype)
        info = compare(cpp.einsum("bij,bjk->bik", a, b), np.einsum("bij,bjk->bik", a, b), _rtol, _atol, f"bij,bjk->bik [{batch},{i},{j},{k}]")
        assert info["pass"], info.get("error")

    def test_vs_batch_matmul(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((4, 5, 6), seed=1, dtype=dtype)
        b = random_array((4, 6, 3), seed=2, dtype=dtype)
        cpp_r = np.asarray(cpp.einsum("bij,bjk->bik", a, b))
        np_r = a @ b
        info = compare(cpp_r, np_r, _rtol, _atol, "bij,bjk->bik vs batch matmul")
        assert info["pass"], info.get("error")


class TestEinsum_aij_aij_to_ai:
    """Pattern: 'aij,aij->ai' — batch row-wise dot product."""

    @pytest.mark.parametrize("a_sz,i_sz,j_sz", [(1, 3, 2), (3, 5, 4), (5, 10, 3)])
    def test_random(self, cpp, rtol, atol, dtype, a_sz, i_sz, j_sz):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((a_sz, i_sz, j_sz), seed=1, dtype=dtype)
        b = random_array((a_sz, i_sz, j_sz), seed=2, dtype=dtype)
        info = compare(cpp.einsum("aij,aij->ai", a, b), np.einsum("aij,aij->ai", a, b), _rtol, _atol, f"aij,aij->ai [{a_sz},{i_sz},{j_sz}]")
        assert info["pass"], info.get("error")


class TestEinsum_broadcast_matmul:
    """Pattern: 'ijk,mkl->mjl' — broadcast matmul with contraction."""

    @pytest.mark.parametrize("i,j,k,m,l", [(2, 3, 2, 3, 2), (1, 3, 2, 2, 3), (3, 2, 3, 2, 2)])
    def test_random(self, cpp, rtol, atol, dtype, i, j, k, m, l):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((i, j, k), seed=1, dtype=dtype)
        b = random_array((m, k, l), seed=2, dtype=dtype)
        info = compare(cpp.einsum("ijk,mkl->mjl", a, b), np.einsum("ijk,mkl->mjl", a, b), _rtol, _atol, f"ijk,mkl->mjl [{i},{j},{k},{m},{l}]")
        assert info["pass"], info.get("error")


class TestEinsum_nij_nmj_to_nmi:
    """Pattern: 'nij,nmj->nmi' — batched matmul with double transpose."""

    @pytest.mark.parametrize("n,i,j,m", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, rtol, atol, dtype, n, i, j, m):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((n, i, j), seed=1, dtype=dtype)
        b = random_array((n, m, j), seed=2, dtype=dtype)
        info = compare(cpp.einsum("nij,nmj->nmi", a, b), np.einsum("nij,nmj->nmi", a, b), _rtol, _atol, f"nij,nmj->nmi [{n},{i},{j},{m}]")
        assert info["pass"], info.get("error")


class TestEinsum_aij_jka_to_aik:
    """Pattern: 'aij,jka->aik' — batch matmul with transpose."""

    @pytest.mark.parametrize("a_sz,i,j,k", [(1, 3, 3, 2), (3, 2, 4, 3), (5, 3, 2, 4)])
    def test_random(self, cpp, rtol, atol, dtype, a_sz, i, j, k):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((a_sz, i, j), seed=1, dtype=dtype)
        b = random_array((j, k, a_sz), seed=2, dtype=dtype)
        info = compare(cpp.einsum("aij,jka->aik", a, b), np.einsum("aij,jka->aik", a, b), _rtol, _atol, f"aij,jka->aik [{a_sz},{i},{j},{k}]")
        assert info["pass"], info.get("error")


# ---------------------------------------------------------------------------
# Implicit mode patterns (no '->')
# ---------------------------------------------------------------------------


class TestEinsumImplicit:
    """Implicit mode: output labels are those appearing exactly once."""

    def test_ij_ij(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=dtype)
        info = compare(cpp.einsum("ij,ij", a, b), np.einsum("ij,ij", a, b), _rtol, _atol, "ij,ij implicit")
        assert info["pass"], info.get("error")

    def test_ij_jk(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
        b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=dtype)
        info = compare(cpp.einsum("ij,jk", a, b), np.einsum("ij,jk", a, b), _rtol, _atol, "ij,jk implicit")
        assert info["pass"], info.get("error")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEinsumEdgeCases:
    def test_single_element(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1.0]], dtype=dtype)
        b = np.array([[2.0]], dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, "single element")
        assert info["pass"], info.get("error")

    def test_large_values(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=dtype)
        b = np.array([[5e10, 6e10], [7e10, 8e10]], dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, label="large values")
        assert info["pass"], info.get("error")

    def test_negative(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=dtype)
        b = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, "negative values")
        assert info["pass"], info.get("error")

    def test_zeros(self, cpp, rtol, atol, dtype):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = np.zeros((3, 2), dtype=dtype)
        b = np.ones((3, 2), dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, "zeros input")
        assert info["pass"], info.get("error")


class TestEinsumLargeGateMachine:
    """Realistic gate_machine sizes: [N-1, dims] where N ~ 100."""

    @pytest.mark.parametrize("n,dims", [(10, 2), (50, 3), (100, 2), (200, 3)])
    def test_ij_ij_to_i(self, cpp, rtol, atol, dtype, n, dims):
        _rtol, _atol = tolerance_for(dtype, rtol, atol)
        a = random_array((n, dims), seed=1, dtype=dtype)
        b = random_array((n, dims), seed=2, dtype=dtype)
        info = compare(cpp.einsum("ij,ij->i", a, b), np.einsum("ij,ij->i", a, b), _rtol, _atol, f"gate_machine [{n},{dims}]")
        assert info["pass"], info.get("error")
