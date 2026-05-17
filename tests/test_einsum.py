#!/usr/bin/env python3
"""
Test numpy.einsum C++ implementation against Python numpy.einsum.

Each test case:
1. Generates random input arrays
2. Runs np.einsum(subscripts, a, b) in Python
3. Runs melange_reward.einsum(subscripts, a, b) in C++
4. Compares results with strict tolerance

Usage:
    source .venv/bin/activate
    python tpp_onemodel_cpp/numpy/tests/test_einsum.py
"""

import numpy as np
import sys
import os

# Add melange root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import melange_reward
except ImportError:
    print("ERROR: melange_reward module not found. Build it first: bash build.sh")
    sys.exit(1)


def assert_allclose(cpp_result, py_result, rtol=1e-12, atol=1e-12, label=""):
    """Compare C++ and Python results with tolerance."""
    cpp = np.asarray(cpp_result, dtype=np.float64)
    py = np.asarray(py_result, dtype=np.float64)

    if cpp.shape != py.shape:
        raise AssertionError(
            f"{label}: shape mismatch: C++ {cpp.shape} vs Python {py.shape}"
        )

    if not np.allclose(cpp, py, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(cpp - py))
        raise AssertionError(
            f"{label}: numerical mismatch (max diff={max_diff:.2e})\n"
            f"  C++ first 5: {cpp.flatten()[:5]}\n"
            f"  Py  first 5: {py.flatten()[:5]}"
        )


def test_pattern_ij_ij_to_i():
    """Pattern: 'ij,ij->i' — row-wise dot product (gate_machine_reward.py ×3)"""
    print("Testing 'ij,ij->i' ...", end=" ")

    # Small fixed test
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64)
    py_r = np.einsum("ij,ij->i", a, b)
    cpp_r = melange_reward.einsum("ij,ij->i", a, b)
    assert_allclose(cpp_r, py_r, label="ij,ij->i fixed")

    # Random tests with various shapes
    for m in [1, 3, 10, 100]:
        for n in [1, 2, 3, 5]:
            a = np.random.randn(m, n).astype(np.float64)
            b = np.random.randn(m, n).astype(np.float64)
            py_r = np.einsum("ij,ij->i", a, b)
            cpp_r = melange_reward.einsum("ij,ij->i", a, b)
            assert_allclose(cpp_r, py_r, label=f"ij,ij->i [{m},{n}]")

    print("OK")


def test_pattern_ij_jk_to_ik():
    """Pattern: 'ij,jk->ik' — matrix multiplication"""
    print("Testing 'ij,jk->ik' ...", end=" ")

    # Fixed test
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64)
    py_r = np.einsum("ij,jk->ik", a, b)
    cpp_r = melange_reward.einsum("ij,jk->ik", a, b)
    assert_allclose(cpp_r, py_r, label="ij,jk->ik fixed")

    # Random tests
    for i_sz in [1, 3, 10]:
        for j_sz in [1, 3, 10]:
            for k_sz in [1, 3, 10]:
                a = np.random.randn(i_sz, j_sz).astype(np.float64)
                b = np.random.randn(j_sz, k_sz).astype(np.float64)
                py_r = np.einsum("ij,jk->ik", a, b)
                cpp_r = melange_reward.einsum("ij,jk->ik", a, b)
                assert_allclose(cpp_r, py_r, label=f"ij,jk->ik [{i_sz},{j_sz},{k_sz}]")

    # Compare with numpy.matmul
    a = np.random.randn(4, 5).astype(np.float64)
    b = np.random.randn(5, 3).astype(np.float64)
    cpp_r = melange_reward.einsum("ij,jk->ik", a, b)
    np_r = a @ b
    assert_allclose(cpp_r, np_r, label="ij,jk->ik vs matmul")

    print("OK")


def test_pattern_bij_bjk_to_bik():
    """Pattern: 'bij,bjk->bik' — batch matrix multiplication (astra_planner_transform.py)"""
    print("Testing 'bij,bjk->bik' ...", end=" ")

    # Fixed test
    a = np.array([[[1.0, 2.0], [3.0, 4.0]],
                  [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    b = np.array([[[0.5, 1.5], [2.5, 3.5]],
                  [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float64)
    py_r = np.einsum("bij,bjk->bik", a, b)
    cpp_r = melange_reward.einsum("bij,bjk->bik", a, b)
    assert_allclose(cpp_r, py_r, label="bij,bjk->bik fixed")

    # Random tests
    for b_sz in [1, 3, 5]:
        for i_sz in [1, 3, 5]:
            for j_sz in [1, 3, 5]:
                for k_sz in [1, 3, 5]:
                    a = np.random.randn(b_sz, i_sz, j_sz).astype(np.float64)
                    b_ = np.random.randn(b_sz, j_sz, k_sz).astype(np.float64)
                    py_r = np.einsum("bij,bjk->bik", a, b_)
                    cpp_r = melange_reward.einsum("bij,bjk->bik", a, b_)
                    assert_allclose(cpp_r, py_r, label=f"bij,bjk->bik [{b_sz},{i_sz},{j_sz},{k_sz}]")

    # Compare with batch matmul
    a = np.random.randn(8, 4, 6).astype(np.float64)
    b_ = np.random.randn(8, 6, 3).astype(np.float64)
    cpp_r = melange_reward.einsum("bij,bjk->bik", a, b_)
    np_r = a @ b_
    assert_allclose(cpp_r, np_r, label="bij,bjk->bik vs batch matmul")

    print("OK")


def test_pattern_aij_aij_to_ai():
    """Pattern: 'aij,aij->ai' — batch row-wise dot product (rotation_bboxes.py)"""
    print("Testing 'aij,aij->ai' ...", end=" ")

    for a_sz in [1, 3, 5]:
        for i_sz in [1, 3, 10]:
            for j_sz in [1, 2, 5]:
                a = np.random.randn(a_sz, i_sz, j_sz).astype(np.float64)
                b = np.random.randn(a_sz, i_sz, j_sz).astype(np.float64)
                py_r = np.einsum("aij,aij->ai", a, b)
                cpp_r = melange_reward.einsum("aij,aij->ai", a, b)
                assert_allclose(cpp_r, py_r, label=f"aij,aij->ai [{a_sz},{i_sz},{j_sz}]")

    print("OK")


def test_pattern_implicit_mode():
    """Pattern: 'ij,ij' (implicit mode, no '->') — should compute 'i' output"""
    print("Testing implicit mode 'ij,ij' ...", end=" ")

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)

    # np.einsum('ij,ij') should produce [1*0.5+2*1.5, 3*2.5+4*3.5] = [3.5, 21.5]
    py_r = np.einsum("ij,ij", a, b)
    cpp_r = melange_reward.einsum("ij,ij", a, b)
    assert_allclose(cpp_r, py_r, label="ij,ij implicit")

    # Also test with different labels
    for m, n in [(3, 2), (5, 4), (1, 1)]:
        a = np.random.randn(m, n).astype(np.float64)
        b = np.random.randn(m, n).astype(np.float64)
        py_r = np.einsum("ij,ij", a, b)
        cpp_r = melange_reward.einsum("ij,ij", a, b)
        assert_allclose(cpp_r, py_r, label=f"ij,ij implicit [{m},{n}]")

    print("OK")


def test_pattern_implicit_matmul():
    """Pattern: 'ij,jk' (implicit matmul)"""
    print("Testing implicit mode 'ij,jk' ...", end=" ")

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64)
    py_r = np.einsum("ij,jk", a, b)
    cpp_r = melange_reward.einsum("ij,jk", a, b)
    assert_allclose(cpp_r, py_r, label="ij,jk implicit")

    print("OK")


def test_pattern_broadcast_matmul():
    """Pattern: 'ijk,mkl->mjl' — broadcast matmul (auto_regressive_transform.py)"""
    print("Testing 'ijk,mkl->mjl' ...", end=" ")

    # sum over i and k: result[m,j,l] = sum_i sum_k A[i,j,k] * B[m,k,l]
    for i_sz in [1, 2]:
        for j_sz in [1, 3]:
            for k_sz in [1, 2]:
                for m_sz in [1, 3]:
                    for l_sz in [1, 2]:
                        a = np.random.randn(i_sz, j_sz, k_sz).astype(np.float64)
                        b = np.random.randn(m_sz, k_sz, l_sz).astype(np.float64)
                        py_r = np.einsum("ijk,mkl->mjl", a, b)
                        cpp_r = melange_reward.einsum("ijk,mkl->mjl", a, b)
                        assert_allclose(cpp_r, py_r,
                                        label=f"ijk,mkl->mjl [{i_sz},{j_sz},{k_sz},{m_sz},{l_sz}]")

    print("OK")


def test_pattern_nij_nmj_to_nmi():
    """Pattern: 'nij,nmj->nmi' — batched matmul with double transpose (coords_transform.py)"""
    print("Testing 'nij,nmj->nmi' ...", end=" ")

    for n_sz in [1, 3, 5]:
        for i_sz in [1, 3]:
            for j_sz in [1, 3]:
                for m_sz in [1, 3]:
                    # nij: A[n,i,j], nmj: B[n,m,j] → nmi: C[n,m,i] = sum_j A[n,i,j]*B[n,m,j]
                    a = np.random.randn(n_sz, i_sz, j_sz).astype(np.float64)
                    b = np.random.randn(n_sz, m_sz, j_sz).astype(np.float64)
                    py_r = np.einsum("nij,nmj->nmi", a, b)
                    cpp_r = melange_reward.einsum("nij,nmj->nmi", a, b)
                    assert_allclose(cpp_r, py_r,
                                    label=f"nij,nmj->nmi [{n_sz},{i_sz},{j_sz},{m_sz}]")

    print("OK")


def test_pattern_aij_jka_to_aik():
    """Pattern: 'aij,jka->aik' — batch matmul with transpose (rotation_bboxes.py)"""
    print("Testing 'aij,jka->aik' ...", end=" ")

    # A[a,i,j], B[j,k,a] → C[a,i,k] = sum_j A[a,i,j] * B[j,k,a]
    for a_sz in [1, 3, 5]:
        for i_sz in [1, 3]:
            for j_sz in [1, 3]:
                for k_sz in [1, 3]:
                    a = np.random.randn(a_sz, i_sz, j_sz).astype(np.float64)
                    b = np.random.randn(j_sz, k_sz, a_sz).astype(np.float64)
                    py_r = np.einsum("aij,jka->aik", a, b)
                    cpp_r = melange_reward.einsum("aij,jka->aik", a, b)
                    assert_allclose(cpp_r, py_r,
                                    label=f"aij,jka->aik [{a_sz},{i_sz},{j_sz},{k_sz}]")

    print("OK")


def test_pattern_large_gate_machine():
    """Simulate gate_machine_reward.py einsum usage with realistic shapes."""
    print("Testing gate_machine pattern 'ij,ij->i' (realistic) ...", end=" ")

    # gate_machine uses AB of shape [N-1, 2] or [N-1, 3] where N ~ 100
    for n in [10, 50, 100, 200]:
        for dims in [2, 3]:
            a = np.random.randn(n, dims).astype(np.float64)
            b = np.random.randn(n, dims).astype(np.float64)
            py_r = np.einsum("ij,ij->i", a, b)
            cpp_r = melange_reward.einsum("ij,ij->i", a, b)
            assert_allclose(cpp_r, py_r, label=f"gate_machine [{n},{dims}]")

    print("OK")


def test_edge_cases():
    """Test edge cases: single-element arrays, large values."""
    print("Testing edge cases ...", end=" ")

    # Single element
    a = np.array([[1.0]], dtype=np.float64)
    b = np.array([[2.0]], dtype=np.float64)
    py_r = np.einsum("ij,ij->i", a, b)
    cpp_r = melange_reward.einsum("ij,ij->i", a, b)
    assert_allclose(cpp_r, py_r, label="single element")

    # Large values
    a = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=np.float64)
    b = np.array([[5e10, 6e10], [7e10, 8e10]], dtype=np.float64)
    py_r = np.einsum("ij,ij->i", a, b)
    cpp_r = melange_reward.einsum("ij,ij->i", a, b)
    assert_allclose(cpp_r, py_r, label="large values", rtol=1e-10)

    # Negative values
    a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    b = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float64)
    py_r = np.einsum("ij,ij->i", a, b)
    cpp_r = melange_reward.einsum("ij,ij->i", a, b)
    assert_allclose(cpp_r, py_r, label="negative values")

    # Zeros
    a = np.zeros((3, 2), dtype=np.float64)
    b = np.ones((3, 2), dtype=np.float64)
    py_r = np.einsum("ij,ij->i", a, b)
    cpp_r = melange_reward.einsum("ij,ij->i", a, b)
    assert_allclose(cpp_r, py_r, label="zeros")

    print("OK")


def test_1d_output():
    """Test patterns that produce 1D output."""
    print("Testing 1D output patterns ...", end=" ")

    # ji,ji->j (same as ij,ij->i but with different label)
    a = np.random.randn(5, 3).astype(np.float64)
    b = np.random.randn(5, 3).astype(np.float64)
    py_r = np.einsum("ji,ji->j", a, b)
    cpp_r = melange_reward.einsum("ji,ji->j", a, b)
    assert_allclose(cpp_r, py_r, label="ji,ji->j")

    print("OK")


def main():
    print("=" * 60)
    print("numpy.einsum C++ vs Python Comparison Tests")
    print("=" * 60)

    tests = [
        test_pattern_ij_ij_to_i,
        test_pattern_ij_jk_to_ik,
        test_pattern_bij_bjk_to_bik,
        test_pattern_aij_aij_to_ai,
        test_pattern_implicit_mode,
        test_pattern_implicit_matmul,
        test_pattern_broadcast_matmul,
        test_pattern_nij_nmj_to_nmi,
        test_pattern_aij_jka_to_aik,
        test_pattern_large_gate_machine,
        test_edge_cases,
        test_1d_output,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
