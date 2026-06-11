// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/linalg.h                            [PUBLIC HEADER]
//
//  Linear algebra and einsum.
//
//  numpy.dot          numpy.linalg.norm (scalar + axis)
//  numpy.linalg.inv   numpy.linalg.matmul  (2-D, 1-D×2-D, 2-D×1-D, batched 3-D)
//  numpy.einsum       (2-operand, explicit + implicit mode)
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <immintrin.h>   // SSE/AVX intrinsics (SSE2 is baseline on x86_64)

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Internal detail headers ──────────────────────────────────────────────────
// Backend selected at compile time:
//   NUMPYCPP_STD_ONLY not defined (default):
//     blas_bridge.h — bit-exact (OpenBLAS ILP64, dlsym)
//   NUMPYCPP_STD_ONLY defined:
//     std_linalg_backend.h — pure C++ loops, performance-first
#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  define NUMPYCPP_INTERNAL_INCLUDE
#  define _NUMPYCPP_LINALG_OWNS_GUARD
#endif
#ifdef NUMPYCPP_STD_ONLY
#  include "detail/std_linalg_backend.h"
#else
#  include "detail/blas_bridge.h"
#endif
#ifdef _NUMPYCPP_LINALG_OWNS_GUARD
#  undef NUMPYCPP_INTERNAL_INCLUDE
#  undef _NUMPYCPP_LINALG_OWNS_GUARD
#endif

// reduce.h supplies pairwise_sum (needed by norm_sq) and norm_axis
#include "reduce.h"

#include "detail/macros.h"   // NUMPY_SMALL_STACK

namespace numpy {

// ============================================================================
// Squared L2 norm / flat dot helper (used by linalg::norm + linalg::matmul)
// ============================================================================

/// Squared L2 norm — stack allocation for n ≤ NUMPY_SMALL_STACK
template<typename T>
inline T norm_sq(const T* data, size_t n) {
    T buf[NUMPY_SMALL_STACK];
    T* squares = (n <= NUMPY_SMALL_STACK) ? buf : new T[n];
    for (size_t i = 0; i < n; ++i) squares[i] = data[i] * data[i];
    T result = pairwise_sum(squares, n);
    if (n > NUMPY_SMALL_STACK) delete[] squares;
    return result;
}

/// numpy.dot(a, b, out=None)
/// Routes through OpenBLAS sdot_64_/ddot_64_ for bit-exact match with np.dot.
template<typename T>
inline T dot(const T* a, const T* b, size_t n) {
    return detail::blas_ops<T>::dot(a, b, n);
}

// ============================================================================
// numpy.linalg
// ============================================================================

namespace linalg {

/// numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) — vector / Frobenius
template<typename T>
inline T norm(const T* data, size_t n) {
    return numpy::detail::blas_ops<T>::norm(data, n);
}

/// numpy.linalg.norm(x, ord=None, axis=N, keepdims=False) — N-D
template<typename T>
inline void norm_axis(const T* src, T* dst,
                      const ptrdiff_t* shape, int ndim, int axis) {
    numpy::norm_axis(src, dst, shape, ndim, axis);
}

/// numpy.linalg.inv(a) — matrix inverse (square N×N)
/// Uses LAPACKE getrf+getri (bitexact) or Gauss-Jordan (std backend).
/// Returns true on success; false if matrix is singular or LAPACK unavailable.
template<typename T>
inline bool inv(const T* A, T* A_inv, size_t N) {
    // Copy input to output buffer (inv modifies in-place)
    for (size_t i = 0; i < N * N; ++i) A_inv[i] = A[i];
    bool ok = numpy::detail::blas_ops<T>::inv(A_inv, N);
    if (ok) {
        // Normalise -0.0 → +0.0 (LAPACK build variance in signed-zero output)
        for (size_t i = 0; i < N * N; ++i)
            if (A_inv[i] == T(0)) A_inv[i] = T(0);
    }
    return ok;
}

/// numpy.matmul — dispatch helper (mirrors numpy's cblas_matrixproduct)
/// M==1&&N==1 → sdot   M==1 → gemv(Trans)   N==1 → gemv(NoTrans)   else → gemm
template<typename T>
inline void matmul_slice(const T* A, const T* B, T* C,
                          size_t M, size_t K, size_t N) {
    if (M == 1 && N == 1) {
        C[0] = numpy::detail::blas_ops<T>::dot(A, B, K);
    } else if (M == 1) {
        numpy::detail::blas_ops<T>::gemvt(B, A, C, K, N);
    } else if (N == 1) {
        numpy::detail::blas_ops<T>::gemv(A, B, C, M, K);
    } else {
        numpy::detail::blas_ops<T>::gemm(A, B, C, M, K, N);
    }
}

/// numpy.matmul — 2D: C[M,N] = A[M,K] @ B[K,N]
template<typename T>
inline void matmul(const T* A, const T* B, T* C,
                   size_t M, size_t K, size_t N) {
    matmul_slice<T>(A, B, C, M, K, N);
}

/// numpy.matmul — 2D×1D: y[M] = A[M,K] @ x[K]
template<typename T>
inline void matmul_mv(const T* A, const T* x, T* y, size_t M, size_t K) {
    numpy::detail::blas_ops<T>::gemv(A, x, y, M, K);
}

/// numpy.matmul — 1D×2D: y[N] = a[K] @ B[K,N]  (= B^T @ a)
/// When N==1, numpy uses sdot (same dot path), not sgemv.
template<typename T>
inline void matmul_vm(const T* a, const T* B, T* y, size_t K, size_t N) {
    if (N == 1)
        y[0] = numpy::detail::blas_ops<T>::dot(a, B, K);
    else
        numpy::detail::blas_ops<T>::gemvt(B, a, y, K, N);
}

/// numpy.matmul — batched 3D: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N]
template<typename T>
inline void matmul(const T* A, const T* B, T* C,
                   size_t batch, size_t M, size_t K, size_t N) {
    for (size_t b = 0; b < batch; ++b)
        matmul_slice<T>(A + b * M * K, B + b * K * N, C + b * M * N, M, K, N);
}

} // namespace linalg

// ============================================================================
// numpy.einsum  (2-operand, explicit + implicit mode)
// ============================================================================

namespace einsum_detail {

// ── Subscript parsing ────────────────────────────────────────────────────────
struct ParsedSubscripts {
    std::vector<std::string> inputs;
    std::string output;
    bool explicit_mode;
};

inline ParsedSubscripts parse_subscripts(const std::string& subscripts) {
    ParsedSubscripts result{};
    auto arrow = subscripts.find("->");
    std::string lhs;
    if (arrow != std::string::npos) {
        lhs = subscripts.substr(0, arrow);
        result.output = subscripts.substr(arrow + 2);
        result.explicit_mode = true;
    } else {
        lhs = subscripts;
    }
    size_t start = 0;
    while (start <= lhs.size()) {
        auto comma = lhs.find(',', start);
        if (comma == std::string::npos) {
            result.inputs.push_back(lhs.substr(start));
            break;
        }
        result.inputs.push_back(lhs.substr(start, comma - start));
        start = comma + 1;
    }
    return result;
}

inline std::vector<ptrdiff_t> compute_strides(
        const std::vector<ptrdiff_t>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<ptrdiff_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

inline ptrdiff_t flat_index(const ptrdiff_t* coord,
                              const ptrdiff_t* strides, int ndim) {
    ptrdiff_t idx = 0;
    for (int i = 0; i < ndim; ++i) idx += coord[i] * strides[i];
    return idx;
}

// ── SIMD sum-of-products kernels (SSE, matches numpy's exact path) ───────────
#define EINSUM_PREFETCH_F64 (16 * 2)
#define EINSUM_PREFETCH_F32 (16 * 4)

inline double einsum_reduce_f64(const double* a, const double* b, size_t n) {
    __m128d v_accum = _mm_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm_prefetch((const char*)(a + i + EINSUM_PREFETCH_F64), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + i + EINSUM_PREFETCH_F64), _MM_HINT_T0);
        __m128d a3 = _mm_loadu_pd(a + i + 6), b3 = _mm_loadu_pd(b + i + 6);
        __m128d a2 = _mm_loadu_pd(a + i + 4), b2 = _mm_loadu_pd(b + i + 4);
        __m128d a1 = _mm_loadu_pd(a + i + 2), b1 = _mm_loadu_pd(b + i + 2);
        __m128d a0 = _mm_loadu_pd(a + i),     b0 = _mm_loadu_pd(b + i);
        v_accum = _mm_add_pd(v_accum, _mm_mul_pd(a3, b3));
        v_accum = _mm_add_pd(v_accum, _mm_mul_pd(a2, b2));
        __m128d t1 = _mm_add_pd(v_accum, _mm_mul_pd(a1, b1));
        v_accum    = _mm_add_pd(_mm_mul_pd(a0, b0), t1);
    }
    size_t remaining = n - i;
    while (remaining > 0) {
        __m128d va, vb;
        if (remaining >= 2) {
            va = _mm_loadu_pd(a + i); vb = _mm_loadu_pd(b + i);
            i += 2; remaining -= 2;
        } else {
            va = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(a + i)));
            vb = _mm_castsi128_pd(_mm_loadl_epi64((const __m128i*)(b + i)));
            i += 1; remaining = 0;
        }
        v_accum = _mm_add_pd(v_accum, _mm_mul_pd(va, vb));
    }
    return _mm_cvtsd_f64(_mm_hadd_pd(v_accum, v_accum));
}

inline float einsum_reduce_f32(const float* a, const float* b, size_t n) {
    __m128 v_accum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_prefetch((const char*)(a + i + EINSUM_PREFETCH_F32), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + i + EINSUM_PREFETCH_F32), _MM_HINT_T0);
        __m128 a3 = _mm_loadu_ps(a + i + 12), b3 = _mm_loadu_ps(b + i + 12);
        __m128 a2 = _mm_loadu_ps(a + i +  8), b2 = _mm_loadu_ps(b + i +  8);
        __m128 a1 = _mm_loadu_ps(a + i +  4), b1 = _mm_loadu_ps(b + i +  4);
        __m128 a0 = _mm_loadu_ps(a + i),      b0 = _mm_loadu_ps(b + i);
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(a3, b3));
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(a2, b2));
        __m128 t1 = _mm_add_ps(v_accum, _mm_mul_ps(a1, b1));
        v_accum   = _mm_add_ps(_mm_mul_ps(a0, b0), t1);
    }
    size_t remaining = n - i;
    while (remaining > 0) {
        __m128 va, vb;
        if (remaining >= 4) {
            va = _mm_loadu_ps(a + i); vb = _mm_loadu_ps(b + i);
            i += 4; remaining -= 4;
        } else if (remaining == 3) {
            __m128i ta = _mm_loadl_epi64((const __m128i*)(a + i));
            __m128i tb = _mm_loadl_epi64((const __m128i*)(b + i));
            ta = _mm_insert_epi32(ta, ((const int*)(a + i))[2], 2);
            tb = _mm_insert_epi32(tb, ((const int*)(b + i))[2], 2);
            va = _mm_castsi128_ps(ta); vb = _mm_castsi128_ps(tb);
            i += 3; remaining = 0;
        } else if (remaining == 2) {
            va = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(a + i)));
            vb = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(b + i)));
            i += 2; remaining = 0;
        } else {
            va = _mm_castsi128_ps(_mm_cvtsi32_si128(((const int*)(a + i))[0]));
            vb = _mm_castsi128_ps(_mm_cvtsi32_si128(((const int*)(b + i))[0]));
            i += 1; remaining = 0;
        }
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(va, vb));
    }
    __m128 sum_halves = _mm_hadd_ps(v_accum, v_accum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
}

inline std::string implicit_output_labels(
        const std::vector<std::string>& il) {
    std::map<char, int> cnt;
    for (const auto& s : il) {
        std::set<char> seen;
        for (char c : s)
            if (seen.insert(c).second) cnt[c]++;
    }
    std::string out;
    for (const auto& [c, n] : cnt)
        if (n == 1) out += c;
    std::sort(out.begin(), out.end());
    return out;
}

// ── Main computation ─────────────────────────────────────────────────────────

/// numpy.einsum(subscripts, *operands, ...) — 2-operand only
template<typename T>
void einsum(const std::string& subscripts,
            const T* a_ptr, const ptrdiff_t* a_shape, int a_ndim,
            const T* b_ptr, const ptrdiff_t* b_shape, int b_ndim,
            T* result_ptr) {
    using namespace std;

    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    string out_labels = parsed.output;

    if (in_labels.size() != 2)
        throw invalid_argument("einsum: currently only supports 2 operands, got " +
                               to_string(in_labels.size()));

    if (!parsed.explicit_mode)
        out_labels = implicit_output_labels(in_labels);

    vector<ptrdiff_t> shapes[2] = {
        vector<ptrdiff_t>(a_shape, a_shape + a_ndim),
        vector<ptrdiff_t>(b_shape, b_shape + b_ndim)
    };
    int ndim[2] = {a_ndim, b_ndim};

    for (int i = 0; i < 2; ++i)
        if (static_cast<int>(in_labels[i].size()) != ndim[i])
            throw invalid_argument(
                "einsum: subscript '" + in_labels[i] + "' has " +
                to_string(in_labels[i].size()) + " labels but operand " +
                to_string(i) + " has " + to_string(ndim[i]) + " dimensions");

    map<char, vector<pair<int, int>>> label_axis;
    map<char, ptrdiff_t> label_size;
    for (int i = 0; i < 2; ++i)
        for (int ax = 0; ax < ndim[i]; ++ax) {
            char l = in_labels[i][ax];
            label_axis[l].push_back({i, ax});
            ptrdiff_t sz = shapes[i][ax];
            auto it = label_size.find(l);
            if (it == label_size.end()) label_size[l] = sz;
            else if (it->second != sz)
                throw invalid_argument(
                    "einsum: label '" + string(1, l) + "' has inconsistent sizes");
        }

    set<char> out_set(out_labels.begin(), out_labels.end());
    set<char> all_labels;
    for (const auto& s : in_labels)
        for (char c : s) all_labels.insert(c);

    vector<char> sum_labels;
    for (char c : all_labels)
        if (!out_set.count(c)) sum_labels.push_back(c);

    vector<char> output_labels(out_labels.begin(), out_labels.end());

    for (char c : output_labels)
        if (!all_labels.count(c))
            throw invalid_argument(
                "einsum: output label '" + string(1, c) + "' not found in any input");

    vector<ptrdiff_t> output_shape;
    for (char c : output_labels)
        output_shape.push_back(label_size[c]);

    // ── Fast path: single contraction label, last axis in both operands ──────
    if (sum_labels.size() == 1) {
        char clabel = sum_labels[0];
        ptrdiff_t csize = label_size[clabel];
        int a_caxis = -1, b_caxis = -1;
        for (const auto& [inp, ax] : label_axis[clabel]) {
            if (inp == 0) a_caxis = ax;
            if (inp == 1) b_caxis = ax;
        }
        if (a_caxis == ndim[0] - 1 && b_caxis == ndim[1] - 1 && csize > 0) {
            auto a_str = compute_strides(shapes[0]);
            auto b_str = compute_strides(shapes[1]);
            int n_out = static_cast<int>(output_labels.size());

            vector<ptrdiff_t> stepA(n_out), stepB(n_out), ostride(n_out);
            for (int d = 0; d < n_out; ++d) {
                char c = output_labels[d];
                int aa = -1, ba = -1;
                for (const auto& [inp, ax] : label_axis[c]) {
                    if (inp == 0) aa = ax; else ba = ax;
                }
                stepA[d] = (aa >= 0) ? a_str[aa] : 0;
                stepB[d] = (ba >= 0) ? b_str[ba] : 0;
            }
            { ptrdiff_t acc = 1;
              for (int d = n_out - 1; d >= 0; --d) { ostride[d] = acc; acc *= output_shape[d]; } }

            ptrdiff_t output_total = 1;
            for (ptrdiff_t s : output_shape) output_total *= s;
            const size_t csize_u = static_cast<size_t>(csize);

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(output_total > 256)
            #endif
            for (ptrdiff_t oi = 0; oi < output_total; ++oi) {
                ptrdiff_t rem = oi, a_off = 0, b_off = 0;
                for (int d = 0; d < n_out; ++d) {
                    ptrdiff_t coord = rem / ostride[d];
                    rem -= coord * ostride[d];
                    a_off += coord * stepA[d];
                    b_off += coord * stepB[d];
                }
                if constexpr (std::is_same_v<T, double>)
                    result_ptr[oi] = einsum_reduce_f64(a_ptr + a_off, b_ptr + b_off, csize_u);
                else
                    result_ptr[oi] = einsum_reduce_f32(a_ptr + a_off, b_ptr + b_off, csize_u);
            }
            return;
        }
    }

    // ── Scalar path: general case ────────────────────────────────────────────
    vector<char> iter_labels = output_labels;
    iter_labels.insert(iter_labels.end(), sum_labels.begin(), sum_labels.end());
    int n_iter   = static_cast<int>(iter_labels.size());
    int n_output = static_cast<int>(output_labels.size());

    vector<ptrdiff_t> iter_sizes(n_iter);
    for (int i = 0; i < n_iter; ++i)
        iter_sizes[i] = label_size[iter_labels[i]];

    auto strides_a = compute_strides(shapes[0]);
    auto strides_b = compute_strides(shapes[1]);

    vector<int> iax_flat(n_iter * 2, -1);
    for (int li = 0; li < n_iter; ++li) {
        char l = iter_labels[li];
        for (const auto& [inp, ax] : label_axis[l])
            iax_flat[li * 2 + inp] = ax;
    }
    int (*iax)[2] = reinterpret_cast<int(*)[2]>(iax_flat.data());

    vector<ptrdiff_t> iter_coord(n_iter, 0);
    vector<ptrdiff_t> ic0(ndim[0], 0), ic1(ndim[1], 0);
    ptrdiff_t* ic[2] = {ic0.data(), ic1.data()};

    ptrdiff_t iter_total = 1;
    for (ptrdiff_t s : iter_sizes) iter_total *= s;

    ptrdiff_t current_output_idx = -1;
    T accumulator = T(0);

    for (ptrdiff_t flat = 0; flat < iter_total; ++flat) {
        if (flat > 0)
            for (int d = n_iter - 1; d >= 0; --d)
                if (++iter_coord[d] < iter_sizes[d]) break;
                else iter_coord[d] = 0;

        bool is_new_output = true;
        for (int i = n_output; i < n_iter; ++i)
            if (iter_coord[i] != 0) { is_new_output = false; break; }

        if (is_new_output) {
            if (current_output_idx >= 0)
                result_ptr[current_output_idx] = accumulator;
            ptrdiff_t out_idx = 0, out_stride = 1;
            for (int i = n_output - 1; i >= 0; --i) {
                out_idx += iter_coord[i] * out_stride;
                if (i > 0) out_stride *= iter_sizes[i];
            }
            current_output_idx = out_idx;
            accumulator = T(0);
        }

        memset(ic[0], 0, static_cast<size_t>(ndim[0]) * sizeof(ptrdiff_t));
        memset(ic[1], 0, static_cast<size_t>(ndim[1]) * sizeof(ptrdiff_t));
        for (int li = 0; li < n_iter; ++li)
            for (int inp = 0; inp < 2; ++inp) {
                int ax = iax[li][inp];
                if (ax >= 0) ic[inp][ax] = iter_coord[li];
            }

        accumulator += a_ptr[flat_index(ic[0], strides_a.data(), ndim[0])]
                     * b_ptr[flat_index(ic[1], strides_b.data(), ndim[1])];
    }
    if (current_output_idx >= 0)
        result_ptr[current_output_idx] = accumulator;
}

/// Compute output shape for numpy.einsum
inline std::vector<ptrdiff_t> einsum_output_shape(
        const std::string& subscripts,
        const ptrdiff_t* a_shape, int a_ndim,
        const ptrdiff_t* b_shape, int b_ndim) {
    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    std::string out_labels = parsed.output;
    if (!parsed.explicit_mode)
        out_labels = implicit_output_labels(in_labels);

    std::map<char, ptrdiff_t> label_size;
    for (int i = 0; i < a_ndim; ++i) label_size[in_labels[0][i]] = a_shape[i];
    for (int i = 0; i < b_ndim; ++i) label_size[in_labels[1][i]] = b_shape[i];

    std::vector<ptrdiff_t> output_shape;
    for (char c : out_labels)
        output_shape.push_back(label_size[c]);
    return output_shape;
}

} // namespace einsum_detail

} // namespace numpy
