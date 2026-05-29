// Native C++ einsum implementation — zero pybind11 dependency.
// Supports 2-operand explicit and implicit mode summation.

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <type_traits>

// immintrin.h provides all SSE/AVX intrinsics needed by
// einsum_reduce_f32/f64 (mulps, addps, haddps, etc.).
#include <immintrin.h>

namespace numpy {
namespace einsum_detail {

// ============================================================================
// Subscript parsing
// ============================================================================
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

inline std::vector<ptrdiff_t> compute_strides(const std::vector<ptrdiff_t>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<ptrdiff_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

inline ptrdiff_t flat_index(const std::vector<ptrdiff_t>& coord,
                              const std::vector<ptrdiff_t>& strides) {
    ptrdiff_t idx = 0;
    for (size_t i = 0; i < coord.size(); ++i)
        idx += coord[i] * strides[i];
    return idx;
}

// ============================================================================
// SIMD sum-of-products reduction — matches numpy's
// *_sum_of_products_contig_contig_outstride0_two exactly.
//
// numpy's multiarray module uses SSE baseline SIMD with SEPARATE
// mul+add (NOT FMA). Despite the CPU having AVX512 and FMA3, the
// einsum kernel in the baseline multiarray module is compiled for SSE.
//
// Key observations from disassembling numpy's .so:
//   - Uses xmm registers (SSE, 2-wide f64 / 4-wide f32)
//   - mulpd/mulps + addpd/addps (NOT vfmadd)
//   - haddpd/haddps for horizontal sum
//   - Forward accumulation chain (NOT reverse-FMA)
// ============================================================================

inline double einsum_reduce_f64(const double* a, const double* b, size_t n) {
    __m128d v_accum = _mm_setzero_pd();
    const int vstep = 2;
    size_t i = 0;
    const size_t vstepx4 = vstep * 4;  // 8

    // 4x unrolled block — forward mul+add chain, exactly as numpy SSE
    for (; i + vstepx4 <= n; i += vstepx4) {
        __m128d a3 = _mm_loadu_pd(a + i + 6);
        __m128d b3 = _mm_loadu_pd(b + i + 6);
        __m128d a2 = _mm_loadu_pd(a + i + 4);
        __m128d b2 = _mm_loadu_pd(b + i + 4);
        __m128d a1 = _mm_loadu_pd(a + i + 2);
        __m128d b1 = _mm_loadu_pd(b + i + 2);
        __m128d a0 = _mm_loadu_pd(a + i);
        __m128d b0 = _mm_loadu_pd(b + i);

        // numpy's exact forward chain:
        //   accum += a3*b3  (mul+add)
        //   accum += a2*b2
        //   temp = accum + a1*b1
        //   accum = a0*b0 + temp
        v_accum = _mm_add_pd(v_accum, _mm_mul_pd(a3, b3));
        v_accum = _mm_add_pd(v_accum, _mm_mul_pd(a2, b2));
        __m128d t1 = _mm_add_pd(v_accum, _mm_mul_pd(a1, b1));
        v_accum    = _mm_add_pd(_mm_mul_pd(a0, b0), t1);
    }

    // Tail loop: numpy-style load_tillz_f64
    size_t remaining = n - i;
    while (remaining > 0) {
        __m128d va, vb;
        if (remaining >= 2) {
            va = _mm_loadu_pd(a + i);
            vb = _mm_loadu_pd(b + i);
            i += 2; remaining -= 2;
        } else {
            // remaining == 1: use movq (same as _mm_loadl_epi64)
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
    const int vstep = 4;
    size_t i = 0;
    const size_t vstepx4 = vstep * 4;  // 16

    // 4x unrolled block — forward mul+add chain, exactly as numpy SSE
    for (; i + vstepx4 <= n; i += vstepx4) {
        __m128 a3 = _mm_loadu_ps(a + i + 12);
        __m128 b3 = _mm_loadu_ps(b + i + 12);
        __m128 a2 = _mm_loadu_ps(a + i + 8);
        __m128 b2 = _mm_loadu_ps(b + i + 8);
        __m128 a1 = _mm_loadu_ps(a + i + 4);
        __m128 b1 = _mm_loadu_ps(b + i + 4);
        __m128 a0 = _mm_loadu_ps(a + i);
        __m128 b0 = _mm_loadu_ps(b + i);

        // numpy's exact forward chain:
        //   accum += a3*b3  (mul+add)
        //   accum += a2*b2
        //   temp = accum + a1*b1
        //   accum = a0*b0 + temp
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(a3, b3));
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(a2, b2));
        __m128 t1 = _mm_add_ps(v_accum, _mm_mul_ps(a1, b1));
        v_accum    = _mm_add_ps(_mm_mul_ps(a0, b0), t1);
    }

    // Tail loop: numpy-style load_tillz_f32
    size_t remaining = n - i;
    while (remaining > 0) {
        __m128 va, vb;
        if (remaining >= 4) {
            va = _mm_loadu_ps(a + i);
            vb = _mm_loadu_ps(b + i);
            i += 4; remaining -= 4;
        } else if (remaining == 3) {
            // numpy's exact load pattern for 3 elements:
            // movq + movd + punpcklqdq
            __m128i ta = _mm_loadl_epi64((const __m128i*)(a + i));
            __m128i tb = _mm_loadl_epi64((const __m128i*)(b + i));
            ta = _mm_insert_epi32(ta, ((const int*)(a + i))[2], 2);
            tb = _mm_insert_epi32(tb, ((const int*)(b + i))[2], 2);
            va = _mm_castsi128_ps(ta);
            vb = _mm_castsi128_ps(tb);
            i += 3; remaining = 0;
        } else if (remaining == 2) {
            // numpy: movq
            va = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(a + i)));
            vb = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)(b + i)));
            i += 2; remaining = 0;
        } else { // remaining == 1
            // numpy: movd
            va = _mm_castsi128_ps(_mm_cvtsi32_si128(((const int*)(a + i))[0]));
            vb = _mm_castsi128_ps(_mm_cvtsi32_si128(((const int*)(b + i))[0]));
            i += 1; remaining = 0;
        }
        v_accum = _mm_add_ps(v_accum, _mm_mul_ps(va, vb));
    }

    // numpy: haddps(v,v); haddps(v,v)
    __m128 sum_halves = _mm_hadd_ps(v_accum, v_accum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
}

inline std::string implicit_output_labels(const std::vector<std::string>& il) {
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

// ============================================================================
// Main einsum computation
// ============================================================================

/// numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K',
///              casting='safe', optimize=False)
//  Currently supports 2-operand patterns only.
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

    for (int i = 0; i < 2; ++i) {
        if (static_cast<int>(in_labels[i].size()) != ndim[i])
            throw invalid_argument(
                "einsum: subscript '" + in_labels[i] + "' has " +
                to_string(in_labels[i].size()) + " labels but operand " +
                to_string(i) + " has " + to_string(ndim[i]) + " dimensions");
    }

    // Label → (input_idx, axis) mapping
    map<char, vector<pair<int, int>>> label_axis;
    map<char, ptrdiff_t> label_size;

    for (int i = 0; i < 2; ++i) {
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

    // Output shape
    vector<ptrdiff_t> output_shape;
    for (char c : output_labels)
        output_shape.push_back(label_size[c]);

    // ================================================================
    // Fast path: single contraction label that is the LAST axis in
    // BOTH operands (C-contiguous). Use BLAS dot per output element.
    // ================================================================
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

            // Map each output label to its axis in operands A and B.
            // output_label → (a_axis_or_neg, b_axis_or_neg)
            vector<pair<int, int>> out_label_axes;
            for (char c : output_labels) {
                int aa = -1, ba = -1;
                for (const auto& [inp, ax] : label_axis[c]) {
                    if (inp == 0) aa = ax;
                    if (inp == 1) ba = ax;
                }
                out_label_axes.push_back({aa, ba});
            }

            ptrdiff_t output_total = 1;
            for (ptrdiff_t s : output_shape) output_total *= s;

            vector<ptrdiff_t> out_coord(output_shape.size(), 0);

            for (ptrdiff_t oi = 0; oi < output_total; ++oi) {
                ptrdiff_t rem = oi;
                for (int d = static_cast<int>(output_shape.size()) - 1; d >= 0; --d) {
                    out_coord[d] = rem % output_shape[d];
                    rem /= output_shape[d];
                }

                ptrdiff_t a_off = 0, b_off = 0;
                for (size_t d = 0; d < out_label_axes.size(); ++d) {
                    int aa = out_label_axes[d].first;
                    int ba = out_label_axes[d].second;
                    if (aa >= 0) a_off += out_coord[d] * a_str[aa];
                    if (ba >= 0) b_off += out_coord[d] * b_str[ba];
                }

                if constexpr (std::is_same_v<T, double>)
                    result_ptr[oi] = einsum_reduce_f64(a_ptr + a_off, b_ptr + b_off, static_cast<size_t>(csize));
                else
                    result_ptr[oi] = einsum_reduce_f32(a_ptr + a_off, b_ptr + b_off, static_cast<size_t>(csize));
            }
            return;
        }
    }

    // ================================================================
    // Scalar path: general case.
    // ================================================================
    vector<char> iter_labels = output_labels;
    iter_labels.insert(iter_labels.end(), sum_labels.begin(), sum_labels.end());
    int n_iter = static_cast<int>(iter_labels.size());
    int n_output = static_cast<int>(output_labels.size());

    vector<ptrdiff_t> iter_sizes(n_iter);
    for (int i = 0; i < n_iter; ++i)
        iter_sizes[i] = label_size[iter_labels[i]];

    auto iter_strides = compute_strides(iter_sizes);
    auto strides_a = compute_strides(shapes[0]);
    auto strides_b = compute_strides(shapes[1]);
    const vector<ptrdiff_t>* input_strides[2] = {&strides_a, &strides_b};

    vector<vector<int>> iter_input_axis(n_iter, vector<int>(2, -1));
    for (int li = 0; li < n_iter; ++li) {
        char l = iter_labels[li];
        for (const auto& [inp, ax] : label_axis[l])
            iter_input_axis[li][inp] = ax;
    }

    vector<ptrdiff_t> iter_coord(n_iter, 0);
    vector<ptrdiff_t> input_coord[2] = {
        vector<ptrdiff_t>(ndim[0], 0),
        vector<ptrdiff_t>(ndim[1], 0)
    };

    ptrdiff_t iter_total = 1;
    for (ptrdiff_t s : iter_sizes) iter_total *= s;

    ptrdiff_t current_output_idx = -1;
    T accumulator = T(0);

    for (ptrdiff_t flat = 0; flat < iter_total; ++flat) {
        ptrdiff_t rem = flat;
        for (int i = 0; i < n_iter; ++i) {
            iter_coord[i] = rem / iter_strides[i];
            rem %= iter_strides[i];
        }

        bool is_new_output = true;
        for (int i = n_output; i < n_iter; ++i) {
            if (iter_coord[i] != 0) { is_new_output = false; break; }
        }

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

        for (int inp = 0; inp < 2; ++inp)
            fill(input_coord[inp].begin(), input_coord[inp].end(), 0);

        for (int li = 0; li < n_iter; ++li)
            for (int inp = 0; inp < 2; ++inp) {
                int ax = iter_input_axis[li][inp];
                if (ax >= 0) input_coord[inp][ax] = iter_coord[li];
            }

        ptrdiff_t idx_a = flat_index(input_coord[0], *input_strides[0]);
        ptrdiff_t idx_b = flat_index(input_coord[1], *input_strides[1]);

        accumulator += a_ptr[idx_a] * b_ptr[idx_b];
    }

    if (current_output_idx >= 0)
        result_ptr[current_output_idx] = accumulator;
}

/// numpy.einsum(subscripts, *operands) — compute output shape
inline std::vector<ptrdiff_t> einsum_output_shape(const std::string& subscripts,
                                                    const ptrdiff_t* a_shape, int a_ndim,
                                                    const ptrdiff_t* b_shape, int b_ndim) {
    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    std::string out_labels = parsed.output;

    if (!parsed.explicit_mode)
        out_labels = implicit_output_labels(in_labels);

    std::map<char, ptrdiff_t> label_size;
    for (int i = 0; i < a_ndim; ++i)
        label_size[in_labels[0][i]] = a_shape[i];
    for (int i = 0; i < b_ndim; ++i)
        label_size[in_labels[1][i]] = b_shape[i];

    std::vector<ptrdiff_t> output_shape;
    for (char c : out_labels)
        output_shape.push_back(label_size[c]);
    return output_shape;
}

} // namespace einsum_detail
} // namespace numpy
