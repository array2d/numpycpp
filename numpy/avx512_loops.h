// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  This file contains AVX-512 template specializations that override the  ║
// ║  generic loops in core.h.  It is x86_64 + AVX-512F specific and must   ║
// ║  be included INSIDE namespace numpy at the end of core.h — nowhere else.║
// ║                                                                          ║
// ║  ✗  #include "numpy/avx512_loops.h"     ← compile error                ║
// ║  ✓  #include "numpy/core.h"             ← only correct entry point      ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// AVX-512 wide-loop specializations for array math functions.
//
// Maintains 0 ULP vs numpy by using the exact same computation paths:
//   F64: 8-wide SVML loops  (_mm512_loadu_pd / __svml_exp8 / _mm512_storeu_pd)
//   F32 SVML group: 16-wide (_mm512_loadu_ps / __svml_tanf16 / _mm512_storeu_ps)
//   F32 poly group: 16-wide inline polynomial — constants pre-broadcast before loop
//
// Key design decisions:
//   - No masked loads past array boundary (scalar tail loop avoids UB)
//   - Static function-pointer init: resolved once on first call after numpy is loaded
//   - #ifdef __AVX512F__ guards everything; safe on non-AVX-512 builds
//   - F32 poly: all 14-17 constants are non-static locals defined BEFORE the loop.
//     GCC keeps them in zmm8-zmm31 throughout the loop (no per-iteration reloads),
//     matching numpy's simd_exp_FLOAT vbroadcastss-before-loop register pattern.
//     Previously these called noinline helpers → 32768 call/returns per 524k array.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "avx512_loops.h is an internal header — do not include directly. \
Use #include \"numpy/core.h\" instead."
#endif

#ifdef __AVX512F__
#include <immintrin.h>

// Bit-pattern float32 constant helper (avoids float-literal rounding errors).
// Used in the polynomial specializations below.
#define NUMPYCPP_BF(x) \
    (_mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(x))))

// ============================================================================
// sqrt / abs — simple wide-loop specializations (IEEE 754 exact, no SVML needed)
// ============================================================================

template<>
__attribute__((target("avx512f")))
inline void sqrt<double>(const double* __restrict__ s,
                          double* __restrict__ d, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(d+i, _mm512_sqrt_pd(_mm512_loadu_pd(s+i)));
    for (; i < n; ++i) d[i] = std::sqrt(s[i]);
}

template<>
__attribute__((target("avx512f")))
inline void sqrt<float>(const float* __restrict__ s,
                         float* __restrict__ d, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(d+i, _mm512_sqrt_ps(_mm512_loadu_ps(s+i)));
    for (; i < n; ++i) d[i] = std::sqrt(s[i]);
}

template<>
__attribute__((target("avx512f")))
inline void abs<double>(const double* __restrict__ s,
                         double* __restrict__ d, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(d+i, _mm512_abs_pd(_mm512_loadu_pd(s+i)));
    for (; i < n; ++i) d[i] = std::abs(s[i]);
}

template<>
__attribute__((target("avx512f")))
inline void abs<float>(const float* __restrict__ s,
                        float* __restrict__ d, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(d+i, _mm512_abs_ps(_mm512_loadu_ps(s+i)));
    for (; i < n; ++i) d[i] = std::abs(s[i]);
}

// ============================================================================
// F64 8-wide SVML template specializations
// ============================================================================

// fn_name = numpy:: template name; scalar_fn = detail:: scalar fallback suffix
#define NUMPYCPP_WIDE_F64(fn_name, svml_sym, scalar_fn)                         \
template<>                                                                       \
__attribute__((target("avx512f")))                                               \
inline void fn_name<double>(const double* __restrict__ s,                       \
                              double* __restrict__ d, size_t n) {               \
    using F8 = __m512d(__m512d);                                                 \
    static F8* fn = (F8*)detail::resolve_svml(svml_sym);                        \
    size_t i = 0;                                                                \
    if (__builtin_expect(fn != nullptr, 1)) {                                   \
        for (; i + 8 <= n; i += 8)                                              \
            _mm512_storeu_pd(d+i, fn(_mm512_loadu_pd(s+i)));                    \
    }                                                                            \
    for (; i < n; ++i) d[i] = detail::scalar_fn##_f64(s[i]);                   \
}

NUMPYCPP_WIDE_F64(exp,    "__svml_exp8",   exp)
NUMPYCPP_WIDE_F64(log,    "__svml_log8",   log)
NUMPYCPP_WIDE_F64(sin,    "__svml_sin8",   sin)
NUMPYCPP_WIDE_F64(cos,    "__svml_cos8",   cos)
NUMPYCPP_WIDE_F64(tan,    "__svml_tan8",   tan)
NUMPYCPP_WIDE_F64(arcsin, "__svml_asin8",  asin)
NUMPYCPP_WIDE_F64(arccos, "__svml_acos8",  acos)
NUMPYCPP_WIDE_F64(arctan, "__svml_atan8",  atan)
NUMPYCPP_WIDE_F64(log10,  "__svml_log108", log10)
NUMPYCPP_WIDE_F64(log2,   "__svml_log28",  log2)
NUMPYCPP_WIDE_F64(cbrt,   "__svml_cbrt8",  cbrt)
NUMPYCPP_WIDE_F64(expm1,  "__svml_expm18", expm1)
NUMPYCPP_WIDE_F64(log1p,  "__svml_log1p8", log1p)
#undef NUMPYCPP_WIDE_F64

// ============================================================================
// F32 16-wide SVML template specializations (SVML-backed functions)
// ============================================================================

#define NUMPYCPP_WIDE_F32_SVML(fn_name, svml_sym, scalar_fn)                    \
template<>                                                                       \
__attribute__((target("avx512f")))                                               \
inline void fn_name<float>(const float* __restrict__ s,                         \
                             float* __restrict__ d, size_t n) {                 \
    using F16 = __m512(__m512);                                                  \
    static F16* fn = (F16*)detail::resolve_svml(svml_sym);                      \
    size_t i = 0;                                                                \
    if (__builtin_expect(fn != nullptr, 1)) {                                   \
        for (; i + 16 <= n; i += 16)                                            \
            _mm512_storeu_ps(d+i, fn(_mm512_loadu_ps(s+i)));                    \
    }                                                                            \
    for (; i < n; ++i) d[i] = detail::scalar_fn##_f32(s[i]);                   \
}

NUMPYCPP_WIDE_F32_SVML(tan,    "__svml_tanf16",   tan)
NUMPYCPP_WIDE_F32_SVML(arcsin, "__svml_asinf16",  asin)
NUMPYCPP_WIDE_F32_SVML(arccos, "__svml_acosf16",  acos)
NUMPYCPP_WIDE_F32_SVML(arctan, "__svml_atanf16",  atan)
NUMPYCPP_WIDE_F32_SVML(log10,  "__svml_log10f16", log10)
NUMPYCPP_WIDE_F32_SVML(log2,   "__svml_log2f16",  log2)
NUMPYCPP_WIDE_F32_SVML(cbrt,   "__svml_cbrtf16",  cbrt)
NUMPYCPP_WIDE_F32_SVML(expm1,  "__svml_expm1f16", expm1)
NUMPYCPP_WIDE_F32_SVML(log1p,  "__svml_log1pf16", log1p)
#undef NUMPYCPP_WIDE_F32_SVML

// ============================================================================
// F32 16-wide polynomial specializations — exp / log / sin / cos
//
// All polynomial constants are non-static locals defined BEFORE the loop.
// GCC (with avx512f target) keeps them in zmm8-zmm31 for the entire loop —
// zero per-iteration constant loads, matching numpy's simd_exp_FLOAT pattern.
// Each function self-contained: coefficients match npy_math_float.h exactly.
// ============================================================================

// ----------------------------------------------------------------------------
// exp<float> — Cody-Waite range reduction + Pade P5/Q2 + vscalef
// ----------------------------------------------------------------------------
template<>
__attribute__((target("avx512f")))
inline void exp<float>(const float* __restrict__ s,
                        float* __restrict__ d, size_t n) {
    // Range limits and polynomial coefficients from npy_expf (npy_math_float.h)
    const __m512 VXmax  = _mm512_set1_ps(88.72283935546875f);
    const __m512 VXmin  = _mm512_set1_ps(-103.97208404541015625f);
    const __m512 Vlog2e = NUMPYCPP_BF(0x3fb8aa3bu);
    const __m512 Vln2h  = NUMPYCPP_BF(0xbf317200u);  // -ln(2) high part
    const __m512 Vln2l  = NUMPYCPP_BF(0xb5bfbe8eu);  // -ln(2) low part
    const __m512 Vmagic = NUMPYCPP_BF(0x4b400000u);  // 2^23 (round-to-int trick)
    const __m512 Vp0    = NUMPYCPP_BF(0x3f800000u);  // 1.0f
    const __m512 Vp1    = NUMPYCPP_BF(0x3f39cbd5u);
    const __m512 Vp2    = NUMPYCPP_BF(0x3e7d4c58u);
    const __m512 Vp3    = NUMPYCPP_BF(0x3d517d8cu);
    const __m512 Vp4    = NUMPYCPP_BF(0x3bdd7159u);
    const __m512 Vp5    = NUMPYCPP_BF(0x3a053dd8u);
    const __m512 Vq0    = NUMPYCPP_BF(0x3f800000u);  // 1.0f
    const __m512 Vq1    = NUMPYCPP_BF(0xbe8c6857u);
    const __m512 Vq2    = NUMPYCPP_BF(0x3cb0e832u);
    const __m512 Vinf   = _mm512_set1_ps(__builtin_inff());

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(s + i);

        // qf = rint(x * log2(e)) via add/sub magic trick (plain mul+add, no FMA)
        __m512 qf = _mm512_add_ps(_mm512_mul_ps(x, Vlog2e), Vmagic);
        qf = _mm512_sub_ps(qf, Vmagic);  // qf is now integer-valued

        // Cody-Waite: r = x + qf*ln2h + qf*ln2l  (plain mul+add, matches scalar)
        __m512 r = _mm512_add_ps(x, _mm512_mul_ps(qf, Vln2h));
        r        = _mm512_add_ps(r, _mm512_mul_ps(qf, Vln2l));

        // P(r) Horner — degree 5 numerator
        __m512 num = _mm512_fmadd_ps(Vp5, r, Vp4);
        num = _mm512_fmadd_ps(num, r, Vp3);
        num = _mm512_fmadd_ps(num, r, Vp2);
        num = _mm512_fmadd_ps(num, r, Vp1);
        num = _mm512_fmadd_ps(num, r, Vp0);

        // Q(r) Horner — degree 2 denominator
        __m512 den = _mm512_fmadd_ps(Vq2, r, Vq1);
        den = _mm512_fmadd_ps(den, r, Vq0);

        // poly = P(r)/Q(r) * 2^qi  (qf is already integer-valued → scalef exact)
        __m512 poly = _mm512_scalef_ps(_mm512_div_ps(num, den), qf);

        // overflow → +Inf, underflow → 0.0f  (NaN propagates through; ±Inf correct)
        poly = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, VXmax, _CMP_GE_OQ),
                                    poly, Vinf);
        poly = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, VXmin, _CMP_LE_OQ),
                                    poly, _mm512_setzero_ps());
        // NaN passthrough: ordered comparisons above return false for NaN → poly holds
        // polynomial-derived NaN; blend back original x to guarantee bit-exact match
        // with numpy's scalar npy_expf which returns x unchanged for NaN input.
        __mmask16 is_nan_e = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        poly = _mm512_mask_blend_ps(is_nan_e, poly, x);
        _mm512_storeu_ps(d + i, poly);
    }
    for (; i < n; ++i) d[i] = detail::exp_npy_f32(s[i]);
}

// ----------------------------------------------------------------------------
// log<float> — exponent extract + Pade P5/Q5 + k*ln2
// ----------------------------------------------------------------------------
template<>
__attribute__((target("avx512f")))
inline void log<float>(const float* __restrict__ s,
                        float* __restrict__ d, size_t n) {
    // Coefficients from npy_logf (npy_math_float.h)
    const __m512 Vsqrt_half = NUMPYCPP_BF(0x3f3504f3u);
    const __m512 Vln2       = NUMPYCPP_BF(0x3f317218u);
    // P: p0=0 (absorbed into final mul), p1..p5
    const __m512 Vp1 = NUMPYCPP_BF(0x3f800000u);
    const __m512 Vp2 = NUMPYCPP_BF(0x4007361cu);
    const __m512 Vp3 = NUMPYCPP_BF(0x3fbd70a9u);
    const __m512 Vp4 = NUMPYCPP_BF(0x3ec30333u);
    const __m512 Vp5 = NUMPYCPP_BF(0x3cd42bcdu);
    // Q: q0..q5
    const __m512 Vq0 = NUMPYCPP_BF(0x3f800000u);
    const __m512 Vq1 = NUMPYCPP_BF(0x4027361cu);
    const __m512 Vq2 = NUMPYCPP_BF(0x401cfe0du);
    const __m512 Vq3 = NUMPYCPP_BF(0x3f7c8ae4u);
    const __m512 Vq4 = NUMPYCPP_BF(0x3e1e5bf3u);
    const __m512 Vq5 = NUMPYCPP_BF(0x3bc083dfu);
    // Special case return values
    const __m512 Vneginf = _mm512_set1_ps(-__builtin_inff());
    const __m512 Vneqnan = NUMPYCPP_BF(0xffc00000u);  // -quiet NaN
    const __m512 Vposinf = _mm512_set1_ps(__builtin_inff());

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(s + i);
        __m512i bits = _mm512_castps_si512(x);

        // Extract biased exponent: k = exp_field - 126 (= unbiased + 1)
        __m512i exp_field = _mm512_and_si512(_mm512_srli_epi32(bits, 23),
                                              _mm512_set1_epi32(0xff));
        __m512 exponent = _mm512_sub_ps(_mm512_cvtepi32_ps(exp_field),
                                         _mm512_set1_ps(126.0f));

        // Build m in [0.5, 1): replace exponent bits with 0x3f (biased = 63)
        __m512i mant_bits = _mm512_or_si512(
                                _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffff)),
                                _mm512_set1_epi32(0x3f000000));
        __m512 m = _mm512_castsi512_ps(mant_bits);

        // If m <= sqrt(1/2): m *= 2, exponent -= 1  (keeps m in [sqrt(1/2), 1))
        __mmask16 small = _mm512_cmp_ps_mask(m, Vsqrt_half, _CMP_LE_OQ);
        m        = _mm512_mask_add_ps(m,        small, m,        m);
        exponent = _mm512_mask_sub_ps(exponent, small, exponent,
                                       _mm512_set1_ps(1.0f));

        __m512 y = _mm512_sub_ps(m, _mm512_set1_ps(1.0f));  // y in (sqrt(1/2)-1, 1)

        // P(y): p0=0 → last Horner step is mul (fma with zero addend = mul)
        __m512 num = _mm512_fmadd_ps(Vp5, y, Vp4);
        num = _mm512_fmadd_ps(num, y, Vp3);
        num = _mm512_fmadd_ps(num, y, Vp2);
        num = _mm512_fmadd_ps(num, y, Vp1);
        num = _mm512_mul_ps(num, y);  // fma(num, y, p0=0) = num * y

        // Q(y)
        __m512 den = _mm512_fmadd_ps(Vq5, y, Vq4);
        den = _mm512_fmadd_ps(den, y, Vq3);
        den = _mm512_fmadd_ps(den, y, Vq2);
        den = _mm512_fmadd_ps(den, y, Vq1);
        den = _mm512_fmadd_ps(den, y, Vq0);

        // result = exponent * ln(2) + P(y)/Q(y)
        __m512 result = _mm512_fmadd_ps(exponent, Vln2, _mm512_div_ps(num, den));

        // Special cases (checked after main polynomial — masks override)
        // Note: integer bit ops above strip NaN bits → must restore NaN explicitly.
        __mmask16 is_zero = _mm512_cmpeq_epi32_mask(
                                _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                                _mm512_setzero_si512());
        __mmask16 is_neg    = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 is_posinf = _mm512_cmp_ps_mask(x, Vposinf, _CMP_EQ_OQ);
        __mmask16 is_nan    = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q); // x!=x → NaN
        result = _mm512_mask_blend_ps(is_zero,   result, Vneginf);
        result = _mm512_mask_blend_ps(is_neg,    result, Vneqnan);
        result = _mm512_mask_blend_ps(is_posinf, result, Vposinf);
        result = _mm512_mask_blend_ps(is_nan,    result, x);  // NaN passthrough
        _mm512_storeu_ps(d + i, result);
    }
    for (; i < n; ++i) d[i] = detail::log_npy_f32(s[i]);
}

// ----------------------------------------------------------------------------
// sin<float> — Cody-Waite PI/2 reduction + degree-9 sincos polynomial
// ----------------------------------------------------------------------------
template<>
__attribute__((target("avx512f")))
inline void sin<float>(const float* __restrict__ s,
                        float* __restrict__ d, size_t n) {
    // Constants from npy_sincosf (npy_math_float.h)
    const __m512 Vtwo_over_pi = NUMPYCPP_BF(0x3f22f983u);
    const __m512 Vpio2_high   = NUMPYCPP_BF(0xbfc90fd8u);
    const __m512 Vpio2_med    = NUMPYCPP_BF(0xb4a8885au);
    const __m512 Vpio2_low    = NUMPYCPP_BF(0xa7c234c4u);
    const __m512 Vmagic       = NUMPYCPP_BF(0x4b400000u);
    const __m512 Vcos_c0 = NUMPYCPP_BF(0x3f800000u);
    const __m512 Vcos_c2 = NUMPYCPP_BF(0xbf000000u);
    const __m512 Vcos_c4 = NUMPYCPP_BF(0x3d2aaa9eu);
    const __m512 Vcos_c6 = NUMPYCPP_BF(0xbab6036eu);
    const __m512 Vcos_c8 = NUMPYCPP_BF(0x37cc730bu);
    const __m512 Vsin_s3 = NUMPYCPP_BF(0xbe2aaaabu);
    const __m512 Vsin_s5 = NUMPYCPP_BF(0x3c0888cdu);
    const __m512 Vsin_s7 = NUMPYCPP_BF(0xb95035ddu);
    const __m512 Vsin_s9 = NUMPYCPP_BF(0x363e9ddeu);
    const __m512 Vmax    = _mm512_set1_ps(117435.992f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(s + i);

        // Round x/(pi/2) to nearest integer
        __m512 qf = _mm512_add_ps(_mm512_mul_ps(x, Vtwo_over_pi), Vmagic);
        qf = _mm512_sub_ps(qf, Vmagic);

        // Cody-Waite range reduction (3-part, plain mul+add)
        __m512 r = _mm512_add_ps(x, _mm512_mul_ps(qf, Vpio2_high));
        r        = _mm512_add_ps(r, _mm512_mul_ps(qf, Vpio2_med));
        r        = _mm512_add_ps(r, _mm512_mul_ps(qf, Vpio2_low));

        __m512 r2 = _mm512_mul_ps(r, r);

        // cos polynomial in r2
        __m512 cp = _mm512_fmadd_ps(Vcos_c8, r2, Vcos_c6);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c4);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c2);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c0);

        // sin polynomial in r, r2
        __m512 sp = _mm512_fmadd_ps(Vsin_s9, r2, Vsin_s7);
        sp = _mm512_fmadd_ps(sp, r2, Vsin_s5);
        sp = _mm512_fmadd_ps(sp, r2, Vsin_s3);
        sp = _mm512_mul_ps(sp, r2);        // sp*r2 (+0 offset)
        sp = _mm512_fmadd_ps(sp, r, r);   // fma(sp, r, r)

        // Quadrant selection: iq&1 → use cos poly; iq&2 → negate
        __m512i iq = _mm512_cvttps_epi32(qf);
        __mmask16 sel_cp = _mm512_test_epi32_mask(iq, _mm512_set1_epi32(1));
        __m512 result = _mm512_mask_blend_ps(sel_cp, sp, cp);
        __mmask16 neg  = _mm512_test_epi32_mask(iq, _mm512_set1_epi32(2));
        result = _mm512_mask_sub_ps(result, neg, _mm512_setzero_ps(), result);

        // sin(±0) = ±0: signed-zero must be preserved (numpy matches IEEE 754)
        // fma(sp, r, r) with r=±0 gives +0 due to IEEE 754 RN rules → blend back x
        __mmask16 is_zero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ);
        result = _mm512_mask_blend_ps(is_zero, result, x);

        // Out-of-range scalar fallback (rarely triggered; branch predicted-not-taken)
        __mmask16 inr = _mm512_cmp_ps_mask(_mm512_abs_ps(x), Vmax, _CMP_LE_OQ);
        if (__builtin_expect(inr != 0xFFFF, 0)) {
            float xt[16], rt[16];
            _mm512_storeu_ps(xt, x);
            _mm512_storeu_ps(rt, result);
            for (int j = 0; j < 16; ++j)
                if (!((inr >> j) & 1)) rt[j] = std::sin(xt[j]);
            result = _mm512_loadu_ps(rt);
        }
        // NaN passthrough: blend back original x after fallback so NaN output = NaN
        // input (bit-exact with numpy's scalar npy_sinf which returns x for NaN).
        __mmask16 is_nan_s = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        result = _mm512_mask_blend_ps(is_nan_s, result, x);
        _mm512_storeu_ps(d + i, result);
    }
    // sin_f32 adds signed-zero fix: sin(±0)=±0 (npy_sinf polynomial gives +0 for -0).
    for (; i < n; ++i) d[i] = detail::sin_f32(s[i]);
}

// ----------------------------------------------------------------------------
// cos<float> — same polynomial as sin, shifted quadrant index by +1
// ----------------------------------------------------------------------------
template<>
__attribute__((target("avx512f")))
inline void cos<float>(const float* __restrict__ s,
                        float* __restrict__ d, size_t n) {
    const __m512 Vtwo_over_pi = NUMPYCPP_BF(0x3f22f983u);
    const __m512 Vpio2_high   = NUMPYCPP_BF(0xbfc90fd8u);
    const __m512 Vpio2_med    = NUMPYCPP_BF(0xb4a8885au);
    const __m512 Vpio2_low    = NUMPYCPP_BF(0xa7c234c4u);
    const __m512 Vmagic       = NUMPYCPP_BF(0x4b400000u);
    const __m512 Vcos_c0 = NUMPYCPP_BF(0x3f800000u);
    const __m512 Vcos_c2 = NUMPYCPP_BF(0xbf000000u);
    const __m512 Vcos_c4 = NUMPYCPP_BF(0x3d2aaa9eu);
    const __m512 Vcos_c6 = NUMPYCPP_BF(0xbab6036eu);
    const __m512 Vcos_c8 = NUMPYCPP_BF(0x37cc730bu);
    const __m512 Vsin_s3 = NUMPYCPP_BF(0xbe2aaaabu);
    const __m512 Vsin_s5 = NUMPYCPP_BF(0x3c0888cdu);
    const __m512 Vsin_s7 = NUMPYCPP_BF(0xb95035ddu);
    const __m512 Vsin_s9 = NUMPYCPP_BF(0x363e9ddeu);
    const __m512 Vmax    = _mm512_set1_ps(71476.0625f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(s + i);

        __m512 qf = _mm512_add_ps(_mm512_mul_ps(x, Vtwo_over_pi), Vmagic);
        qf = _mm512_sub_ps(qf, Vmagic);

        __m512 r = _mm512_add_ps(x, _mm512_mul_ps(qf, Vpio2_high));
        r        = _mm512_add_ps(r, _mm512_mul_ps(qf, Vpio2_med));
        r        = _mm512_add_ps(r, _mm512_mul_ps(qf, Vpio2_low));

        __m512 r2 = _mm512_mul_ps(r, r);

        __m512 cp = _mm512_fmadd_ps(Vcos_c8, r2, Vcos_c6);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c4);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c2);
        cp = _mm512_fmadd_ps(cp, r2, Vcos_c0);

        __m512 sp = _mm512_fmadd_ps(Vsin_s9, r2, Vsin_s7);
        sp = _mm512_fmadd_ps(sp, r2, Vsin_s5);
        sp = _mm512_fmadd_ps(sp, r2, Vsin_s3);
        sp = _mm512_mul_ps(sp, r2);
        sp = _mm512_fmadd_ps(sp, r, r);

        // cos: iq = int(qf) + 1  (shift by 1 vs sin)
        __m512i iq = _mm512_add_epi32(_mm512_cvttps_epi32(qf), _mm512_set1_epi32(1));
        __mmask16 sel_cp = _mm512_test_epi32_mask(iq, _mm512_set1_epi32(1));
        __m512 result = _mm512_mask_blend_ps(sel_cp, sp, cp);
        __mmask16 neg  = _mm512_test_epi32_mask(iq, _mm512_set1_epi32(2));
        result = _mm512_mask_sub_ps(result, neg, _mm512_setzero_ps(), result);

        __mmask16 inr = _mm512_cmp_ps_mask(_mm512_abs_ps(x), Vmax, _CMP_LE_OQ);
        if (__builtin_expect(inr != 0xFFFF, 0)) {
            float xt[16], rt[16];
            _mm512_storeu_ps(xt, x);
            _mm512_storeu_ps(rt, result);
            for (int j = 0; j < 16; ++j)
                if (!((inr >> j) & 1)) rt[j] = std::cos(xt[j]);
            result = _mm512_loadu_ps(rt);
        }
        // NaN passthrough: blend back original x (bit-exact with numpy scalar npy_cosf).
        __mmask16 is_nan_c = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
        result = _mm512_mask_blend_ps(is_nan_c, result, x);
        _mm512_storeu_ps(d + i, result);
    }
    for (; i < n; ++i) d[i] = detail::cos_npy_f32(s[i]);
}

#undef NUMPYCPP_BF

#endif // __AVX512F__
