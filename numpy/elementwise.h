// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/elementwise.h                       [PUBLIC HEADER]
//
//  Element-wise operations (unary, binary, comparison, logical, type conv).
//
//  Unary math:
//      numpy.sqrt    numpy.abs     numpy.exp     numpy.log
//      numpy.sin     numpy.cos     numpy.tan     numpy.cbrt
//      numpy.expm1   numpy.log1p   numpy.power   numpy.clip
//      numpy.log10   numpy.log2    numpy.arcsin  numpy.arccos
//      numpy.arctan  numpy.round   numpy.floor   numpy.ceil
//      numpy.degrees numpy.radians numpy.sign
//
//  Binary element-wise:
//      numpy.hypot   numpy.arctan2  numpy.maximum  numpy.minimum
//
//  Comparison (T → bool):
//      numpy.greater  numpy.less  numpy.equal  numpy.greater_equal
//      numpy.less_equal  numpy.not_equal
//
//  Logical (bool → bool):
//      numpy.logical_and  numpy.logical_or  numpy.logical_not  numpy.logical_xor
//
//  Special value tests:
//      numpy.isnan  numpy.isinf  numpy.isfinite
//
//  Type conversion:
//      ndarray.astype   truncate_to_float32
//
//  Recommended entry point: #include "numpy/numpy.h"
//  Direct include is also valid for standalone use.
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

// ── Internal detail headers ──────────────────────────────────────────────────
// math_backend.h selects the math implementation at compile time:
//   NUMPYCPP_STD_ONLY not set → svml_bridge.h (bit-exact, dlsym + SVML)
//   NUMPYCPP_STD_ONLY     set → std_math_backend.h (pure <cmath>, perf-first)
// avx512_loops.h provides AVX-512 specialisations; skipped in STD_ONLY mode.
// Both require NUMPYCPP_INTERNAL_INCLUDE; we manage that here.
#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  define NUMPYCPP_INTERNAL_INCLUDE
#  define _NUMPYCPP_EW_OWNS_GUARD
#endif
#include "detail/math_backend.h"
#include "detail/macros.h"   // NUMPY_UNROLL4

namespace numpy {

// ============================================================================
// Unary element-wise math — T in → T out
// ============================================================================

/// numpy.sqrt(x, /, out=None, *, where=True, ...)
template<typename T>
inline void sqrt(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::sqrt(src[i]));
}

/// numpy.abs(x, /, out=None, *, where=True, ...)
template<typename T>
inline void abs(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::abs(src[i]));
}

/// numpy.exp(x, /, out=None, *, where=True, ...)
template<typename T>
inline void exp(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::exp(src[i]));
}

/// numpy.log(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log(src[i]));
}

/// numpy.sin(x, /, out=None, *, where=True, ...)
template<typename T>
inline void sin(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::sin(src[i]));
}

/// numpy.cos(x, /, out=None, *, where=True, ...)
template<typename T>
inline void cos(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::cos(src[i]));
}

/// numpy.tan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void tan(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::tan(src[i]));
}

/// numpy.cbrt(x, /, out=None, *, where=True, ...)
template<typename T>
inline void cbrt(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::cbrt(src[i]));
}

/// numpy.expm1(x, /, out=None, *, where=True, ...)
template<typename T>
inline void expm1(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::expm1(src[i]));
}

/// numpy.log1p(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log1p(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log1p(src[i]));
}

/// numpy.power(x1, x2, /, out=None, *, where=True, ...) — scalar exponent
template<typename T>
inline void power(const T* src, T* dst, size_t n, T exponent) {
    NUMPY_UNROLL4(i, dst[i] = detail::pow(src[i], exponent));
}

/// numpy.clip(a, a_min, a_max, out=None, **kwargs)
template<typename T>
inline void clip(const T* src, T* dst, size_t n, T min_val, T max_val) {
    NUMPY_UNROLL4(i, dst[i] = std::max(min_val, std::min(max_val, src[i])));
}

/// numpy.log10(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log10(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log10(src[i]));
}

/// numpy.log2(x, /, out=None, *, where=True, ...)
template<typename T>
inline void log2(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::log2(src[i]));
}

/// numpy.arcsin(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arcsin(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::asin(src[i]));
}

/// numpy.arccos(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arccos(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::acos(src[i]));
}

/// numpy.arctan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void arctan(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan(src[i]));
}

/// numpy.round(a, decimals=0, out=None)
template<typename T>
inline void round(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::nearbyint(src[i]));
}

/// numpy.floor(x, /, out=None, *, where=True, ...)
template<typename T>
inline void floor(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::floor(src[i]));
}

/// numpy.ceil(x, /, out=None, *, where=True, ...)
template<typename T>
inline void ceil(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::ceil(src[i]));
}

/// numpy.degrees(x, /, out=None, *, where=True, ...)
template<typename T>
inline void degrees(const T* src, T* dst, size_t n) {
    T factor = T(180.0) / T(M_PI);
    NUMPY_UNROLL4(i, dst[i] = src[i] * factor);
}

/// numpy.radians(x, /, out=None, *, where=True, ...)
template<typename T>
inline void radians(const T* src, T* dst, size_t n) {
    T factor = T(M_PI) / T(180.0);
    NUMPY_UNROLL4(i, dst[i] = src[i] * factor);
}

/// numpy.sign(x, /, out=None, *, where=True, ...)
/// NaN input → NaN output (numpy behaviour).
template<typename T>
inline void sign(const T* src, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isnan(src[i]) ? src[i]
                                                  : T((src[i] > T(0)) - (src[i] < T(0))));
}

// ============================================================================
// Binary element-wise — 2 arrays T in → T out
// ============================================================================

/// numpy.hypot(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void hypot(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::hypot(a[i], b[i]));
}

/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void arctan2(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan2(a[i], b[i]));
}

/// numpy.arctan2(x1, x2, /, out=None, *, where=True, ...) — array-scalar
template<typename T>
inline void arctan2(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = detail::atan2(src[i], b));
}

/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void maximum(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::max(a[i], b[i]));
}

/// numpy.maximum(x1, x2, /, out=None, *, where=True, ...) — array-scalar
template<typename T>
inline void maximum(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = std::max(src[i], b));
}

/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — array-array
template<typename T>
inline void minimum(const T* a, const T* b, T* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::min(a[i], b[i]));
}

/// numpy.minimum(x1, x2, /, out=None, *, where=True, ...) — array-scalar
template<typename T>
inline void minimum(const T* src, T* dst, size_t n, T b) {
    NUMPY_UNROLL4(i, dst[i] = std::min(src[i], b));
}

// ============================================================================
// Comparison — T in → bool out
// ============================================================================

/// numpy.greater(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void greater(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] > threshold));
}

/// numpy.less(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void less(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] < threshold));
}

/// numpy.equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void equal(const T* src, bool* dst, size_t n, T val) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] == val));
}

/// numpy.greater_equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void greater_equal(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] >= threshold));
}

/// numpy.less_equal(x1, x2, /, out=None, *, where=True, ...)
template<typename T>
inline void less_equal(const T* src, bool* dst, size_t n, T threshold) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] <= threshold));
}

/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — scalar variant
template<typename T>
inline void not_equal_scalar(const T* src, bool* dst, size_t n, T val) {
    NUMPY_UNROLL4(i, dst[i] = (src[i] != val));
}

/// numpy.not_equal(x1, x2, /, out=None, *, where=True, ...) — array variant
template<typename T>
inline void not_equal_array(const T* a, const T* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = (a[i] != b[i]));
}

// ============================================================================
// Logical — bool in → bool out
// ============================================================================

/// numpy.logical_and(x1, x2, /, out=None, *, where=True, ...)
inline void logical_and(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] && b[i]);
}

/// numpy.logical_or(x1, x2, /, out=None, *, where=True, ...)
inline void logical_or(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] || b[i]);
}

/// numpy.logical_not(x, /, out=None, *, where=True, ...)
inline void logical_not(const bool* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = !src[i]);
}

/// numpy.logical_xor(x1, x2, /, out=None, *, where=True, ...)
inline void logical_xor(const bool* a, const bool* b, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = a[i] ^ b[i]);
}

// ============================================================================
// Special value tests — T in → bool out
// ============================================================================

/// numpy.isnan(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isnan(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isnan(src[i]));
}

/// numpy.isinf(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isinf(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isinf(src[i]));
}

/// numpy.isfinite(x, /, out=None, *, where=True, ...)
template<typename T>
inline void isfinite(const T* src, bool* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = std::isfinite(src[i]));
}

// ============================================================================
// Type conversion (treated as element-wise cast)
// ============================================================================

/// ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
template<typename Tout, typename Tin>
inline void astype(const Tin* src, Tout* dst, size_t n) {
    NUMPY_UNROLL4(i, dst[i] = static_cast<Tout>(src[i]));
}

/// float64 → float32 → float64 roundtrip (for precision testing)
inline void truncate_to_float32(const double* src, double* dst, size_t n) {
    NUMPY_UNROLL4(i, { float tmp = static_cast<float>(src[i]);
                       dst[i] = static_cast<double>(tmp); });
}

// ============================================================================
// AVX-512 wide-loop template specialisations (0 ULP, ~8-16x faster)
// Must appear inside namespace numpy after all primary templates.
// Skipped in STD_ONLY mode (no SVML, no AVX-512 intrinsics needed).
// ============================================================================
#ifndef NUMPYCPP_STD_ONLY
#include "detail/avx512_loops.h"
#endif

} // namespace numpy

// Release the internal-include guard if we set it ourselves.
#ifdef _NUMPYCPP_EW_OWNS_GUARD
#  undef NUMPYCPP_INTERNAL_INCLUDE
#  undef _NUMPYCPP_EW_OWNS_GUARD
#endif
