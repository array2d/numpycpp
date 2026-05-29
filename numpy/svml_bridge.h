// SVML/npy bridge — resolve functions from numpy's _multiarray_umath.so.
//
// numpy uses SVML for ALL transcendental functions when available:
//   f64: __svml_exp8, __svml_log8, __svml_sin8, __svml_cos8, __svml_tan8,
//        __svml_asin8, __svml_acos8, __svml_atan8, __svml_log108, __svml_log28,
//        __svml_exp28, npy_pow, npy_atan2
//   f32: __svml_expf16, __svml_logf16, __svml_sinf16, __svml_cosf16, __svml_tanf16,
//        __svml_asinf16, __svml_acosf16, __svml_atanf16, __svml_log10f16, __svml_log2f16,
//        __svml_exp2f16, npy_pow, npy_atan2
//
// Call bridge_init(path_to_multiarray_umath_so) before first use.

#pragma once

#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include "npy_math_float.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace numpy {
namespace svml {

inline void* g_svml_handle = nullptr;

inline void bridge_init(const char* numpy_so_path) {
    static bool initialized = false;
    if (initialized || !numpy_so_path) return;
    initialized = true;
    g_svml_handle = dlopen(numpy_so_path, RTLD_NOLOAD | RTLD_LAZY);
    fprintf(stderr, "[SVML] bridge_init(%s) -> handle=%p dlerror=%s\n",
            numpy_so_path, g_svml_handle, dlerror());
}

inline void* resolve_svml(const char* name) {
    void* h = g_svml_handle;
    if (h) {
        void* ptr = dlsym(h, name);
        fprintf(stderr, "[SVML] resolve(%s) -> %p dlerror=%s\n", name, ptr, dlerror());
        return ptr;
    }
    fprintf(stderr, "[SVML] resolve(%s) -> NO HANDLE (g_svml_handle is null)\n", name);
    return nullptr;
}

#ifdef __AVX512F__

// ============================================================================
// float64 — SVML for most, npy_pow / npy_atan2 for pow/atan2
// ============================================================================

inline double exp_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_exp8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::exp(x);
}
inline double log_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_log8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::log(x);
}
inline double sin_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_sin8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::sin(x);
}
inline double cos_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_cos8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::cos(x);
}
inline double tan_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_tan8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::tan(x);
}
inline double asin_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_asin8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::asin(x);
}
inline double acos_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_acos8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::acos(x);
}
inline double atan_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_atan8");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::atan(x);
}
inline double log10_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_log108");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::log10(x);
}
inline double log2_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_log28");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::log2(x);
}
inline double exp2_f64(double x) {
    static auto fn = (__m512d (*)(__m512d))resolve_svml("__svml_exp28");
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));
    return std::exp2(x);
}

// numpy uses npy_pow / npy_atan2 (scalar C functions, not SVML vector)
inline double pow_f64(double x, double e) {
    static auto fn = (double (*)(double, double))resolve_svml("npy_pow");
    if (fn) return fn(x, e);
    return std::pow(x, e);
}
inline double atan2_f64(double y, double x) {
    static auto fn = (double (*)(double, double))resolve_svml("npy_atan2");
    if (fn) return fn(y, x);
    return std::atan2(y, x);
}

// ============================================================================
// float32 — SVML for ALL transcendental functions
// ============================================================================

// exp/log/sin/cos — use numpy's own polynomial approximations for bit-exact results
inline float exp_f32(float x)  { return npy_float_math::npy_expf(x); }
inline float log_f32(float x)  { return npy_float_math::npy_logf(x); }
inline float sin_f32(float x)  { return npy_float_math::npy_sinf(x); }
inline float cos_f32(float x)  { return npy_float_math::npy_cosf(x); }
inline float tan_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_tanf16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::tan(x);
}
inline float asin_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_asinf16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::asin(x);
}
inline float acos_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_acosf16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::acos(x);
}
inline float atan_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_atanf16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::atan(x);
}
inline float log10_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_log10f16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::log10(x);
}
inline float log2_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_log2f16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::log2(x);
}
inline float exp2_f32(float x) {
    static auto fn = (__m512 (*)(__m512))resolve_svml("__svml_exp2f16");
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));
    return std::exp2(x);
}
inline float pow_f32(float x, float e) { return std::pow(x, e); }
inline float atan2_f32(float y, float x) { return std::atan2(y, x); }

#else // !__AVX512F__

inline double exp_f64(double x)  { return std::exp(x); }
inline double log_f64(double x)  { return std::log(x); }
inline double sin_f64(double x)  { return std::sin(x); }
inline double cos_f64(double x)  { return std::cos(x); }
inline double tan_f64(double x)  { return std::tan(x); }
inline double asin_f64(double x) { return std::asin(x); }
inline double acos_f64(double x) { return std::acos(x); }
inline double atan_f64(double x) { return std::atan(x); }
inline double log10_f64(double x){ return std::log10(x); }
inline double log2_f64(double x) { return std::log2(x); }
inline double exp2_f64(double x) { return std::exp2(x); }
inline double pow_f64(double x, double e) { return std::pow(x, e); }
inline double atan2_f64(double y, double x) { return std::atan2(y, x); }
inline float  exp_f32(float x)   { return npy_float_math::npy_expf(x); }
inline float  log_f32(float x)   { return npy_float_math::npy_logf(x); }
inline float  sin_f32(float x)   { return npy_float_math::npy_sinf(x); }
inline float  cos_f32(float x)   { return npy_float_math::npy_cosf(x); }
inline float  tan_f32(float x)   { return std::tan(x); }
inline float  asin_f32(float x)  { return std::asin(x); }
inline float  acos_f32(float x)  { return std::acos(x); }
inline float  atan_f32(float x)  { return std::atan(x); }
inline float  log10_f32(float x) { return std::log10(x); }
inline float  log2_f32(float x)  { return std::log2(x); }
inline float  exp2_f32(float x)  { return std::exp2(x); }
inline float  pow_f32(float x, float e)   { return std::pow(x, e); }
inline float  atan2_f32(float y, float x) { return std::atan2(y, x); }

#endif // __AVX512F__

// ============================================================================
// Template dispatchers
// ============================================================================

template<typename T> struct svml_impl;
template<> struct svml_impl<double> {
    static double exp(double x)  { return exp_f64(x); }
    static double log(double x)  { return log_f64(x); }
    static double sin(double x)  { return sin_f64(x); }
    static double cos(double x)  { return cos_f64(x); }
    static double tan(double x)  { return tan_f64(x); }
    static double asin(double x) { return asin_f64(x); }
    static double acos(double x) { return acos_f64(x); }
    static double atan(double x) { return atan_f64(x); }
    static double log10(double x){ return log10_f64(x); }
    static double log2(double x) { return log2_f64(x); }
    static double exp2(double x) { return exp2_f64(x); }
    static double pow(double x, double e)  { return pow_f64(x, e); }
    static double atan2(double y, double x) { return atan2_f64(y, x); }
};
template<> struct svml_impl<float> {
    static float exp(float x)  { return exp_f32(x); }
    static float log(float x)  { return log_f32(x); }
    static float sin(float x)  { return sin_f32(x); }
    static float cos(float x)  { return cos_f32(x); }
    static float tan(float x)  { return tan_f32(x); }
    static float asin(float x) { return asin_f32(x); }
    static float acos(float x) { return acos_f32(x); }
    static float atan(float x) { return atan_f32(x); }
    static float log10(float x){ return log10_f32(x); }
    static float log2(float x) { return log2_f32(x); }
    static float exp2(float x) { return exp2_f32(x); }
    static float pow(float x, float e)   { return pow_f32(x, e); }
    static float atan2(float y, float x) { return atan2_f32(y, x); }
};

template<typename T> inline T exp(T x)  { return svml_impl<T>::exp(x); }
template<typename T> inline T log(T x)  { return svml_impl<T>::log(x); }
template<typename T> inline T sin(T x)  { return svml_impl<T>::sin(x); }
template<typename T> inline T cos(T x)  { return svml_impl<T>::cos(x); }
template<typename T> inline T tan(T x)  { return svml_impl<T>::tan(x); }
template<typename T> inline T asin(T x) { return svml_impl<T>::asin(x); }
template<typename T> inline T acos(T x) { return svml_impl<T>::acos(x); }
template<typename T> inline T atan(T x) { return svml_impl<T>::atan(x); }
template<typename T> inline T log10(T x) { return svml_impl<T>::log10(x); }
template<typename T> inline T log2(T x)  { return svml_impl<T>::log2(x); }
template<typename T> inline T exp2(T x)  { return svml_impl<T>::exp2(x); }
template<typename T> inline T pow(T x, T e)    { return svml_impl<T>::pow(x, e); }
template<typename T> inline T atan2(T y, T x)  { return svml_impl<T>::atan2(y, x); }

} // namespace svml
} // namespace numpy
