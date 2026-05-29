// SVML/npy bridge — resolve functions from numpy's _multiarray_umath.so.
//
// numpy uses SVML for ALL transcendental functions when available.
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
    fprintf(stderr, "[SVML] resolve(%s) -> NO HANDLE\n", name);
    return nullptr;
}

// ============================================================================
// SVML wrapper macros — consolidate repetitive f64/f32 patterns
// ============================================================================

#ifdef __AVX512F__

// 1-arg f64: __svml_XXX8 → __m512d → extract scalar
// 1-arg f32: __svml_XXXf16 → __m512 → extract scalar
#define NUMPY_SVML_F64(name, sym, fallback)          \
    inline double name(double x) {                    \
        static auto fn = (__m512d (*)(__m512d))       \
            resolve_svml(sym);                        \
        if (fn) return _mm512_cvtsd_f64(              \
            fn(_mm512_set1_pd(x)));                   \
        return fallback(x);                           \
    }
#define NUMPY_SVML_F32(name, sym, fallback)          \
    inline float name(float x) {                      \
        static auto fn = (__m512 (*)(__m512))         \
            resolve_svml(sym);                        \
        if (fn) return _mm512_cvtss_f32(              \
            fn(_mm512_set1_ps(x)));                   \
        return fallback(x);                           \
    }

NUMPY_SVML_F64(exp_f64,  "__svml_exp8",  std::exp)
NUMPY_SVML_F64(log_f64,  "__svml_log8",  std::log)
NUMPY_SVML_F64(sin_f64,  "__svml_sin8",  std::sin)
NUMPY_SVML_F64(cos_f64,  "__svml_cos8",  std::cos)
NUMPY_SVML_F64(tan_f64,  "__svml_tan8",  std::tan)
NUMPY_SVML_F64(asin_f64, "__svml_asin8", std::asin)
NUMPY_SVML_F64(acos_f64, "__svml_acos8", std::acos)
NUMPY_SVML_F64(atan_f64, "__svml_atan8", std::atan)
NUMPY_SVML_F64(log10_f64,"__svml_log108",std::log10)
NUMPY_SVML_F64(log2_f64, "__svml_log28", std::log2)
NUMPY_SVML_F64(exp2_f64, "__svml_exp28", std::exp2)

// pow/atan2 — 2-arg scalar (npy_pow / npy_atan2)
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

// f32: exp/log/sin/cos use numpy's own polynomial approximations
inline float exp_f32(float x)  { return npy_float_math::npy_expf(x); }
inline float log_f32(float x)  { return npy_float_math::npy_logf(x); }
inline float sin_f32(float x)  { return npy_float_math::npy_sinf(x); }
inline float cos_f32(float x)  { return npy_float_math::npy_cosf(x); }

NUMPY_SVML_F32(tan_f32,  "__svml_tanf16",  std::tan)
NUMPY_SVML_F32(asin_f32, "__svml_asinf16", std::asin)
NUMPY_SVML_F32(acos_f32, "__svml_acosf16", std::acos)
NUMPY_SVML_F32(atan_f32, "__svml_atanf16", std::atan)
NUMPY_SVML_F32(log10_f32,"__svml_log10f16",std::log10)
NUMPY_SVML_F32(log2_f32, "__svml_log2f16", std::log2)
NUMPY_SVML_F32(exp2_f32, "__svml_exp2f16", std::exp2)

inline float pow_f32(float x, float e)   { return std::pow(x, e); }
inline float atan2_f32(float y, float x) { return std::atan2(y, x); }

#undef NUMPY_SVML_F64
#undef NUMPY_SVML_F32

#else // !__AVX512F__

// Non-AVX512: all 1-arg wrappers degrade to std or npy_math
#define NUMPY_FB_F64(name, fallback) inline double name(double x) { return fallback(x); }
#define NUMPY_FB_F32(name, fallback) inline float name(float x)   { return fallback(x); }

NUMPY_FB_F64(exp_f64,  std::exp)
NUMPY_FB_F64(log_f64,  std::log)
NUMPY_FB_F64(sin_f64,  std::sin)
NUMPY_FB_F64(cos_f64,  std::cos)
NUMPY_FB_F64(tan_f64,  std::tan)
NUMPY_FB_F64(asin_f64, std::asin)
NUMPY_FB_F64(acos_f64, std::acos)
NUMPY_FB_F64(atan_f64, std::atan)
NUMPY_FB_F64(log10_f64,std::log10)
NUMPY_FB_F64(log2_f64, std::log2)
NUMPY_FB_F64(exp2_f64, std::exp2)
inline double pow_f64(double x, double e)   { return std::pow(x, e); }
inline double atan2_f64(double y, double x) { return std::atan2(y, x); }

NUMPY_FB_F32(exp_f32,  npy_float_math::npy_expf)
NUMPY_FB_F32(log_f32,  npy_float_math::npy_logf)
NUMPY_FB_F32(sin_f32,  npy_float_math::npy_sinf)
NUMPY_FB_F32(cos_f32,  npy_float_math::npy_cosf)
NUMPY_FB_F32(tan_f32,  std::tan)
NUMPY_FB_F32(asin_f32, std::asin)
NUMPY_FB_F32(acos_f32, std::acos)
NUMPY_FB_F32(atan_f32, std::atan)
NUMPY_FB_F32(log10_f32,std::log10)
NUMPY_FB_F32(log2_f32, std::log2)
NUMPY_FB_F32(exp2_f32, std::exp2)
inline float pow_f32(float x, float e)   { return std::pow(x, e); }
inline float atan2_f32(float y, float x) { return std::atan2(y, x); }

#undef NUMPY_FB_F64
#undef NUMPY_FB_F32

#endif // __AVX512F__

// ============================================================================
// Template dispatchers — svml_impl<T> + free function templates
// ============================================================================

#define NUMPY_SVML_METHODS(T, suff)                                  \
template<> struct svml_impl<T> {                                     \
    static T exp(T x)  { return exp_##suff(x); }                     \
    static T log(T x)  { return log_##suff(x); }                     \
    static T sin(T x)  { return sin_##suff(x); }                     \
    static T cos(T x)  { return cos_##suff(x); }                     \
    static T tan(T x)  { return tan_##suff(x); }                     \
    static T asin(T x) { return asin_##suff(x); }                    \
    static T acos(T x) { return acos_##suff(x); }                    \
    static T atan(T x) { return atan_##suff(x); }                    \
    static T log10(T x){ return log10_##suff(x); }                   \
    static T log2(T x) { return log2_##suff(x); }                    \
    static T exp2(T x) { return exp2_##suff(x); }                    \
    static T pow(T x, T e)    { return pow_##suff(x, e); }           \
    static T atan2(T y, T x)  { return atan2_##suff(y, x); }         \
};

template<typename T> struct svml_impl;
NUMPY_SVML_METHODS(double, f64)
NUMPY_SVML_METHODS(float,  f32)
#undef NUMPY_SVML_METHODS

// 1-arg dispatchers
#define NUMPY_SVML_D1(name) \
    template<typename T> inline T name(T x) { return svml_impl<T>::name(x); }
NUMPY_SVML_D1(exp)
NUMPY_SVML_D1(log)
NUMPY_SVML_D1(sin)
NUMPY_SVML_D1(cos)
NUMPY_SVML_D1(tan)
NUMPY_SVML_D1(asin)
NUMPY_SVML_D1(acos)
NUMPY_SVML_D1(atan)
NUMPY_SVML_D1(log10)
NUMPY_SVML_D1(log2)
NUMPY_SVML_D1(exp2)
#undef NUMPY_SVML_D1

// 2-arg dispatchers (parameter names differ: pow(x,e) vs atan2(y,x))
template<typename T> inline T pow(T x, T e)    { return svml_impl<T>::pow(x, e); }
template<typename T> inline T atan2(T y, T x)  { return svml_impl<T>::atan2(y, x); }

} // namespace svml
} // namespace numpy
