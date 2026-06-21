// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  This file bridges numpycpp to numpy's SVML / npy_* scalar kernels.    ║
// ║  It is x86_64 + Linux specific (dlsym, /proc/self/maps, AVX-512).      ║
// ║  All symbols live in numpy::detail — an implementation namespace that   ║
// ║  external code must never reference.                                     ║
// ║                                                                          ║
// ║  ✗  #include "numpy/detail/svml_bridge.h"      ← compile error                ║
// ║  ✗  numpy::detail::exp_svml_f64(x)      ← undefined behaviour          ║
// ║  ✓  #include "numpycpp/numpy.h"             ← recommended entry point      ║
// ║  ✓  numpy::exp(src, dst, n)             ← public API                    ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// SVML/npy bridge — bit-exact math on every x86_64 architecture.
//   AVX-512 HW → __svml_exp8 (SVML vector) → resolved via dlsym
//   non-AVX-512 → npy_exp (scalar)         → resolved via dlsym
// CPU feature detection is at RUNTIME; AVX-512 intrinsics are isolated behind
// __attribute__((target("avx512f"))) — safe on non-AVX-512 CPUs (no SIGILL).
// The .so path is auto-discovered via /proc/self/maps — no manual init needed.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "svml_bridge.h is an internal header — do not include directly. \
Use #include "numpycpp/numpy.h" instead."
#endif

#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include "npy_math_float.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace numpy {
namespace detail {
// Internal dispatch namespace — use numpy::exp() etc., not numpy::detail::exp().
//
// fork 安全设计:
//   g_svml_gen 是全局代计数器，fork 后自增，所有 resolve_cached<T>() 检测到
//   代变化后自动重新 dlsym。spawn 模式一切从零初始化，无问题。
//   static 缓存函数指针避免每元素 dlsym 开销。
//
//   find_umath_path() 保持无 static — 每次调用重新扫描 /proc/self/maps。

#include <unistd.h>  // getpid

inline void* g_svml_handle = nullptr;
inline pid_t g_svml_pid = 0;
inline int  g_svml_gen  = 0;    // fork 检测代计数器

/// 每次调用重新扫描 /proc/self/maps（无 static 缓存）
inline std::string find_umath_path() {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("_multiarray_umath") != std::string::npos &&
            line.find(".so") != std::string::npos) {
            auto pos = line.rfind('/');
            auto start = line.rfind(' ', pos);
            if (start != std::string::npos && pos != std::string::npos)
                return line.substr(start + 1);
        }
    }
    return "";
}

inline void bridge_init(const char* numpy_so_path) {
    (void)numpy_so_path;
}

/// 解析原始符号（无缓存）。fork 检测: pid 变化 → handle 作废 → 重建。
inline void* resolve_svml(const char* name) {
    pid_t pid = getpid();
    if (pid != g_svml_pid) {
        g_svml_handle = nullptr;
        g_svml_pid = pid;
        g_svml_gen++;              // 所有 resolve_cached 缓存失效
    }
    if (!g_svml_handle) {
        std::string path = find_umath_path();
        if (!path.empty())
            g_svml_handle = dlopen(path.c_str(), RTLD_NOLOAD | RTLD_LAZY);
    }
    if (g_svml_handle) return dlsym(g_svml_handle, name);
    return nullptr;
}

/// 带代缓存的符号解析。static 缓存 + fork 后自动重解析。
template<typename FnSig>
inline FnSig resolve_cached(const char* sym) {
    static FnSig fn = nullptr;
    static int gen = 0;
    if (gen != g_svml_gen) {
        fn = reinterpret_cast<FnSig>(resolve_svml(sym));
        gen = g_svml_gen;
    }
    return fn;
}

// ============================================================================
// Runtime CPU detection
// ============================================================================

inline bool cpu_has_avx512f() {
#ifdef __AVX512F__
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

// ============================================================================
// AVX-512 SVML path — isolated behind target attribute to prevent SIGILL
// on non-AVX-512 hardware.
// ============================================================================

#ifdef __AVX512F__

#define NUMPY_SVML_F64(name, svml_sym, npy_sym)                         \
    __attribute__((target("avx512f")))                                   \
    inline double name##_svml_f64(double x) {                            \
        static auto fn = (__m512d(*)(__m512d))nullptr;                   \
        static int _gen = -1;                                             \
        if (_gen != g_svml_gen) { fn = (__m512d(*)(__m512d))resolve_svml(svml_sym); _gen = g_svml_gen; } \
        if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x)));         \
        return std::name(x);                                             \
    }

#define NUMPY_SVML_F32(name, svml_sym, npy_sym)                         \
    __attribute__((target("avx512f")))                                   \
    inline float name##_svml_f32(float x) {                              \
        static auto fn = (__m512(*)(__m512))nullptr;                     \
        static int _gen = -1;                                             \
        if (_gen != g_svml_gen) { fn = (__m512(*)(__m512))resolve_svml(svml_sym); _gen = g_svml_gen; } \
        if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x)));         \
        return std::name(x);                                             \
    }

NUMPY_SVML_F64(exp,   "__svml_exp8",   "npy_exp")
NUMPY_SVML_F64(log,   "__svml_log8",   "npy_log")
NUMPY_SVML_F64(sin,   "__svml_sin8",   "npy_sin")
NUMPY_SVML_F64(cos,   "__svml_cos8",   "npy_cos")
NUMPY_SVML_F64(tan,   "__svml_tan8",   "npy_tan")
NUMPY_SVML_F64(asin,  "__svml_asin8",  "npy_asin")
NUMPY_SVML_F64(acos,  "__svml_acos8",  "npy_acos")
NUMPY_SVML_F64(atan,  "__svml_atan8",  "npy_atan")
NUMPY_SVML_F64(log10, "__svml_log108", "npy_log10")
NUMPY_SVML_F64(log2,  "__svml_log28",  "npy_log2")
NUMPY_SVML_F64(exp2,  "__svml_exp28",  "npy_exp2")
NUMPY_SVML_F64(cbrt,  "__svml_cbrt8",  "npy_cbrt")
NUMPY_SVML_F64(expm1, "__svml_expm18", "npy_expm1")
NUMPY_SVML_F64(log1p, "__svml_log1p8", "npy_log1p")

NUMPY_SVML_F32(tan,   "__svml_tanf16",  "npy_tanf")
NUMPY_SVML_F32(asin,  "__svml_asinf16", "npy_asinf")
NUMPY_SVML_F32(acos,  "__svml_acosf16", "npy_acosf")
NUMPY_SVML_F32(atan,  "__svml_atanf16", "npy_atanf")
NUMPY_SVML_F32(log10, "__svml_log10f16","npy_log10f")
NUMPY_SVML_F32(log2,  "__svml_log2f16", "npy_log2f")
NUMPY_SVML_F32(exp2,  "__svml_exp2f16", "npy_exp2f")
NUMPY_SVML_F32(cbrt,  "__svml_cbrtf16", "npy_cbrtf")
NUMPY_SVML_F32(expm1, "__svml_expm1f16","npy_expm1f")
NUMPY_SVML_F32(log1p, "__svml_log1pf16","npy_log1pf")

// pow / atan2 — SVML 2-arg: 使用 __svml_pow8 / __svml_atan28 向量符号，
// 广播标量到 __m512，调用 SVML 向量函数，提取结果。确保与 numpy 内部使用的
// SVML 实现位级一致（npy_pow / npy_atan2 是标量 libm 回退，会差 1 ULP）。
__attribute__((target("avx512f")))
inline double pow_svml_f64(double x, double e) {
    static auto fn = (__m512d(*)(__m512d,__m512d))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (__m512d(*)(__m512d,__m512d))resolve_svml("__svml_pow8"); _gen = g_svml_gen; }
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(x), _mm512_set1_pd(e)));
    return std::pow(x, e);
}
__attribute__((target("avx512f")))
inline float pow_svml_f32(float x, float e) {
    static auto fn = (__m512(*)(__m512,__m512))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (__m512(*)(__m512,__m512))resolve_svml("__svml_powf16"); _gen = g_svml_gen; }
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(x), _mm512_set1_ps(e)));
    return std::pow(x, e);
}
__attribute__((target("avx512f")))
inline double atan2_svml_f64(double y, double x) {
    static auto fn = (__m512d(*)(__m512d,__m512d))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (__m512d(*)(__m512d,__m512d))resolve_svml("__svml_atan28"); _gen = g_svml_gen; }
    if (fn) return _mm512_cvtsd_f64(fn(_mm512_set1_pd(y), _mm512_set1_pd(x)));
    return std::atan2(y, x);
}
__attribute__((target("avx512f")))
inline float atan2_svml_f32(float y, float x) {
    static auto fn = (__m512(*)(__m512,__m512))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (__m512(*)(__m512,__m512))resolve_svml("__svml_atan2f16"); _gen = g_svml_gen; }
    if (fn) return _mm512_cvtss_f32(fn(_mm512_set1_ps(y), _mm512_set1_ps(x)));
    return std::atan2(y, x);
}

#undef NUMPY_SVML_F64
#undef NUMPY_SVML_F32

#endif // __AVX512F__

// ============================================================================
// Scalar npy_* path — used when AVX-512 not available at runtime.
// Resolves numpy's scalar math functions (npy_exp, npy_log, ...) via dlsym.
// ============================================================================

#define NUMPY_NPY_F64(name, fallback_expr)                             \
    inline double name##_npy_f64(double x) {                            \
        static auto fn = (double(*)(double))nullptr;                    \
        static int _gen = -1;                                            \
        if (_gen != g_svml_gen) {                                       \
            fn = (double(*)(double))resolve_svml("npy_" #name);         \
            _gen = g_svml_gen;                                          \
        }                                                              \
        if (fn) return fn(x);                                           \
        return (fallback_expr);                                         \
    }

#define NUMPY_NPY_F32(name, fallback_expr)                              \
    inline float name##_npy_f32(float x) {                              \
        static auto fn = (float(*)(float))nullptr;                      \
        static int _gen = -1;                                            \
        if (_gen != g_svml_gen) {                                       \
            fn = (float(*)(float))resolve_svml("npy_" #name "f");       \
            _gen = g_svml_gen;                                          \
        }                                                              \
        if (fn) return fn(x);                                           \
        return (fallback_expr);                                         \
    }

NUMPY_NPY_F64(exp,   std::exp(x))
NUMPY_NPY_F64(log,   std::log(x))
NUMPY_NPY_F64(sin,   std::sin(x))
NUMPY_NPY_F64(cos,   std::cos(x))
NUMPY_NPY_F64(tan,   std::tan(x))
NUMPY_NPY_F64(asin,  std::asin(x))
NUMPY_NPY_F64(acos,  std::acos(x))
NUMPY_NPY_F64(atan,  std::atan(x))
NUMPY_NPY_F64(log10, std::log10(x))
NUMPY_NPY_F64(log2,  std::log2(x))
NUMPY_NPY_F64(exp2,  std::exp2(x))
NUMPY_NPY_F64(cbrt,  std::cbrt(x))
NUMPY_NPY_F64(expm1, std::expm1(x))
NUMPY_NPY_F64(log1p, std::log1p(x))

// f32: fallback via numpy's own polynomial approximations
// f32 exp/log/sin/cos: numpy's own polynomial approximations (npy_math_float.h)
// are the canonical implementation for float32. dlsym npy_expf/etc. would
// resolve a different (non-matching) code path. Never use dlsym for these.
inline float exp_npy_f32(float x)  { return npy_expf(x); }
inline float log_npy_f32(float x)  { return npy_logf(x); }
inline float sin_npy_f32(float x)  { return npy_sinf(x); }
inline float cos_npy_f32(float x)  { return npy_cosf(x); }
NUMPY_NPY_F32(tan,   std::tan(x))
NUMPY_NPY_F32(asin,  std::asin(x))
NUMPY_NPY_F32(acos,  std::acos(x))
NUMPY_NPY_F32(atan,  std::atan(x))
NUMPY_NPY_F32(log10, std::log10(x))
NUMPY_NPY_F32(log2,  std::log2(x))
NUMPY_NPY_F32(exp2,  std::exp2(x))
NUMPY_NPY_F32(cbrt,  std::cbrt(x))
NUMPY_NPY_F32(expm1, std::expm1(x))
NUMPY_NPY_F32(log1p, std::log1p(x))

// hypot — numpy matches libm bit-exact for both f32 and f64
inline double hypot_f64(double x, double y) { return std::hypot(x, y); }
inline float  hypot_f32(float x, float y)   { return std::hypot(x, y); }

inline double pow_npy_f64(double x, double e) {
    static auto fn = (double(*)(double,double))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (double(*)(double,double))resolve_svml("npy_pow"); _gen = g_svml_gen; }
    if (fn) return fn(x, e);
    return std::pow(x, e);
}
inline float pow_npy_f32(float x, float e) {
    static auto fn = (float(*)(float,float))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (float(*)(float,float))resolve_svml("npy_powf"); _gen = g_svml_gen; }
    if (fn) return fn(x, e);
    return std::pow(x, e);
}
inline double atan2_npy_f64(double y, double x) {
    static auto fn = (double(*)(double,double))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (double(*)(double,double))resolve_svml("npy_atan2"); _gen = g_svml_gen; }
    if (fn) return fn(y, x);
    return std::atan2(y, x);
}
inline float atan2_npy_f32(float y, float x) {
    static auto fn = (float(*)(float,float))nullptr;
    static int _gen = -1;
    if (_gen != g_svml_gen) { fn = (float(*)(float,float))resolve_svml("npy_atan2f"); _gen = g_svml_gen; }
    if (fn) return fn(y, x);
    return std::atan2(y, x);
}

#undef NUMPY_NPY_F64
#undef NUMPY_NPY_F32

// ============================================================================
// Dispatchers — select SVML (AVX-512) or npy_* (scalar) at runtime.
//
// SVML call sites are guarded by #ifdef __AVX512F__ so the dispatchers
// compile cleanly when -mavx512f is absent (e.g. cloud CI runners where
// CPUID reports avx512f but the hypervisor traps ZMM execution).
// Without -mavx512f: always use the scalar npy_* path — still bit-exact.
// ============================================================================

// 1-arg: SVML 优先 — __svml_exp8 等在非 AVX-512 上也是位级一致 (1000/1000)。
// 2-arg: npy_* 优先 — pow_f64/atan2_f64 下方单独定义。
#ifdef __AVX512F__
#define DISPATCH_F64(name) \
    inline double name##_f64(double x) { \
        if (cpu_has_avx512f()) return name##_svml_f64(x); \
        return name##_npy_f64(x); }
#define DISPATCH_F32(name) \
    inline float name##_f32(float x) { \
        if (cpu_has_avx512f()) return name##_svml_f32(x); \
        return name##_npy_f32(x); }
#else
#define DISPATCH_F64(name) \
    inline double name##_f64(double x) { return name##_npy_f64(x); }
#define DISPATCH_F32(name) \
    inline float name##_f32(float x)  { return name##_npy_f32(x); }
#endif

DISPATCH_F64(exp)
DISPATCH_F64(log)
// sin_f64: custom — SVML scalar broadcast path loses signed zero (sin(-0)→+0).
// IEEE 754 requires sin(±0) = ±0; preserve sign of zero explicitly.
inline double sin_f64(double x) {
#ifdef __AVX512F__
    double r = cpu_has_avx512f() ? sin_svml_f64(x) : sin_npy_f64(x);
#else
    double r = sin_npy_f64(x);
#endif
    if (__builtin_expect(x == 0.0 && r == 0.0, 0)) return x;
    return r;
}
DISPATCH_F64(cos)
DISPATCH_F64(tan)
DISPATCH_F64(asin)
DISPATCH_F64(acos)
DISPATCH_F64(atan)
DISPATCH_F64(log10)
DISPATCH_F64(log2)
DISPATCH_F64(exp2)
DISPATCH_F64(cbrt)
DISPATCH_F64(expm1)
DISPATCH_F64(log1p)
DISPATCH_F32(tan)
DISPATCH_F32(asin)
DISPATCH_F32(acos)
DISPATCH_F32(atan)
DISPATCH_F32(log10)
DISPATCH_F32(log2)
DISPATCH_F32(exp2)
DISPATCH_F32(cbrt)
DISPATCH_F32(expm1)
DISPATCH_F32(log1p)

// f32 exp/log/sin/cos: numpy uses its own polynomial approximations
// (npy_math_float.h), NOT SVML. These are bit-exact on all architectures.
inline float exp_f32(float x)  { return exp_npy_f32(x); }
inline float log_f32(float x)  { return log_npy_f32(x); }
// sin_f32: npy_sinf polynomial computes fma(sp,r,r) with r=±0 → +0 (IEEE RN rule),
// losing the sign.  Restore: IEEE 754 mandates sin(±0) = ±0.
inline float sin_f32(float x) {
    float r = sin_npy_f32(x);
    if (__builtin_expect(x == 0.0f && r == 0.0f, 0)) return x;  // sin(±0)=±0
    return r;
}
inline float cos_f32(float x)  { return cos_npy_f32(x); }

// pow / atan2 dispatchers
inline double pow_f64(double x, double e) {
    return pow_npy_f64(x, e);  // npy_pow 已验证位级一致
}
inline float pow_f32(float x, float e) {
    return pow_npy_f32(x, e);
}
inline double atan2_f64(double y, double x) {
    return atan2_npy_f64(y, x);  // npy_atan2 已验证位级一致，SVML broadcast 有 1 ULP 差
}
inline float atan2_f32(float y, float x) {
    return atan2_npy_f32(y, x);
}

// sqrt — always std::sqrt (numpy's sqrt matches IEEE 754)
inline double sqrt_f64(double x) { return std::sqrt(x); }
inline float  sqrt_f32(float x)  { return std::sqrt(x); }

#undef DISPATCH_F64
#undef DISPATCH_F32

// ============================================================================
// 1-arg dispatchers — inline overloads, 零 static, 零 template struct
// ============================================================================

#define NUMPY_D1(name) \
    inline double name(double x) { return name##_f64(x); } \
    inline float  name(float x)  { return name##_f32(x); }

NUMPY_D1(exp)
NUMPY_D1(log)
NUMPY_D1(sin)
NUMPY_D1(cos)
NUMPY_D1(tan)
NUMPY_D1(asin)
NUMPY_D1(acos)
NUMPY_D1(atan)
NUMPY_D1(log10)
NUMPY_D1(log2)
NUMPY_D1(exp2)
NUMPY_D1(cbrt)
NUMPY_D1(expm1)
NUMPY_D1(log1p)
NUMPY_D1(sqrt)
#undef NUMPY_D1

// 2-arg dispatchers
inline double pow(double x, double e)   { return pow_f64(x, e); }
inline float  pow(float x, float e)     { return pow_f32(x, e); }
inline double atan2(double y, double x) { return atan2_f64(y, x); }
inline float  atan2(float y, float x)   { return atan2_f32(y, x); }
inline double hypot(double x, double y) { return hypot_f64(x, y); }
inline float  hypot(float x, float y)   { return hypot_f32(x, y); }

} // namespace detail
} // namespace numpy
