// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  Pure C++ standard-library math backend.                                ║
// ║  Provides the same numpy::detail API as svml_bridge.h but uses          ║
// ║  only <cmath> — no dlopen, no SVML, no AVX-512, no numpy dependency.   ║
// ║                                                                          ║
// ║  ✗  #include "numpy/detail/std_math_backend.h"  ← compile error        ║
// ║  ✓  #include "numpy/numpy.h"                    ← entry point           ║
// ║  ✓  cmake … -DNUMPYCPP_STD_ONLY=ON             ← selects this backend  ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// Precision notes (NUMPYCPP_STD_ONLY=ON):
//   • Transcendental accuracy: glibc <cmath> — typically 0–1 ULP from true
//     value but NOT bit-identical to numpy (numpy uses its own npy_* kernels).
//   • FMA contracts are allowed (no -ffp-contract=off required) — the
//     compiler can fuse multiply-add for free performance.
//   • With -O3 -march=native the compiler auto-vectorises the loops in
//     elementwise.h; no explicit AVX-512 intrinsics needed.
//   • sin(±0) / cos(±0) / atan2(-0,x): glibc matches IEEE 754 sign rules
//     directly — no manual fixup needed.

#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "std_math_backend.h is an internal header — do not include directly. \
Use #include \"numpy/numpy.h\" instead."
#endif

#include <cmath>

namespace numpy {
namespace detail {

// ============================================================================
// svml_impl<T> — same template interface as svml_bridge.h.
// elementwise.h calls numpy::detail::exp(x) etc. via these specialisations.
// ============================================================================

template<typename T> struct svml_impl;

template<> struct svml_impl<double> {
    static double exp  (double x) { return std::exp(x);   }
    static double log  (double x) { return std::log(x);   }
    static double sin  (double x) { return std::sin(x);   }
    static double cos  (double x) { return std::cos(x);   }
    static double tan  (double x) { return std::tan(x);   }
    static double asin (double x) { return std::asin(x);  }
    static double acos (double x) { return std::acos(x);  }
    static double atan (double x) { return std::atan(x);  }
    static double log10(double x) { return std::log10(x); }
    static double log2 (double x) { return std::log2(x);  }
    static double exp2 (double x) { return std::exp2(x);  }
    static double cbrt (double x) { return std::cbrt(x);  }
    static double expm1(double x) { return std::expm1(x); }
    static double log1p(double x) { return std::log1p(x); }
    static double sqrt (double x) { return std::sqrt(x);  }
    static double pow  (double x, double e) { return std::pow(x, e);   }
    static double atan2(double y, double x) { return std::atan2(y, x); }
    static double hypot(double x, double y) { return std::hypot(x, y); }
};

template<> struct svml_impl<float> {
    static float exp  (float x) { return std::exp(x);   }
    static float log  (float x) { return std::log(x);   }
    static float sin  (float x) { return std::sin(x);   }
    static float cos  (float x) { return std::cos(x);   }
    static float tan  (float x) { return std::tan(x);   }
    static float asin (float x) { return std::asin(x);  }
    static float acos (float x) { return std::acos(x);  }
    static float atan (float x) { return std::atan(x);  }
    static float log10(float x) { return std::log10(x); }
    static float log2 (float x) { return std::log2(x);  }
    static float exp2 (float x) { return std::exp2(x);  }
    static float cbrt (float x) { return std::cbrt(x);  }
    static float expm1(float x) { return std::expm1(x); }
    static float log1p(float x) { return std::log1p(x); }
    static float sqrt (float x) { return std::sqrt(x);  }
    static float pow  (float x, float e) { return std::pow(x, e);   }
    static float atan2(float y, float x) { return std::atan2(y, x); }
    static float hypot(float x, float y) { return std::hypot(x, y); }
};

// ============================================================================
// Free function templates — identical signature to svml_bridge.h exports.
// elementwise.h calls numpy::detail::exp(x), numpy::detail::pow(x,e), etc.
// ============================================================================

#define NUMPY_STD_D1(name) \
    template<typename T> inline T name(T x) { return svml_impl<T>::name(x); }
NUMPY_STD_D1(exp)
NUMPY_STD_D1(log)
NUMPY_STD_D1(sin)
NUMPY_STD_D1(cos)
NUMPY_STD_D1(tan)
NUMPY_STD_D1(asin)
NUMPY_STD_D1(acos)
NUMPY_STD_D1(atan)
NUMPY_STD_D1(log10)
NUMPY_STD_D1(log2)
NUMPY_STD_D1(exp2)
NUMPY_STD_D1(cbrt)
NUMPY_STD_D1(expm1)
NUMPY_STD_D1(log1p)
NUMPY_STD_D1(sqrt)
#undef NUMPY_STD_D1

template<typename T> inline T pow  (T x, T e)   { return svml_impl<T>::pow(x, e);   }
template<typename T> inline T atan2(T y, T x)   { return svml_impl<T>::atan2(y, x); }
template<typename T> inline T hypot(T x, T y)   { return svml_impl<T>::hypot(x, y); }

} // namespace detail
} // namespace numpy
