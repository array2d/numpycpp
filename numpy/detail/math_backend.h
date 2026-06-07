// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  Compile-time backend selector for element-wise math functions.         ║
// ║                                                                          ║
// ║  NUMPYCPP_STD_ONLY not defined (default):                               ║
// ║    → svml_bridge.h + npy_math_float.h                                   ║
// ║    → bit-exact with numpy (dlsym, SVML, AVX-512 runtime dispatch)       ║
// ║                                                                          ║
// ║  NUMPYCPP_STD_ONLY defined:                                             ║
// ║    → std_math_backend.h                                                  ║
// ║    → pure <cmath>, 0-2 ULP from numpy, no external dependencies         ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "math_backend.h is an internal header — do not include directly. \
Use #include \"numpy/numpy.h\" instead."
#endif

#ifdef NUMPYCPP_STD_ONLY
#  include "std_math_backend.h"
#else
#  include "npy_math_float.h"
#  include "svml_bridge.h"
#endif
