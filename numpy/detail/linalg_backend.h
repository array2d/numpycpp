// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INTERNAL HEADER — DIRECT INCLUSION IS A COMPILE ERROR                 ║
// ║                                                                          ║
// ║  Compile-time backend selector for linear algebra operations.           ║
// ║                                                                          ║
// ║  NUMPYCPP_STD_ONLY not defined (default):                               ║
// ║    → blas_bridge.h                                                       ║
// ║    → bit-exact with numpy.dot / numpy.matmul (OpenBLAS ILP64, dlsym)   ║
// ║                                                                          ║
// ║  NUMPYCPP_STD_ONLY defined:                                             ║
// ║    → std_linalg_backend.h                                                ║
// ║    → pure C++ loops, auto-vectorised by compiler, 0-2 ULP from numpy   ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#pragma once

#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  error "linalg_backend.h is an internal header — do not include directly. \
Use #include \"numpy/numpy.h\" instead."
#endif

#ifdef NUMPYCPP_STD_ONLY
#  include "std_linalg_backend.h"
#else
#  include "blas_bridge.h"
#endif
