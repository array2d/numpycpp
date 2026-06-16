// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpycpp/numpy.h                             [PUBLIC HEADER]
//
//  Single umbrella header — includes all numpycpp C++ core modules.
//
//  Usage:
//      #include <numpycpp/numpy.h>
//
//  Or include individual modules as needed:
//      #include <numpycpp/numpycpp/init.h>         // zeros_like, ones_like, full
//      #include <numpycpp/numpycpp/elementwise.h>  // sqrt, exp, sin, astype, …
//      #include <numpycpp/numpycpp/reduce.h>       // sum, mean, std, var, …
//      #include <numpycpp/numpycpp/manipulation.h> // transpose, take, slice, put
//      #include <numpycpp/numpycpp/io.h>           // isin, interp, unwrap, …
//      #include <numpycpp/numpycpp/linalg.h>       // dot, norm, matmul, einsum
//
// ════════════════════════════════════════════════════════════════════════════
#pragma once

// Internal math backend — loaded first so init.h et al. can use detail::pow etc.
#ifndef NUMPYCPP_INTERNAL_INCLUDE
#  define NUMPYCPP_INTERNAL_INCLUDE
#  ifdef NUMPYCPP_STD_ONLY
#    include "detail/std_math_backend.h"
#  else
#    include "detail/npy_math_float.h"
#    include "detail/svml_bridge.h"
#  endif
#  undef NUMPYCPP_INTERNAL_INCLUDE
#endif

#include "init.h"
#include "elementwise.h"
#include "reduce.h"
#include "manipulation.h"
#include "io.h"
#include "linalg.h"
