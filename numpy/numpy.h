// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/numpy.h                             [PUBLIC HEADER]
//
//  Single umbrella header — includes all numpycpp modules.
//
//  Usage:
//      #include "numpy/numpy.h"
//
//  Or include individual modules as needed:
//      #include "numpy/init.h"         // zeros_like, ones_like, full
//      #include "numpy/elementwise.h"  // sqrt, exp, sin, astype, …
//      #include "numpy/reduce.h"       // sum, mean, std, var, …
//      #include "numpy/manipulation.h" // transpose, take, slice, putmask, …
//      #include "numpy/io.h"           // isin, interp, unwrap, …
//      #include "numpy/linalg.h"       // dot, norm, matmul, einsum
//
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include "init.h"
#include "elementwise.h"
#include "reduce.h"
#include "manipulation.h"
#include "io.h"
#include "linalg.h"
