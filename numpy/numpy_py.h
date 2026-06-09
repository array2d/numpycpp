// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpy/numpy_py.h                           [PUBLIC HEADER]
//
//  Single umbrella header — includes all pybind11 wrapper modules.
//
//  Usage:
//      #include "numpy/numpy_py.h"
//
//  Prerequisite:
//      #include <pybind11/pybind11.h>
//      #include <pybind11/stl.h>
//
//  Or include individual wrapper modules as needed:
//      #include "numpy/init_py.h"         // zeros_like, ones_like, full, …
//      #include "numpy/elementwise_py.h"  // sqrt, exp, sin, astype, …
//      #include "numpy/reduce_py.h"       // sum, mean, std, var, cumsum, …
//      #include "numpy/manipulation_py.h" // transpose, take, slice, put, …
//      #include "numpy/io_py.h"           // isin, interp, unwrap, asarray, …
//      #include "numpy/linalg_py.h"       // dot, norm, matmul, einsum
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include "init_py.h"
#include "elementwise_py.h"
#include "reduce_py.h"
#include "manipulation_py.h"
#include "io_py.h"
#include "linalg_py.h"
