// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — pycpp/pycpp.h                             [PUBLIC HEADER]
//  Umbrella header: includes all pybind11 wrapper modules.
//
//  Include this single header to get every numpycpp pybind11 wrapper:
//      #include "pycpp/pycpp.h"
//
//  Or include individual modules as needed:
//      #include "pycpp/init_py.h"        // zeros_like, ones_like, full, …
//      #include "pycpp/elementwise_py.h" // sqrt, exp, sin, astype, …
//      #include "pycpp/reduce_py.h"      // sum, mean, std, var, cumsum, …
//      #include "pycpp/manipulation_py.h"// transpose, take, slice, put, …
//      #include "pycpp/io_py.h"          // isin, interp, unwrap, asarray, …
//      #include "pycpp/linalg_py.h"      // dot, norm, matmul, einsum
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include "init_py.h"
#include "elementwise_py.h"
#include "reduce_py.h"
#include "manipulation_py.h"
#include "io_py.h"
#include "linalg_py.h"
