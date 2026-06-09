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

#include "init.h"
#include "elementwise.h"
#include "reduce.h"
#include "manipulation.h"
#include "io.h"
#include "linalg.h"
