// ════════════════════════════════════════════════════════════════════════════
//  numpycpp — numpycpp/numpy_py.h                         [PUBLIC HEADER]
//
//  Single umbrella header — includes all pybind11 wrapper modules.
//
//  Usage:
//      #include <numpycpp/numpy_py.h>
//
//  Prerequisite:
//      #include <pybind11/pybind11.h>
//      #include <pybind11/stl.h>
//
//  Or include individual wrapper modules as needed:
//      #include <numpycpp/numpycpp/init_py.h>
//      #include <numpycpp/numpycpp/elementwise_py.h>
//      #include <numpycpp/numpycpp/reduce_py.h>
//      #include <numpycpp/numpycpp/manipulation_py.h>
//      #include <numpycpp/numpycpp/io_py.h>
//      #include <numpycpp/numpycpp/linalg_py.h>
// ════════════════════════════════════════════════════════════════════════════
#pragma once

#include "init_py.h"
#include "elementwise_py.h"
#include "reduce_py.h"
#include "manipulation_py.h"
#include "io_py.h"
#include "linalg_py.h"
