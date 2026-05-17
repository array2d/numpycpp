// pybind11 module definition for numpycpp.
// Registers double + float32 overloads for every template function.
// All bindings use static_cast to disambiguate wrapper from native overloads in numpy::.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core_py.h"
#include "linalg_py.h"
#include "einsum_py.h"

namespace py = pybind11;

// -- signature-disambiguated binding helpers ----------------------------------

// Unary element-wise: py::array_t<T>(*)(const py::array_t<T>&)
#define BIND_F1(name) \
    m.def(#name, static_cast<py::array_t<double>(*)(const py::array_t<double>&)>(&numpy::name)); \
    m.def(#name, static_cast<py::array_t<float>(*)(const py::array_t<float>&)>(&numpy::name))

// Reduction: T(*)(const py::array_t<T>&)
#define BIND_F_REDUCE(name) \
    m.def(#name, static_cast<double(*)(const py::array_t<double>&)>(&numpy::name)); \
    m.def(#name, static_cast<float(*)(const py::array_t<float>&)>(&numpy::name))

// Reduction returning py::ssize_t
#define BIND_F_REDUCE_PYSSIZE(name) \
    m.def(#name, static_cast<py::ssize_t(*)(const py::array_t<double>&)>(&numpy::name)); \
    m.def(#name, static_cast<py::ssize_t(*)(const py::array_t<float>&)>(&numpy::name))

// 1-arg array transform: py::array_t<T>(*)(const py::array_t<T>&, py::ssize_t)
#define BIND_F_MANIP1(name) \
    m.def(#name, static_cast<py::array_t<double>(*)(const py::array_t<double>&, py::ssize_t)>(&numpy::name)); \
    m.def(#name, static_cast<py::array_t<float>(*)(const py::array_t<float>&, py::ssize_t)>(&numpy::name))

// Stack/concat: py::array_t<T>(*)(const std::vector<py::array_t<T>>&)
#define BIND_F_STACK(name) \
    m.def(#name, static_cast<py::array_t<double>(*)(const std::vector<py::array_t<double>>&)>(&numpy::name)); \
    m.def(#name, static_cast<py::array_t<float>(*)(const std::vector<py::array_t<float>>&)>(&numpy::name))

// ============================================================================

PYBIND11_MODULE(numpycpp, m) {
    m.doc() = "C++ pixel-level alignment of Python numpy, powered by Eigen";

    // -- linalg submodule --------------------------------------------------
    py::module_ la = m.def_submodule("linalg", "numpy.linalg equivalents");
    la.def("norm", static_cast<double(*)(const py::array_t<double>&)>(&numpy::linalg::norm));
    la.def("norm", static_cast<float(*)(const py::array_t<float>&)>(&numpy::linalg::norm));
    la.def("norm_axis1", static_cast<py::array_t<double>(*)(const py::array_t<double>&)>(&numpy::linalg::norm_axis1));
    la.def("norm_axis1", static_cast<py::array_t<double>(*)(const py::array_t<float>&)>(&numpy::linalg::norm_axis1));

    // -- Array creation ----------------------------------------------------
    BIND_F1(zeros_like); BIND_F1(ones_like); BIND_F1(empty_like);
    m.def("full_like", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double)>(&numpy::full_like));
    m.def("full_like", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float)>(&numpy::full_like));
    m.def("full_like_bool", &numpy::full_like_bool);
    m.def("zeros_like_bool", &numpy::zeros_like_bool);
    m.def("ones_like_bool", &numpy::ones_like_bool);
    m.def("zeros", &numpy::zeros);
    m.def("ones", &numpy::ones);

    // -- astype ------------------------------------------------------------
    m.def("astype_int", static_cast<py::array_t<int>(*)(const py::array_t<double>&)>(&numpy::astype_int));
    m.def("astype_bool", static_cast<py::array_t<bool>(*)(const py::array_t<double>&)>(&numpy::astype_bool));
    m.def("astype_bool_from_int", static_cast<py::array_t<bool>(*)(const py::array_t<int>&)>(&numpy::astype_bool_from_int));
    m.def("truncate_to_float32", static_cast<py::array_t<double>(*)(const py::array_t<double>&)>(&numpy::truncate_to_float32));

    // -- Element-wise math -------------------------------------------------
    BIND_F1(sqrt); BIND_F1(abs); BIND_F1(exp); BIND_F1(log);
    BIND_F1(sin); BIND_F1(cos); BIND_F1(tan);
    BIND_F1(log10); BIND_F1(log2); BIND_F1(arcsin); BIND_F1(arccos); BIND_F1(arctan);
    BIND_F1(round); BIND_F1(floor); BIND_F1(ceil);
    BIND_F1(degrees); BIND_F1(radians); BIND_F1(sign);
    m.def("power", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double)>(&numpy::power));
    m.def("power", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float)>(&numpy::power));
    m.def("clip", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double, double)>(&numpy::clip));
    m.def("clip", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float, float)>(&numpy::clip));

    // -- Reduction (single-arg) --------------------------------------------
    BIND_F_REDUCE(sum);
    m.def("mean", static_cast<double(*)(const py::array_t<double>&)>(&numpy::mean));
    m.def("mean", static_cast<float(*)(const py::array_t<float>&)>(&numpy::mean));
    BIND_F_REDUCE(max); BIND_F_REDUCE(min);
    m.def("any", static_cast<bool(*)(const py::array_t<bool>&)>(&numpy::any));
    m.def("all", static_cast<bool(*)(const py::array_t<bool>&)>(&numpy::all));

    // -- mean with axis (always returns double) ----------------------------
    m.def("mean", static_cast<py::array_t<double>(*)(const py::array_t<double>&, int)>(&numpy::mean));
    m.def("mean", static_cast<py::array_t<double>(*)(const py::array_t<float>&,  int)>(&numpy::mean));

    // -- Comparison --------------------------------------------------------
    m.def("greater", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::greater));
    m.def("greater", static_cast<py::array_t<bool>(*)(const py::array_t<float>&, float)>(&numpy::greater));
    m.def("less", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::less));
    m.def("less", static_cast<py::array_t<bool>(*)(const py::array_t<float>&, float)>(&numpy::less));
    m.def("equal", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::equal));
    m.def("equal", static_cast<py::array_t<bool>(*)(const py::array_t<float>&, float)>(&numpy::equal));
    m.def("greater_equal", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::greater_equal));
    m.def("greater_equal", static_cast<py::array_t<bool>(*)(const py::array_t<float>&, float)>(&numpy::greater_equal));
    m.def("less_equal", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::less_equal));
    m.def("less_equal", static_cast<py::array_t<bool>(*)(const py::array_t<float>&, float)>(&numpy::less_equal));

    // -- Logical -----------------------------------------------------------
    m.def("logical_and", static_cast<py::array_t<bool>(*)(const py::array_t<bool>&, const py::array_t<bool>&)>(&numpy::logical_and));
    m.def("logical_or",  static_cast<py::array_t<bool>(*)(const py::array_t<bool>&, const py::array_t<bool>&)>(&numpy::logical_or));
    m.def("logical_not", static_cast<py::array_t<bool>(*)(const py::array_t<bool>&)>(&numpy::logical_not));

    // -- Array access ------------------------------------------------------
    m.def("array_get", static_cast<double(*)(const py::array_t<double>&, py::ssize_t)>(&numpy::array_get));
    m.def("array_get", static_cast<double(*)(const py::array_t<double>&, py::ssize_t, py::ssize_t)>(&numpy::array_get));
    m.def("array_get", static_cast<bool(*)(const py::array_t<bool>&, py::ssize_t)>(&numpy::array_get));
    m.def("array_get", static_cast<double(*)(const py::array&, py::ssize_t)>(&numpy::array_get));

    // -- Dict helpers ------------------------------------------------------
    m.def("get_array", &numpy::get_array);
    m.def("set_array", &numpy::set_array);

    // -- Conversion --------------------------------------------------------
    m.def("asarray", static_cast<py::array_t<double>(*)(const std::vector<double>&)>(&numpy::asarray));
    m.def("asarray", static_cast<py::array_t<double>(*)(const py::array_t<double>&)>(&numpy::asarray));
    m.def("array",   static_cast<py::array_t<double>(*)(const std::vector<double>&)>(&numpy::array));
    m.def("array",   static_cast<py::array_t<double>(*)(const py::array_t<double>&)>(&numpy::array));

    // -- transpose / flatten -----------------------------------------------
    BIND_F1(transpose); BIND_F1(flatten);

    // -- to_vector ---------------------------------------------------------
    m.def("to_vector", static_cast<std::vector<bool>(*)(const py::array_t<bool>&)>(&numpy::to_vector));
    m.def("to_vector", static_cast<std::vector<double>(*)(const py::array_t<double>&)>(&numpy::to_vector));

    // -- Slice helpers (template + non-template overloads) ------------------
    m.def("slice", static_cast<py::array_t<double>(*)(const py::array_t<double>&, py::ssize_t, py::ssize_t)>(&numpy::slice));
    m.def("slice", static_cast<py::array_t<float>(*)(const py::array_t<float>&, py::ssize_t, py::ssize_t)>(&numpy::slice));
    m.def("slice", static_cast<py::array(*)(const py::array&, py::ssize_t, py::ssize_t)>(&numpy::slice));

    // -- Column slice ------------------------------------------------------
    BIND_F_MANIP1(take_cols);

    // -- Slice assignment ---------------------------------------------------
    m.def("slice_assign", static_cast<void(*)(py::array_t<double>, py::ssize_t, double)>(&numpy::slice_assign));
    m.def("slice_assign", static_cast<void(*)(py::array_t<float>,  py::ssize_t, float)>(&numpy::slice_assign));
    m.def("slice_assign", static_cast<void(*)(py::array_t<int>,  py::ssize_t, int)>(&numpy::slice_assign));
    m.def("slice_assign", static_cast<void(*)(py::array_t<bool>, py::ssize_t, bool)>(&numpy::slice_assign));

    // -- Binary element-wise: scalar overloads BEFORE array-array ----------
    m.def("arctan2", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double)>(&numpy::arctan2));
    m.def("arctan2", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float)>(&numpy::arctan2));
    m.def("arctan2", static_cast<py::array_t<double>(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::arctan2));
    m.def("arctan2", static_cast<py::array_t<float>(*)(const py::array_t<float>&, const py::array_t<float>&)>(&numpy::arctan2));

    m.def("maximum", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double)>(&numpy::maximum));
    m.def("maximum", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float)>(&numpy::maximum));
    m.def("maximum", static_cast<py::array_t<double>(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::maximum));
    m.def("maximum", static_cast<py::array_t<float>(*)(const py::array_t<float>&, const py::array_t<float>&)>(&numpy::maximum));

    m.def("minimum", static_cast<py::array_t<double>(*)(const py::array_t<double>&, double)>(&numpy::minimum));
    m.def("minimum", static_cast<py::array_t<float>(*)(const py::array_t<float>&, float)>(&numpy::minimum));
    m.def("minimum", static_cast<py::array_t<double>(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::minimum));
    m.def("minimum", static_cast<py::array_t<float>(*)(const py::array_t<float>&, const py::array_t<float>&)>(&numpy::minimum));

    // -- Comparison helpers ------------------------------------------------
    m.def("not_equal", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, double)>(&numpy::not_equal));
    m.def("not_equal", static_cast<py::array_t<bool>(*)(const py::array_t<float>&,  float)>(&numpy::not_equal));
    m.def("not_equal", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::not_equal));
    m.def("not_equal", static_cast<py::array_t<bool>(*)(const py::array_t<float>&,  const py::array_t<float>&)>(&numpy::not_equal));

    // -- Special value helpers ---------------------------------------------
    m.def("isnan", static_cast<py::array_t<bool>(*)(const py::array_t<double>&)>(&numpy::isnan));
    m.def("isnan", static_cast<py::array_t<bool>(*)(const py::array_t<float>&)>(&numpy::isnan));
    m.def("isinf", static_cast<py::array_t<bool>(*)(const py::array_t<double>&)>(&numpy::isinf));
    m.def("isinf", static_cast<py::array_t<bool>(*)(const py::array_t<float>&)>(&numpy::isinf));
    m.def("isfinite", static_cast<py::array_t<bool>(*)(const py::array_t<double>&)>(&numpy::isfinite));
    m.def("isfinite", static_cast<py::array_t<bool>(*)(const py::array_t<float>&)>(&numpy::isfinite));

    // -- Array manipulation ------------------------------------------------
    m.def("diff", static_cast<py::array_t<double>(*)(const py::array_t<double>&, int, int)>(&numpy::diff));
    m.def("diff", static_cast<py::array_t<float>(*)(const py::array_t<float>&, int, int)>(&numpy::diff),
          py::arg("arr"), py::arg("n") = 1, py::arg("axis") = -1);
    BIND_F_STACK(stack); BIND_F_STACK(concatenate); BIND_F_STACK(vstack); BIND_F_STACK(hstack);

    m.def("where", static_cast<py::array_t<double>(*)(const py::array_t<bool>&, double, double)>(&numpy::where));
    m.def("where", static_cast<py::array_t<float>(*)(const py::array_t<bool>&, float, float)>(&numpy::where));
    m.def("where", static_cast<py::array_t<double>(*)(const py::array_t<bool>&, const py::array_t<double>&, const py::array_t<double>&)>(&numpy::where));
    m.def("where", static_cast<py::array_t<float>(*)(const py::array_t<bool>&, const py::array_t<float>&, const py::array_t<float>&)>(&numpy::where));

    // -- Statistical -------------------------------------------------------
    BIND_F_REDUCE(std); BIND_F_REDUCE(var);

    // -- Set operations ----------------------------------------------------
    m.def("isin", static_cast<py::array_t<bool>(*)(const py::array_t<double>&, const std::vector<double>&)>(&numpy::isin));
    m.def("intersect1d", static_cast<py::array_t<double>(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::intersect1d));

    // -- Interpolation -----------------------------------------------------
    m.def("interp", static_cast<py::array_t<double>(*)(const py::array_t<double>&, const py::array_t<double>&, const py::array_t<double>&)>(&numpy::interp));

    // -- Logical XOR -------------------------------------------------------
    m.def("logical_xor", static_cast<py::array_t<bool>(*)(const py::array_t<bool>&, const py::array_t<bool>&)>(&numpy::logical_xor));

    // -- Sorting and indexing ----------------------------------------------
    m.def("argsort", static_cast<py::array_t<py::ssize_t>(*)(const py::array_t<double>&)>(&numpy::argsort));
    m.def("argsort", static_cast<py::array_t<py::ssize_t>(*)(const py::array_t<float>&)>(&numpy::argsort));
    BIND_F_REDUCE_PYSSIZE(argmax); BIND_F_REDUCE_PYSSIZE(argmin);

    // -- Array transformation ----------------------------------------------
    BIND_F_MANIP1(roll); BIND_F1(flip); BIND_F_MANIP1(repeat); BIND_F_MANIP1(tile);

    // -- Safe division -----------------------------------------------------
    m.def("safe_divide", &numpy::safe_divide, py::arg("a"), py::arg("b"), py::arg("default_val") = 0.0);

    // -- Dot product -------------------------------------------------------
    m.def("dot", static_cast<double(*)(const py::array_t<double>&, const py::array_t<double>&)>(&numpy::dot));
    m.def("dot", static_cast<float(*)(const py::array_t<float>&, const py::array_t<float>&)>(&numpy::dot));

    // -- Einsum ------------------------------------------------------------
    m.def("einsum", static_cast<py::array_t<double>(*)(const std::string&, const py::array_t<double>&, const py::array_t<double>&)>(&numpy::einsum));
    m.def("einsum", static_cast<py::array_t<float>(*)(const std::string&, const py::array_t<float>&, const py::array_t<float>&)>(&numpy::einsum));
}
