// Python Source: numpy/core/einsumfunc.py (numpy.einsum)
// Line Range: numpy.einsum public API
// Alignment: strict

#include "numpy/einsum.h"
#include <stdexcept>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace numpy {

namespace {

// Parse subscript string into input label strings and output label string.
// "ij,ij->i" → inputs=["ij","ij"], output="i"
// "ij,ij"    → inputs=["ij","ij"], output="" (implicit mode)
struct ParsedSubscripts {
    std::vector<std::string> inputs;
    std::string output;
    bool explicit_mode;
};

ParsedSubscripts parse_subscripts(const std::string& subscripts) {
    ParsedSubscripts result;
    result.explicit_mode = false;

    auto arrow_pos = subscripts.find("->");
    std::string lhs;
    if (arrow_pos != std::string::npos) {
        lhs = subscripts.substr(0, arrow_pos);
        result.output = subscripts.substr(arrow_pos + 2);
        result.explicit_mode = true;
    } else {
        lhs = subscripts;
    }

    // Split lhs on ','
    size_t start = 0;
    while (start <= lhs.size()) {
        auto comma = lhs.find(',', start);
        if (comma == std::string::npos) {
            result.inputs.push_back(lhs.substr(start));
            break;
        }
        result.inputs.push_back(lhs.substr(start, comma - start));
        start = comma + 1;
    }

    return result;
}

// Compute strides for an array shape (row-major).
std::vector<py::ssize_t> compute_strides(const std::vector<py::ssize_t>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<py::ssize_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Compute flat index from multi-dimensional coordinate and strides.
py::ssize_t flat_index(const std::vector<py::ssize_t>& coord,
                       const std::vector<py::ssize_t>& strides) {
    py::ssize_t idx = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
        idx += coord[i] * strides[i];
    }
    return idx;
}

// Determine implicit output labels: labels that appear in exactly one input.
std::string implicit_output_labels(const std::vector<std::string>& input_labels) {
    std::map<char, int> label_count;
    for (const auto& s : input_labels) {
        std::set<char> seen;
        for (char c : s) {
            if (seen.insert(c).second) {
                label_count[c]++;
            }
        }
    }
    std::string output;
    for (const auto& [c, count] : label_count) {
        if (count == 1) {
            output += c;
        }
    }
    std::sort(output.begin(), output.end());
    return output;
}

} // anonymous namespace

py::array_t<double> einsum(const std::string& subscripts,
                           const py::array_t<double>& a,
                           const py::array_t<double>& b) {
    // ---- 1. Parse subscripts ----
    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    std::string out_labels = parsed.output;

    if (in_labels.size() != 2) {
        throw std::invalid_argument("einsum: currently only supports 2 operands, got " +
                                    std::to_string(in_labels.size()));
    }

    // Implicit mode: compute output labels
    if (!parsed.explicit_mode) {
        out_labels = implicit_output_labels(in_labels);
    }

    // ---- 2. Gather input arrays and shapes ----
    const py::array_t<double>* arrays[2] = {&a, &b};
    auto bufa = a.request();
    auto bufb = b.request();
    std::vector<py::ssize_t> shapes[2] = {
        std::vector<py::ssize_t>(bufa.shape),
        std::vector<py::ssize_t>(bufb.shape)
    };
    int ndim[2] = {static_cast<int>(shapes[0].size()),
                   static_cast<int>(shapes[1].size())};

    // Verify label count matches ndim
    for (int i = 0; i < 2; ++i) {
        if (static_cast<int>(in_labels[i].size()) != ndim[i]) {
            throw std::invalid_argument(
                "einsum: subscript '" + in_labels[i] + "' has " +
                std::to_string(in_labels[i].size()) + " labels but operand " +
                std::to_string(i) + " has " + std::to_string(ndim[i]) + " dimensions");
        }
    }

    // ---- 3. Build label → (input_idx → axis) mapping and label sizes ----
    // label_axis[l] = [(0, axis_in_a), (1, axis_in_b), ...]
    std::map<char, std::vector<std::pair<int, int>>> label_axis;
    std::map<char, py::ssize_t> label_size;

    for (int i = 0; i < 2; ++i) {
        for (int ax = 0; ax < ndim[i]; ++ax) {
            char l = in_labels[i][ax];
            label_axis[l].push_back({i, ax});
            py::ssize_t sz = shapes[i][ax];
            auto it = label_size.find(l);
            if (it == label_size.end()) {
                label_size[l] = sz;
            } else if (it->second != sz) {
                throw std::invalid_argument(
                    "einsum: label '" + std::string(1, l) +
                    "' has inconsistent sizes: " + std::to_string(it->second) +
                    " vs " + std::to_string(sz));
            }
        }
    }

    // ---- 4. Classify labels ----
    // Output labels: those in out_labels string
    std::set<char> output_label_set(out_labels.begin(), out_labels.end());

    // All unique labels
    std::set<char> all_labels;
    for (const auto& s : in_labels) {
        for (char c : s) all_labels.insert(c);
    }

    // Summation labels: in inputs but NOT in output
    std::vector<char> sum_labels;
    for (char c : all_labels) {
        if (!output_label_set.count(c)) {
            sum_labels.push_back(c);
        }
    }

    // Output labels in order
    std::vector<char> output_labels(out_labels.begin(), out_labels.end());

    // ---- 5. Verify all output labels exist in at least one input ----
    for (char c : output_labels) {
        if (!all_labels.count(c)) {
            throw std::invalid_argument(
                "einsum: output label '" + std::string(1, c) +
                "' not found in any input");
        }
    }

    // ---- 6. Build output shape ----
    std::vector<py::ssize_t> output_shape;
    for (char c : output_labels) {
        output_shape.push_back(label_size[c]);
    }

    // ---- 7. Build iteration structures ----
    // Iteration labels: output labels (outer) + summation labels (inner)
    std::vector<char> iter_labels = output_labels;
    iter_labels.insert(iter_labels.end(), sum_labels.begin(), sum_labels.end());

    int n_iter = static_cast<int>(iter_labels.size());
    int n_output = static_cast<int>(output_labels.size());

    // Sizes for each iter label
    std::vector<py::ssize_t> iter_sizes(n_iter);
    for (int i = 0; i < n_iter; ++i) {
        iter_sizes[i] = label_size[iter_labels[i]];
    }

    // Strides for iter space
    auto iter_strides = compute_strides(iter_sizes);

    // Input strides for each array
    auto strides_a = compute_strides(shapes[0]);
    auto strides_b = compute_strides(shapes[1]);
    const std::vector<py::ssize_t>* input_strides[2] = {&strides_a, &strides_b};

    // For each iter label, store its axis position in each input (-1 if absent)
    // iter_input_axis[l_idx][input_idx] = axis_in_input or -1
    std::vector<std::vector<int>> iter_input_axis(n_iter, std::vector<int>(2, -1));
    for (int li = 0; li < n_iter; ++li) {
        char l = iter_labels[li];
        for (const auto& [inp, ax] : label_axis[l]) {
            iter_input_axis[li][inp] = ax;
        }
    }

    // ---- 8. Allocate output ----
    py::array_t<double> result(output_shape);
    auto res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    py::ssize_t output_total = res_buf.size;

    const double* a_ptr = static_cast<const double*>(bufa.ptr);
    const double* b_ptr = static_cast<const double*>(bufb.ptr);

    std::vector<py::ssize_t> iter_coord(n_iter, 0);
    std::vector<py::ssize_t> input_coord[2] = {
        std::vector<py::ssize_t>(ndim[0], 0),
        std::vector<py::ssize_t>(ndim[1], 0)
    };

    // ---- 9. Compute ----
    // Iterate over all positions in the iter space (output + summation)
    // We iterate in flat-index order over the iter space.
    // For each position, we know if it's a new output element (summation index is 0)
    // and we accumulate into the output element.

    py::ssize_t iter_total = 1;
    for (py::ssize_t s : iter_sizes) iter_total *= s;

    py::ssize_t current_output_idx = -1;
    double accumulator = 0.0;

    for (py::ssize_t flat = 0; flat < iter_total; ++flat) {
        // Decode flat index to iter_coord
        py::ssize_t rem = flat;
        for (int i = 0; i < n_iter; ++i) {
            iter_coord[i] = rem / iter_strides[i];
            rem %= iter_strides[i];
        }

        // Check if summation indices are all 0 → new output element
        bool is_new_output = true;
        for (int i = n_output; i < n_iter; ++i) {
            if (iter_coord[i] != 0) {
                is_new_output = false;
                break;
            }
        }

        if (is_new_output) {
            // Store previous accumulator and compute new output index
            if (current_output_idx >= 0) {
                res_ptr[current_output_idx] = accumulator;
            }

            // Compute output flat index from output labels only
            py::ssize_t out_idx = 0;
            py::ssize_t out_stride = 1;
            for (int i = n_output - 1; i >= 0; --i) {
                out_idx += iter_coord[i] * out_stride;
                if (i > 0) out_stride *= iter_sizes[i];
            }
            current_output_idx = out_idx;
            accumulator = 0.0;
        }

        // Compute input indices from iter_coord
        for (int inp = 0; inp < 2; ++inp) {
            std::fill(input_coord[inp].begin(), input_coord[inp].end(), 0);
        }

        for (int li = 0; li < n_iter; ++li) {
            for (int inp = 0; inp < 2; ++inp) {
                int ax = iter_input_axis[li][inp];
                if (ax >= 0) {
                    input_coord[inp][ax] = iter_coord[li];
                }
            }
        }

        // Compute flat indices into inputs
        py::ssize_t idx_a = flat_index(input_coord[0], *input_strides[0]);
        py::ssize_t idx_b_s = flat_index(input_coord[1], *input_strides[1]);

        // Multiply and accumulate
        accumulator += a_ptr[idx_a] * b_ptr[idx_b_s];
    }

    // Store last accumulator
    if (current_output_idx >= 0) {
        res_ptr[current_output_idx] = accumulator;
    }

    return result;
}

} // namespace numpy
