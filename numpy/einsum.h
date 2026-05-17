// Native C++ einsum implementation — zero pybind11 dependency.
// Supports 2-operand explicit and implicit mode summation.

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

namespace numpy {
namespace einsum_detail {

// ============================================================================
// Subscript parsing
// ============================================================================
struct ParsedSubscripts {
    std::vector<std::string> inputs;
    std::string output;
    bool explicit_mode;
};

inline ParsedSubscripts parse_subscripts(const std::string& subscripts) {
    ParsedSubscripts result{};
    auto arrow = subscripts.find("->");
    std::string lhs;
    if (arrow != std::string::npos) {
        lhs = subscripts.substr(0, arrow);
        result.output = subscripts.substr(arrow + 2);
        result.explicit_mode = true;
    } else {
        lhs = subscripts;
    }

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

inline std::vector<ptrdiff_t> compute_strides(const std::vector<ptrdiff_t>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<ptrdiff_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

inline ptrdiff_t flat_index(const std::vector<ptrdiff_t>& coord,
                              const std::vector<ptrdiff_t>& strides) {
    ptrdiff_t idx = 0;
    for (size_t i = 0; i < coord.size(); ++i)
        idx += coord[i] * strides[i];
    return idx;
}

inline std::string implicit_output_labels(const std::vector<std::string>& il) {
    std::map<char, int> cnt;
    for (const auto& s : il) {
        std::set<char> seen;
        for (char c : s)
            if (seen.insert(c).second) cnt[c]++;
    }
    std::string out;
    for (const auto& [c, n] : cnt)
        if (n == 1) out += c;
    std::sort(out.begin(), out.end());
    return out;
}

// ============================================================================
// Main einsum computation
// ============================================================================

/// numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K',
///              casting='safe', optimize=False)
//  Currently supports 2-operand patterns only.
template<typename T>
void einsum(const std::string& subscripts,
            const T* a_ptr, const ptrdiff_t* a_shape, int a_ndim,
            const T* b_ptr, const ptrdiff_t* b_shape, int b_ndim,
            T* result_ptr) {

    using namespace std;

    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    string out_labels = parsed.output;

    if (in_labels.size() != 2)
        throw invalid_argument("einsum: currently only supports 2 operands, got " +
                               to_string(in_labels.size()));

    if (!parsed.explicit_mode)
        out_labels = implicit_output_labels(in_labels);

    vector<ptrdiff_t> shapes[2] = {
        vector<ptrdiff_t>(a_shape, a_shape + a_ndim),
        vector<ptrdiff_t>(b_shape, b_shape + b_ndim)
    };
    int ndim[2] = {a_ndim, b_ndim};

    for (int i = 0; i < 2; ++i) {
        if (static_cast<int>(in_labels[i].size()) != ndim[i])
            throw invalid_argument(
                "einsum: subscript '" + in_labels[i] + "' has " +
                to_string(in_labels[i].size()) + " labels but operand " +
                to_string(i) + " has " + to_string(ndim[i]) + " dimensions");
    }

    // Label → (input_idx, axis) mapping
    map<char, vector<pair<int, int>>> label_axis;
    map<char, ptrdiff_t> label_size;

    for (int i = 0; i < 2; ++i) {
        for (int ax = 0; ax < ndim[i]; ++ax) {
            char l = in_labels[i][ax];
            label_axis[l].push_back({i, ax});
            ptrdiff_t sz = shapes[i][ax];
            auto it = label_size.find(l);
            if (it == label_size.end()) label_size[l] = sz;
            else if (it->second != sz)
                throw invalid_argument(
                    "einsum: label '" + string(1, l) + "' has inconsistent sizes");
        }
    }

    set<char> out_set(out_labels.begin(), out_labels.end());
    set<char> all_labels;
    for (const auto& s : in_labels)
        for (char c : s) all_labels.insert(c);

    vector<char> sum_labels;
    for (char c : all_labels)
        if (!out_set.count(c)) sum_labels.push_back(c);

    vector<char> output_labels(out_labels.begin(), out_labels.end());

    for (char c : output_labels)
        if (!all_labels.count(c))
            throw invalid_argument(
                "einsum: output label '" + string(1, c) + "' not found in any input");

    // Output shape
    vector<ptrdiff_t> output_shape;
    for (char c : output_labels)
        output_shape.push_back(label_size[c]);

    // Iteration space
    vector<char> iter_labels = output_labels;
    iter_labels.insert(iter_labels.end(), sum_labels.begin(), sum_labels.end());
    int n_iter = static_cast<int>(iter_labels.size());
    int n_output = static_cast<int>(output_labels.size());

    vector<ptrdiff_t> iter_sizes(n_iter);
    for (int i = 0; i < n_iter; ++i)
        iter_sizes[i] = label_size[iter_labels[i]];

    auto iter_strides = compute_strides(iter_sizes);
    auto strides_a = compute_strides(shapes[0]);
    auto strides_b = compute_strides(shapes[1]);
    const vector<ptrdiff_t>* input_strides[2] = {&strides_a, &strides_b};

    vector<vector<int>> iter_input_axis(n_iter, vector<int>(2, -1));
    for (int li = 0; li < n_iter; ++li) {
        char l = iter_labels[li];
        for (const auto& [inp, ax] : label_axis[l])
            iter_input_axis[li][inp] = ax;
    }

    // Compute total output size for validation
    vector<ptrdiff_t> iter_coord(n_iter, 0);
    vector<ptrdiff_t> input_coord[2] = {
        vector<ptrdiff_t>(ndim[0], 0),
        vector<ptrdiff_t>(ndim[1], 0)
    };

    ptrdiff_t iter_total = 1;
    for (ptrdiff_t s : iter_sizes) iter_total *= s;

    ptrdiff_t current_output_idx = -1;
    T accumulator = T(0);

    for (ptrdiff_t flat = 0; flat < iter_total; ++flat) {
        ptrdiff_t rem = flat;
        for (int i = 0; i < n_iter; ++i) {
            iter_coord[i] = rem / iter_strides[i];
            rem %= iter_strides[i];
        }

        bool is_new_output = true;
        for (int i = n_output; i < n_iter; ++i) {
            if (iter_coord[i] != 0) { is_new_output = false; break; }
        }

        if (is_new_output) {
            if (current_output_idx >= 0)
                result_ptr[current_output_idx] = accumulator;

            ptrdiff_t out_idx = 0, out_stride = 1;
            for (int i = n_output - 1; i >= 0; --i) {
                out_idx += iter_coord[i] * out_stride;
                if (i > 0) out_stride *= iter_sizes[i];
            }
            current_output_idx = out_idx;
            accumulator = T(0);
        }

        for (int inp = 0; inp < 2; ++inp)
            fill(input_coord[inp].begin(), input_coord[inp].end(), 0);

        for (int li = 0; li < n_iter; ++li)
            for (int inp = 0; inp < 2; ++inp) {
                int ax = iter_input_axis[li][inp];
                if (ax >= 0) input_coord[inp][ax] = iter_coord[li];
            }

        ptrdiff_t idx_a = flat_index(input_coord[0], *input_strides[0]);
        ptrdiff_t idx_b = flat_index(input_coord[1], *input_strides[1]);

        accumulator += a_ptr[idx_a] * b_ptr[idx_b];
    }

    if (current_output_idx >= 0)
        result_ptr[current_output_idx] = accumulator;
}

/// numpy.einsum(subscripts, *operands) — compute output shape
inline std::vector<ptrdiff_t> einsum_output_shape(const std::string& subscripts,
                                                    const ptrdiff_t* a_shape, int a_ndim,
                                                    const ptrdiff_t* b_shape, int b_ndim) {
    auto parsed = parse_subscripts(subscripts);
    const auto& in_labels = parsed.inputs;
    std::string out_labels = parsed.output;

    if (!parsed.explicit_mode)
        out_labels = implicit_output_labels(in_labels);

    std::map<char, ptrdiff_t> label_size;
    for (int i = 0; i < a_ndim; ++i)
        label_size[in_labels[0][i]] = a_shape[i];
    for (int i = 0; i < b_ndim; ++i)
        label_size[in_labels[1][i]] = b_shape[i];

    std::vector<ptrdiff_t> output_shape;
    for (char c : out_labels)
        output_shape.push_back(label_size[c]);
    return output_shape;
}

} // namespace einsum_detail
} // namespace numpy
