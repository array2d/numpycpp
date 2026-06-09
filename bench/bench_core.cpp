#include "numpycpp/numpy.h"
#include "numpycpp/linalg.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cmath>

// ---- helpers -----------------------------------------------------------------

std::vector<double> make_data(size_t n) {
    std::vector<double> v(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

// ---- element-wise -------------------------------------------------------------

#define BENCH_ELEMWISE(NAME)                                       \
static void BM_##NAME(benchmark::State& state) {                   \
    size_t n = state.range(0);                                     \
    auto src = make_data(n);                                       \
    std::vector<double> dst(n);                                    \
    for (auto _ : state) {                                         \
        numpy::NAME(src.data(), dst.data(), n);                    \
        benchmark::DoNotOptimize(dst.data());                      \
    }                                                              \
    state.SetItemsProcessed(state.iterations() * n);               \
}                                                                  \
BENCHMARK(BM_##NAME)->Range(1 << 10, 1 << 22);

BENCH_ELEMWISE(sqrt)
BENCH_ELEMWISE(abs)
BENCH_ELEMWISE(exp)
BENCH_ELEMWISE(log)
BENCH_ELEMWISE(sin)
BENCH_ELEMWISE(cos)

// ---- reduction ---------------------------------------------------------------

static void BM_sum(benchmark::State& state) {
    size_t n = state.range(0);
    auto src = make_data(n);
    for (auto _ : state) {
        double s = numpy::sum(src.data(), n);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_sum)->Range(1 << 10, 1 << 22);

static void BM_mean(benchmark::State& state) {
    size_t n = state.range(0);
    auto src = make_data(n);
    for (auto _ : state) {
        double m = numpy::mean(src.data(), n);
        benchmark::DoNotOptimize(m);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_mean)->Range(1 << 10, 1 << 22);

static void BM_max(benchmark::State& state) {
    size_t n = state.range(0);
    auto src = make_data(n);
    for (auto _ : state) {
        double m = numpy::max(src.data(), n);
        benchmark::DoNotOptimize(m);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_max)->Range(1 << 10, 1 << 22);

// ---- dot product (1D) ---------------------------------------------------------

static void BM_dot(benchmark::State& state) {
    size_t n = state.range(0);
    auto a = make_data(n);
    auto b = make_data(n);
    for (auto _ : state) {
        double d = numpy::dot(a.data(), b.data(), n);
        benchmark::DoNotOptimize(d);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_dot)->Range(1 << 10, 1 << 22);

// ---- linalg norm --------------------------------------------------------------

static void BM_norm(benchmark::State& state) {
    size_t n = state.range(0);
    auto src = make_data(n);
    for (auto _ : state) {
        double r = numpy::linalg::norm(src.data(), n);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_norm)->Range(1 << 10, 1 << 22);

// ---- main --------------------------------------------------------------------

BENCHMARK_MAIN();
