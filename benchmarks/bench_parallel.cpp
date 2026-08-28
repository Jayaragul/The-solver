// Where does CPU multithreading start to pay for itself, and does it
// change the answer?
//
// prompt.md \S3.2 permits OpenMP "where justified". This is the
// justification, or the refutation: a thread team costs a fork and a
// barrier on every parallel region, and the simplex enters one per pricing
// pass -- thousands of times per solve, on passes that are often only a
// few microseconds long. Parallelizing unconditionally would make small
// models slower, which is the same failure mode GPU pricing has and for
// the same reason. src/parallel/Parallel.hpp gates every parallel loop on
// a work threshold; this benchmark is where that threshold's value comes
// from.
//
// Two things are measured, and the second matters more than the first:
//
//   1. THROUGHPUT vs thread count across a sweep of matrix sizes, to
//      locate the size below which threading is a net loss.
//
//   2. BIT-REPRODUCIBILITY across thread counts. docs/architecture/
//      NUMERICS.md \S1 requires run-to-run identical results, and the
//      parallel loops in this engine are written to give one writer per
//      output with serial summation order precisely so that the thread
//      count cannot perturb the last bits. That is a claim about the code,
//      and claims get checked: this compares the full result vector
//      element-by-element for EXACT equality, not to a tolerance. A
//      tolerance would pass even if the property were broken.

#include "sparse/CSRMatrix.hpp"
#include "sparse/Triplet.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace sihps;

namespace {

CSRMatrix random_csr(std::int32_t rows, std::int32_t cols, std::int32_t per_row, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> col_pick(0, cols - 1);
    std::uniform_real_distribution<double> val(-1.0, 1.0);

    std::vector<Triplet> t;
    t.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(per_row));
    for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t k = 0; k < per_row; ++k) {
            t.push_back(Triplet{i, col_pick(rng), val(rng)});
        }
    }
    return CSRMatrix::from_triplets(rows, cols, t);
}

// An UNGATED copy of CSRMatrix::multiply's inner loop. The shipping
// implementation refuses to go parallel below kParallelNnzThreshold, which
// is exactly the behaviour under test -- so measuring it could never
// reveal where the threshold ought to sit, only confirm where it already
// does. This runs the parallel form unconditionally so the crossover can
// be observed rather than assumed. It is a measuring instrument, not a
// second implementation: it must stay identical to the real loop.
void multiply_ungated(const CSRMatrix& a, const double* x, double* y) {
    const std::int32_t rows = a.rows();
    const std::int32_t* row_ptr = a.row_ptr();
    const std::int32_t* col_idx = a.col_idx();
    const double* values = a.values();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int32_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        const std::int32_t begin = row_ptr[i];
        const std::int32_t end = row_ptr[i + 1];
        for (std::int32_t k = begin; k < end; ++k) acc += values[k] * x[col_idx[k]];
        y[i] = acc;
    }
}

template <typename Fn>
double time_it(Fn&& fn, int reps) {
    fn(); // warm the caches; not timed
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) fn();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void set_threads(int n) {
#if defined(_OPENMP)
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // namespace

int main() {
#if !defined(_OPENMP)
    std::printf("Built WITHOUT OpenMP. Every loop runs serially and the results below are the\n"
                "serial baseline -- which is a supported configuration, not a degraded one.\n\n");
#endif
    const int threads = max_threads();
    std::printf("OpenMP max threads: %d\n\n", threads);

    struct Case {
        std::int32_t rows, cols, per_row;
    };
    const std::vector<Case> cases = {
        {100, 100, 5},      {200, 200, 5},       {400, 400, 5},      {800, 800, 5},
        {1200, 1200, 5},    {2000, 2000, 5},     {5000, 5000, 6},    {20000, 20000, 8},
        {50000, 50000, 10}, {200000, 200000, 12},
    };

    std::printf("%9s %10s | %12s %12s %8s | %12s %8s | %10s\n", "rows", "nnz", "serial (s)",
                "par-ungated", "speedup", "as shipped", "vs ser", "identical");
    std::printf("%s\n", std::string(100, '-').c_str());

    bool all_identical = true;
    for (const auto& c : cases) {
        const CSRMatrix a = random_csr(c.rows, c.cols, c.per_row, 12345u);
        std::vector<double> x(static_cast<std::size_t>(c.cols));
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 1.0 / (1.0 + static_cast<double>(i % 97));
        }
        std::vector<double> y_serial(static_cast<std::size_t>(c.rows));
        std::vector<double> y_par(static_cast<std::size_t>(c.rows));
        std::vector<double> y_ship(static_cast<std::size_t>(c.rows));

        // Small cases are microseconds long, so they need far more
        // repetitions before the timer says anything about them.
        const int reps = (a.nnz() > 1000000) ? 50 : (a.nnz() > 50000 ? 2000 : 20000);

        set_threads(1);
        const double t_serial =
            time_it([&] { multiply_ungated(a, x.data(), y_serial.data()); }, reps);

        set_threads(threads);
        const double t_par = time_it([&] { multiply_ungated(a, x.data(), y_par.data()); }, reps);
        const double t_ship = time_it([&] { a.multiply(x.data(), y_ship.data()); }, reps);

        // EXACT equality, element by element, against BOTH the ungated
        // parallel form and the shipping (possibly gated) one.
        const bool identical = std::equal(y_serial.begin(), y_serial.end(), y_par.begin()) &&
                                std::equal(y_serial.begin(), y_serial.end(), y_ship.begin());
        if (!identical) all_identical = false;

        std::printf("%9d %10d | %12.6f %12.6f %7.2fx | %12.6f %7.2fx | %10s\n", c.rows, a.nnz(),
                    t_serial, t_par, (t_par > 0.0) ? t_serial / t_par : 0.0, t_ship,
                    (t_ship > 0.0) ? t_serial / t_ship : 0.0, identical ? "yes" : "NO -- BUG");
    }

    set_threads(threads);
    std::printf("\nBit-identical across thread counts on every case: %s\n",
                all_identical ? "yes" : "NO -- this violates NUMERICS.md 1");
    std::printf("The threshold in src/parallel/Parallel.hpp should sit just above the largest\n"
                "nnz whose speedup is still <= 1.0x.\n");
    return all_identical ? 0 : 1;
}
