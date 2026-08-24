// Exports each parsed LpProblem to a plain-text canonical form, and
// records this solver's own result for the same instance.
//
// Why this exists: to compare against an independent reference solver
// honestly, BOTH solvers must provably receive the identical problem. If
// the reference read the .mps file with its own parser instead, any
// disagreement would be ambiguous -- a genuine solver difference, or just
// two parsers disagreeing about MPS semantics. Exporting the post-parse
// problem removes that confound entirely: the reference solves exactly the
// matrix and bounds this solver solved.
//
// prompt.md's Benchmark Strategy explicitly calls for benchmarking against
// publicly available reference implementations while the core solver stays
// independent; it also permits Python for benchmarking/orchestration only.
// The reference solver is therefore invoked from a separate Python script
// (benchmarks/compare_reference.py) and is NEVER linked into, wrapped by,
// or called from this project's solver.
//
// Row activity bounds are emitted directly rather than row_types, so
// RANGES rows (two-sided) are represented exactly. Given the augmented
// system A x + s = rhs with s in [slack_lower, slack_upper]:
//
//     A x = rhs - s   =>   row_lb = rhs - slack_upper
//                          row_ub = rhs - slack_lower
//
// Format (one file per instance, whitespace-delimited, "inf"/"-inf" for
// infinite bounds):
//   line 1: n_rows n_cols nnz
//   line 2: row_ptr    (n_rows+1 values)
//   line 3: col_idx    (nnz values)
//   line 4: values     (nnz values)
//   line 5: obj        (n_cols values)
//   line 6: col_lower  (n_cols values)
//   line 7: col_upper  (n_cols values)
//   line 8: row_lb     (n_rows values)
//   line 9: row_ub     (n_rows values)

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/Simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

void write_doubles(std::ofstream& out, const double* v, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (i) out << ' ';
        const double x = v[i];
        if (std::isinf(x)) {
            out << (x > 0 ? "inf" : "-inf");
        } else {
            out.precision(17);
            out << x;
        }
    }
    out << '\n';
}

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPTIMAL";
        case LpStatus::INFEASIBLE: return "INFEASIBLE";
        case LpStatus::UNBOUNDED: return "UNBOUNDED";
        case LpStatus::ITERATION_LIMIT: return "ITERATION_LIMIT";
        case LpStatus::NUMERICAL_FAILURE: return "NUMERICAL_FAILURE";
    }
    return "UNKNOWN";
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::string out_dir = (argc > 2) ? argv[2] : "build/lp_export";
    const std::int32_t max_rows = (argc > 3) ? std::atoi(argv[3]) : 600;

    fs::create_directories(out_dir);

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::ofstream manifest(out_dir + "/ours.csv");
    manifest << "instance,rows,cols,nnz,status,objective,seconds,iterations\n";

    int exported = 0;
    for (const auto& path : files) {
        const std::string name = path.stem().string();

        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) continue;

        const LpProblem p = lp_problem_from_mps(model);
        const auto m = static_cast<std::size_t>(p.n_rows());
        const auto n = static_cast<std::size_t>(p.n_cols());

        std::vector<double> row_lb(m), row_ub(m);
        for (std::size_t i = 0; i < m; ++i) {
            row_lb[i] = p.rhs[i] - p.slack_upper[i];
            row_ub[i] = p.rhs[i] - p.slack_lower[i];
        }

        {
            std::ofstream out(out_dir + "/" + name + ".lp.txt");
            out << p.n_rows() << ' ' << p.n_cols() << ' ' << p.A.nnz() << '\n';
            for (std::int32_t i = 0; i <= p.n_rows(); ++i) {
                if (i) out << ' ';
                out << p.A.row_ptr()[i];
            }
            out << '\n';
            for (std::int32_t k = 0; k < p.A.nnz(); ++k) {
                if (k) out << ' ';
                out << p.A.col_idx()[k];
            }
            out << '\n';
            write_doubles(out, p.A.values(), static_cast<std::size_t>(p.A.nnz()));
            write_doubles(out, p.obj.data(), n);
            write_doubles(out, p.lower.data(), n);
            write_doubles(out, p.upper.data(), n);
            write_doubles(out, row_lb.data(), m);
            write_doubles(out, row_ub.data(), m);
        }

        LpResult r;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            Simplex simplex(p, PricingBackend::CPU);
            r = simplex.solve();
        } catch (const std::exception&) {
            r.status = LpStatus::NUMERICAL_FAILURE;
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        manifest.precision(17);
        manifest << name << ',' << p.n_rows() << ',' << p.n_cols() << ',' << p.A.nnz() << ','
                  << status_str(r.status) << ',' << r.objective_value << ',' << secs << ','
                  << (r.phase1_iterations + r.phase2_iterations) << '\n';
        ++exported;
    }

    std::printf("Exported %d instances to %s (plus ours.csv with this solver's results)\n",
                exported, out_dir.c_str());
    return 0;
}
