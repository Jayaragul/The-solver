// Dantzig vs Devex entering-variable selection, measured head to head on
// real Netlib LP instances.
//
// This exists because prompt.md forbids asserting an algorithmic win
// without evidence. Devex (Harris 1973) is an ESTABLISHED METHOD and the
// literature expects it to reduce iteration counts on degenerate models --
// but "the literature expects it" is not a measurement of THIS
// implementation on THIS workload, so both rules are kept and compared.
//
// Reported per instance: iteration counts and wall-clock for each rule,
// plus whether both rules reached the same objective (they must: the
// pricing rule changes the PATH through the vertices, never the optimum).

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/Simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

struct RunStats {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective = 0.0;
    int iterations = 0;
    double seconds = 0.0;
};

RunStats run_one(const LpProblem& p, PricingRule rule) {
    RunStats s;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        Simplex simplex(p, PricingBackend::CPU, /*use_ruiz_scaling=*/true, rule);
        LpResult r = simplex.solve();
        s.status = r.status;
        s.objective = r.objective_value;
        s.iterations = r.phase1_iterations + r.phase2_iterations;
    } catch (const std::exception&) {
        s.status = LpStatus::NUMERICAL_FAILURE;
    }
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return s;
}

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPT";
        case LpStatus::INFEASIBLE: return "INFEAS";
        case LpStatus::UNBOUNDED: return "UNBND";
        case LpStatus::ITERATION_LIMIT: return "ITERLIM";
        case LpStatus::NUMERICAL_FAILURE: return "NUMFAIL";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/feasible";
    const std::int32_t max_rows = (argc > 2) ? std::atoi(argv[2]) : 600;

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("%14s %6s %6s | %8s %9s %8s | %8s %9s %8s | %8s %8s %s\n", "instance", "rows",
                "cols", "DZ iter", "DZ sec", "DZ st", "DVX iter", "DVX sec", "DVX st", "iter x",
                "time x", "obj");
    std::printf("%s\n", std::string(132, '-').c_str());

    long total_dz_iter = 0, total_dvx_iter = 0;
    double total_dz_sec = 0.0, total_dvx_sec = 0.0;
    int both_opt = 0, obj_mismatch = 0;

    for (const auto& path : files) {
        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) continue;

        const LpProblem p = lp_problem_from_mps(model);
        const RunStats dz = run_one(p, PricingRule::DANTZIG);
        const RunStats dvx = run_one(p, PricingRule::DEVEX);

        bool obj_ok = true;
        if (dz.status == LpStatus::OPTIMAL && dvx.status == LpStatus::OPTIMAL) {
            ++both_opt;
            total_dz_iter += dz.iterations;
            total_dvx_iter += dvx.iterations;
            total_dz_sec += dz.seconds;
            total_dvx_sec += dvx.seconds;
            const double rel = std::fabs(dz.objective - dvx.objective) /
                                (1.0 + std::fabs(dz.objective));
            obj_ok = rel < 1e-6;
            if (!obj_ok) ++obj_mismatch;
        }

        const double iter_x = (dvx.iterations > 0)
                                   ? static_cast<double>(dz.iterations) / dvx.iterations
                                   : 0.0;
        const double time_x = (dvx.seconds > 1e-9) ? dz.seconds / dvx.seconds : 0.0;

        std::printf("%14s %6d %6d | %8d %9.3f %8s | %8d %9.3f %8s | %7.2fx %7.2fx %s\n",
                    path.stem().string().c_str(), model.n_rows, model.n_cols, dz.iterations,
                    dz.seconds, status_str(dz.status), dvx.iterations, dvx.seconds,
                    status_str(dvx.status), iter_x, time_x, obj_ok ? "match" : "MISMATCH");
    }

    std::printf("\n%s\n", std::string(132, '=').c_str());
    std::printf("Instances where both rules reached OPTIMAL: %d\n", both_opt);
    std::printf("Total iterations  Dantzig %ld  vs  Devex %ld", total_dz_iter, total_dvx_iter);
    if (total_dvx_iter > 0) {
        std::printf("   (%.2fx fewer with Devex)",
                     static_cast<double>(total_dz_iter) / static_cast<double>(total_dvx_iter));
    }
    std::printf("\nTotal wall-clock  Dantzig %.3f s  vs  Devex %.3f s", total_dz_sec,
                total_dvx_sec);
    if (total_dvx_sec > 1e-9) {
        std::printf("   (%.2fx)", total_dz_sec / total_dvx_sec);
    }
    std::printf("\nObjective mismatches between rules: %d (must be 0 -- the pricing rule changes "
                "the path, not the optimum)\n",
                obj_mismatch);
    return obj_mismatch == 0 ? 0 : 1;
}
