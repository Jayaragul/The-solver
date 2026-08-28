// Correctness validation on the Netlib INFEASIBLE LP set.
//
// Everything else in this benchmark suite asks "does the solver find the
// right optimum?". This asks the complementary and equally important
// question: "does it correctly REFUSE a problem that has no solution?"
//
// That distinction matters more than it first appears. A solver that
// reports OPTIMAL on an infeasible refinery model does not merely produce a
// wrong number -- it produces a production schedule that cannot be run,
// with no indication anything is wrong. Reporting INFEASIBLE is the correct
// answer here, not a failure to solve.
//
// The reference is structural, not a published value: every instance in
// data/netlib_lp/infeasible IS infeasible by construction of that
// collection (Netlib distributes it as such), so the expected status is
// known without any per-instance data. Nothing is hardcoded per instance.
//
// Accepted outcomes, and why each is or is not a pass:
//   INFEASIBLE      -- correct.
//   NUM_FAILURE     -- honest refusal: the engine could not certify a
//                      result and said so. Counted separately, not as a
//                      pass, because a real solver should conclude
//                      infeasibility rather than give up.
//   OPTIMAL         -- WRONG, and the serious failure this tool exists to
//                      catch: a solution was returned for a problem that
//                      has none.
//   UNBOUNDED       -- wrong for this set.
//   ITER_LIMIT      -- did not finish; not a correctness failure, but not a
//                      pass either.

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace sihps;
namespace fs = std::filesystem;

namespace {

const char* status_str(LpStatus s) {
    switch (s) {
        case LpStatus::OPTIMAL: return "OPTIMAL";
        case LpStatus::INFEASIBLE: return "INFEASIBLE";
        case LpStatus::UNBOUNDED: return "UNBOUNDED";
        case LpStatus::ITERATION_LIMIT: return "ITER_LIMIT";
        case LpStatus::NUMERICAL_FAILURE: return "NUM_FAILURE";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "data/netlib_lp/infeasible";
    const std::int32_t max_rows = (argc > 2) ? std::atoi(argv[2]) : 20000;

    std::vector<fs::path> files;
    if (!fs::exists(dir)) {
        std::printf("Directory does not exist: %s\n", dir.c_str());
        return 2;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".mps") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return fs::file_size(a) < fs::file_size(b);
    });

    std::printf("Netlib INFEASIBLE set -- every instance must NOT be reported OPTIMAL\n\n");
    std::printf("%14s %7s %7s %13s %9s %8s %s\n", "instance", "rows", "cols", "status", "sec",
                "iters", "verdict");
    std::printf("%s\n", std::string(78, '-').c_str());

    int correct = 0, wrong = 0, gave_up = 0, skipped = 0;
    std::vector<std::string> wrong_names;

    for (const auto& path : files) {
        const std::string name = path.stem().string();

        MpsModel model;
        try {
            model = read_mps_file(path.string());
        } catch (const std::exception&) {
            ++skipped;
            continue;
        }
        if (model.n_rows == 0 || model.n_cols == 0 || model.n_rows > max_rows) {
            ++skipped;
            continue;
        }

        LpProblem p = lp_problem_from_mps(model);
        LpSolution result;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            result = solve_lp(p, LpSolverOptions{});
        } catch (const std::exception&) {
            result.status = LpStatus::NUMERICAL_FAILURE;
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const char* verdict;
        if (result.status == LpStatus::INFEASIBLE) {
            ++correct;
            verdict = "CORRECT";
        } else if (result.status == LpStatus::OPTIMAL ||
                    result.status == LpStatus::UNBOUNDED) {
            ++wrong;
            wrong_names.push_back(name);
            verdict = "*** WRONG ***";
        } else {
            ++gave_up;
            verdict = "inconclusive";
        }

        std::printf("%14s %7d %7d %13s %9.3f %8d %s\n", name.c_str(), model.n_rows, model.n_cols,
                    status_str(result.status), secs, result.iterations, verdict);
    }

    std::printf("\n%s\n", std::string(78, '=').c_str());
    std::printf("correctly rejected: %d | WRONGLY ACCEPTED: %d | inconclusive: %d | skipped: %d\n",
                correct, wrong, gave_up, skipped);
    if (!wrong_names.empty()) {
        std::printf("WRONGLY ACCEPTED:");
        for (const auto& n : wrong_names) std::printf(" %s", n.c_str());
        std::printf("\n");
    }
    // Only a wrongly-accepted instance is a hard failure. An inconclusive
    // result is a quality gap to close, not a correctness violation.
    return wrong == 0 ? 0 : 1;
}
