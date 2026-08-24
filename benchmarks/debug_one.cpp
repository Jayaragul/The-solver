// One-off diagnostic: solve a single named instance and print every
// residual/iteration/refactorization field LpSolution exposes, plus both
// presolve on/off and both pricing rules, so a NUM_FAILURE/ITER_LIMIT case
// can be triaged without re-running the whole benchmark suite.
#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <cstdio>
#include <string>

using namespace sihps;

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

void run(const LpProblem& p, bool presolve, PricingRule rule, const char* label) {
    LpSolverOptions opts;
    opts.use_presolve = presolve;
    opts.pricing_rule = rule;
    LpSolution r = solve_lp(p, opts);
    std::printf("%-28s status=%-12s obj=%.6f iters=%d refac=%d primal_res=%.3e dual_res=%.3e "
                "(dual_reduced_space=%d) presolve_sec=%.4f solve_sec=%.4f removed(rows=%d,cols=%d)\n",
                label, status_str(r.status), r.objective_value, r.iterations, r.refactorizations,
                r.primal_residual, r.dual_residual, r.dual_residual_is_reduced_space,
                r.presolve_seconds, r.solve_seconds, r.presolve_removed_rows,
                r.presolve_removed_cols);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <path-to-mps>\n", argv[0]);
        return 2;
    }
    MpsModel model = read_mps_file(argv[1]);
    LpProblem p = lp_problem_from_mps(model);
    std::printf("%s: rows=%d cols=%d nnz=%d\n\n", argv[1], p.n_rows(), p.n_cols(), p.A.nnz());

    run(p, true, PricingRule::DEVEX, "presolve=on  devex");
    run(p, false, PricingRule::DEVEX, "presolve=off devex");
    run(p, true, PricingRule::DANTZIG, "presolve=on  dantzig");
    run(p, false, PricingRule::DANTZIG, "presolve=off dantzig");
    return 0;
}
