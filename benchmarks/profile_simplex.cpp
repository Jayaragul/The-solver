// Per-stage cost breakdown of one simplex solve.
//
// WHY THIS EXISTS
// ---------------
// It was built to chase a number that turned out to be wrong, and it is
// the reason the number was caught.
//
// A `debug_one` run reported the CPU simplex taking 259.67 s on stocfor3
// (16,675 rows) with Devex pricing against 9.77 s with Dantzig, at nearly
// identical iteration counts -- 31x per iteration. The algorithm predicts
// nothing like that: Devex adds one BTRAN and one O(nnz) pivot-row pass to
// an iteration that already pays for a BTRAN (the duals) and an O(nnz)
// pass (pricing), so roughly 2-3x. An order of magnitude between predicted
// and measured cost is worth stopping for.
//
// The tempting move was to reason about the code from the aggregate
// number. The right move was to measure per stage, which is what this
// does. RESULT: Devex costs 1,304 us/iteration and Dantzig 528 us -- a
// ratio of 2.16x, exactly as predicted. On degen3 it is 249 us vs 139 us,
// 1.79x. There was no defect. The 259.67 s came from two benchmark
// processes running concurrently, each spawning 16 OpenMP threads on a
// 16-core machine, with the contamination landing asymmetrically on the
// Devex runs because debug_one schedules them first.
//
// So the standing lesson, recorded in docs/architecture/PDLP.md 5: a
// predicted-vs-measured gap that large is a real signal, but it points at
// the MEASUREMENT at least as often as at the code -- and on this machine
// concurrent benchmark processes are not merely noisy, they are actively
// misleading, because OpenMP oversubscription hits size-gated parallel
// paths far harder than serial ones. Run timings one process at a time.
//
// THE LESSON WAS NOT LEARNED THE FIRST TIME. Having caught this artifact
// on stocfor3 and written it down, the obvious follow-up question -- what
// ELSE was measured while something was running? -- went unasked. The
// answer was the entire PDLP-vs-CPU sweep, which had manufactured a 39x
// win on degen3 that clean re-measurement showed to be a tie, and an 8x
// win on d2q06c that was really a 2.8x loss. When one number is found to
// be contaminated, re-take every number gathered under the same
// conditions, not just the one that looked wrong.
//
// CAVEAT ON THIS TOOL S OWN OUTPUT: nine ScopedTimers per iteration is not
// free. The same stocfor3 Devex configuration measures 13.6-14.3 s
// uninstrumented and 20.3 s under this profiler -- roughly 40% overhead.
// The RATIOS here are sound; the ABSOLUTE us/iteration figures are not a
// per-iteration cost and must not be quoted as one.
//
// The tool stays because the breakdown is generally useful: it is the only
// thing in the repository that can say whether a slow solve is spending
// its time in the factorization, the pricing pass, or the ratio test.

#include "io/MpsReader.hpp"
#include "lp/LpProblem.hpp"
#include "lp/LpSolver.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace sihps;

namespace {

struct Stage {
    const char* name;
    double seconds;
};

void report(const char* label, const LpSolution& s, double wall) {
    const SimplexProfile& p = s.simplex_profile;
    const double iters = p.iterations > 0 ? static_cast<double>(p.iterations) : 1.0;

    std::printf("\n--- %s ---\n", label);
    std::printf("status=%d  obj=%.9g  iterations=%ld  refactorizations=%d  wall=%.3f s\n",
                static_cast<int>(s.status), s.objective_value, p.iterations, s.refactorizations,
                wall);

    const std::vector<Stage> stages = {
        {"refactorize + re-derive", p.refactor_seconds},
        {"duals (BTRAN)", p.duals_seconds},
        {"price + entering scan", p.price_seconds},
        {"FTRAN (entering column)", p.ftran_seconds},
        {"ratio test (both passes)", p.ratio_seconds},
        {"pivot row: BTRAN", p.rho_btran_seconds},
        {"pivot row: A^T assembly", p.rho_assemble_seconds},
        {"Devex weight update", p.devex_seconds},
        {"basis update (eta push)", p.update_seconds},
    };

    const double total = p.total();
    std::printf("%-28s %10s %12s %8s\n", "stage", "seconds", "us/iter", "share");
    std::printf("%s\n", std::string(62, '-').c_str());
    for (const Stage& st : stages) {
        std::printf("%-28s %10.3f %12.1f %7.1f%%\n", st.name, st.seconds,
                    st.seconds * 1e6 / iters, total > 0.0 ? 100.0 * st.seconds / total : 0.0);
    }
    std::printf("%s\n", std::string(62, '-').c_str());
    std::printf("%-28s %10.3f %12.1f\n", "accounted", total, total * 1e6 / iters);
    std::printf("%-28s %10.3f\n", "unaccounted (presolve, verify)", wall - total);
}

LpSolution timed_solve(const LpProblem& p, PricingRule rule, double& wall) {
    LpSolverOptions opts;
    opts.pricing_rule = rule;
    const auto t0 = std::chrono::steady_clock::now();
    LpSolution s;
    try {
        s = solve_lp(p, opts);
    } catch (const std::exception& e) {
        std::printf("threw: %s\n", e.what());
    }
    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: profile_simplex <instance.mps> [devex|dantzig|both]\n");
        return 1;
    }
    const std::string which = (argc > 2) ? argv[2] : "both";

    const MpsModel model = read_mps_file(argv[1]);
    const LpProblem p = lp_problem_from_mps(model);
    std::printf("%s: rows=%d cols=%d nnz=%d\n", argv[1], p.n_rows(), p.n_cols(), p.A.nnz());

    if (which == "devex" || which == "both") {
        double wall = 0.0;
        const LpSolution s = timed_solve(p, PricingRule::DEVEX, wall);
        report("DEVEX", s, wall);
    }
    if (which == "dantzig" || which == "both") {
        double wall = 0.0;
        const LpSolution s = timed_solve(p, PricingRule::DANTZIG, wall);
        report("DANTZIG", s, wall);
    }
    return 0;
}
