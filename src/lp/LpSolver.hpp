#pragma once

#include "../cuda/GpuPdlp.hpp"
#include "LpProblem.hpp"
#include "Presolve.hpp"
#include "Simplex.hpp"

#include <cstdint>
#include <vector>

namespace sihps {

// Which family of algorithm solves the LP.
//
// SIMPLEX is the default and returns an exact vertex solution certified by
// the residual gate in NUMERICS.md 6.
//
// FIRST_ORDER runs GPU PDLP (src/cuda/GpuPdlp.hpp). It is the ONLY place in
// this engine where the GPU beats the CPU, and the reason is structural
// rather than a matter of kernel quality: a simplex iteration needs a host
// decision (which column enters) and therefore a device synchronize every
// iteration, while PDHG needs none and can queue hundreds of iterations per
// sync. docs/architecture/CPU_GPU.md 4 records the measurement that killed
// GPU simplex; 6 records what this path does instead.
//
// The trade is real and is not hidden: a first-order method converges
// quickly to moderate accuracy and slowly to high accuracy, and it returns
// an interior-ish point rather than a vertex. Both methods are held to the
// SAME original-space verification gate, so a FIRST_ORDER result is never
// reported OPTIMAL on the strength of its own iteration count.
// SIMPLEX and FIRST_ORDER pin the algorithm. HYBRID runs the simplex and
// falls back to the first-order solver only when the simplex fails to
// return a verified optimum.
//
// The fallback exists because the two methods fail on disjoint sets of
// models, and neither failure is predictable in advance
// (docs/architecture/PDLP.md 5). On the Netlib feasible set at a
// 20,000-row cap the simplex solves 92 of 93 and abandons `dfl001` at its
// iteration limit after 792 s; the first-order path solves `dfl001` to an
// objective error of 1.4e-07 in about 2 s. Conversely the first-order path
// stalls on six models the simplex disposes of in under 5 s. Running the
// simplex first and falling back costs nothing on the 92 it already solves
// and converts the remaining failure into a pass.
enum class LpMethod { SIMPLEX, FIRST_ORDER, HYBRID };

struct LpSolverOptions {
    bool use_presolve = true;
    LpMethod method = LpMethod::SIMPLEX;
    PdlpParams pdlp;
    bool use_ruiz_scaling = true;
    PricingBackend backend = PricingBackend::CPU;
    PricingRule pricing_rule = PricingRule::DEVEX;

    // AUTO resolves to the primal two-phase path on a cold start -- every
    // solve today -- per docs/architecture/LP.md \S2's decision table and
    // the cold-start measurement in \S2.1. PRIMAL and DUAL pin the choice,
    // which is what benchmarks/bench_lp_algorithm.cpp needs to compare them,
    // and what the future warm-started MILP caller will need to request the
    // dual path deliberately.
    LpAlgorithm algorithm = LpAlgorithm::AUTO;
};

struct LpSolution {
    LpStatus status = LpStatus::NUMERICAL_FAILURE;
    double objective_value = 0.0;
    std::vector<double> x; // ORIGINAL column space, always

    int iterations = 0;
    int refactorizations = 0;
    double presolve_seconds = 0.0;
    double solve_seconds = 0.0;

    // Cumulative time inside pricing alone -- reduced-cost computation plus
    // the entering-variable search, plus the Devex pivot row and weight
    // update when that rule is active. Reported separately because it is
    // the ONLY stage that differs between PricingBackend::CPU and ::GPU, so
    // it is what an honest backend comparison has to isolate
    // (benchmarks/bench_pricing_backend.cpp): a speedup measured on
    // solve_seconds alone would be diluted by the factorization and ratio
    // test, which are identical on both backends.
    double pricing_seconds = 0.0;

    // Which algorithm produced the reported result, and how many dual
    // iterations were spent (nonzero even on a fallback, since that work
    // was really performed -- see Simplex::solve).
    bool used_dual_simplex = false;
    int dual_iterations = 0;

    std::int32_t presolve_removed_rows = 0;
    std::int32_t presolve_removed_cols = 0;

    // Populated only when options.method == FIRST_ORDER. `host_syncs` in
    // particular is worth reading: it is the quantity the GPU path exists
    // to keep small, and a regression there would show up as a slowdown
    // with no other symptom.
    bool used_first_order = false;
    // True when HYBRID actually had to fall back -- i.e. the simplex did
    // not produce a verified optimum. Reported so a benchmark can tell a
    // free fallback from one that was exercised.
    bool first_order_fallback_used = false;
    PdlpStats pdlp;

    // Per-stage simplex cost breakdown (SIMPLEX method only).
    SimplexProfile simplex_profile;

    // Primal residual is measured against the ORIGINAL model -- row bounds
    // and column bounds as the caller stated them -- so it certifies the
    // whole chain (presolve, scaling, simplex, postsolve) rather than any
    // one stage's internal view of itself.
    double primal_residual = 0.0;

    // Dual residual is measured on whatever problem the simplex actually
    // solved. With presolve enabled that is the REDUCED problem: this
    // engine does not yet reconstruct duals through presolve, so dual
    // feasibility is certified for the reduction, and the argument that
    // optimality carries back to the original model rests on the
    // correctness of the reductions rather than on a second numerical
    // check. Stated here rather than left implicit -- see
    // docs/architecture/NUMERICS.md \S6.
    double dual_residual = 0.0;
    bool dual_residual_is_reduced_space = false;
};

// Full LP pipeline: presolve -> scale -> simplex -> postsolve -> verify in
// original space (docs/architecture/SYSTEM.md \S1). This is the entry point
// callers should use; Simplex on its own solves whatever problem it is
// handed and knows nothing about presolve.
LpSolution solve_lp(const LpProblem& problem, const LpSolverOptions& options = {});

} // namespace sihps
