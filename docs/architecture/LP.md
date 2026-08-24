# LP Engine Architecture

**Status:** PHASE 2 architecture. Per prompt.md §2.7, the LP engine is designed independently of the MILP engine that will call it repeatedly, and no single algorithm is assumed universally optimal — the choice is adaptive, based on model characteristics, and stated as such rather than picked by default preference.

---

## 1. Algorithms Evaluated

| Algorithm | Status for v1 | Rationale |
|---|---|---|
| **Dual simplex** | **v1 default for warm starts** (implemented; see §2.1 — cold starts use primal, on measured evidence) | Warm-start-friendly from a parent B&B node's basis (the dominant workload shape in this project — thousands of related re-solves); matches the documented default reasoning in both HiGHS and CPLEX (SOTA.md §1.1, §1.2) |
| **Primal simplex** | v1 fallback, and v1 cold-start default (implemented; §2.1) | Used when dual simplex cannot establish a feasible starting basis, or as the recovery path in `NUMERICS.md` §5's fallback chain |
| **Barrier / interior-point** | **Deferred** | A from-scratch IPM (predictor-corrector, normal-equations or KKT solve) is a large independent engineering effort; SOTA.md §1.3b's literature review found GPU-accelerated IPM factorization speedups to be modest in practice (fill-in erodes sparsity advantage) — not obviously worth building before the simplex core is validated. Revisit once v1 core is benchmarked (Level 6+) |
| **First-order primal-dual (PDLP-style)** | **Deferred (SOTA.md KS-7)** | Not warm-start-friendly for repeated B&B relaxations (iterates are not basic solutions); would introduce a second numerical code path requiring its own validation before the first is even benchmarked |

**Why dual simplex over primal as the default, specifically:** in this project's dominant workload (a B&B node relaxation is the parent's relaxation plus one tightened bound), the parent's optimal basis is typically dual-feasible for the child (only primal feasibility is disturbed by the bound change) — exactly the condition dual simplex exploits for warm starts. This is the same reasoning documented for CPLEX's default (SOTA.md §1.2) and is treated here as LITERATURE EVIDENCE supporting the choice, not a novel claim.

## 2. Adaptive Selection Strategy

```cpp
enum class LPMethod { DUAL_SIMPLEX, PRIMAL_SIMPLEX };

LPMethod select_method(const LPWarmStartContext& ctx) {
    if (ctx.has_parent_basis && ctx.parent_basis_dual_feasible)
        return LPMethod::DUAL_SIMPLEX;
    if (!ctx.has_parent_basis)
        return LPMethod::PRIMAL_SIMPLEX; // cold start: build initial feasible basis directly
    return LPMethod::DUAL_SIMPLEX; // default; NUMERICS.md §5 governs runtime fallback
}
```

### 2.1 Implementation status (Phase 3) — `src/lp/Simplex.cpp`, `src/lp/LpSolver.cpp`

Both algorithms are implemented and both are reachable: `LpAlgorithm::{PRIMAL, DUAL, AUTO}` on `Simplex`, surfaced as `LpSolverOptions::algorithm` on the `solve_lp` pipeline.

**`AUTO` resolves to `PRIMAL` today, and that is this table's own rule, not a preference.** The table selects dual simplex when a *parent basis* is available and dual-feasible for the child; it selects primal on a cold start. Every call today is a cold start, because no warm-start entry point exists yet — the MILP engine that will supply parent bases is not built. When it is, `AUTO` gains the parent-basis branch. Resolving `AUTO` to dual before then would be selecting an algorithm outside the conditions this table justifies it under.

**EXPERIMENTAL RESULT — cold-start primal vs dual** (`benchmarks/bench_lp_algorithm.cpp`, Netlib feasible set to 2600 rows, full `solve_lp` pipeline including presolve, Devex pricing, Ruiz scaling, 79 instances where the dual path was actually entered and both algorithms reached `OPTIMAL`):

| | total iterations | total wall-clock |
|---|---|---|
| Primal two-phase | 164,580 | 32.99 s |
| Dual simplex | 446,641 (**2.71×**) | 106.28 s (**3.22×**) |

The aggregate is not the whole finding, and the spread is the part that matters:

| instance | primal | dual | |
|---|---|---|---|
| `d6cube` | 31,033 it / 5.18 s | 1,070 it / 0.19 s | dual **29× fewer iterations** |
| `degen3` | 6,845 it / 1.10 s | 12,359 it / 2.02 s | dual 1.8× worse |
| `pilotnov` | 3,110 it / 0.29 s | 86,420 it / 6.78 s | dual runs, fails verification, primal fallback answers |
| `pilot87` | 14,071 it / 13.1 s | 110,278 it / 74.8 s | dual 5.7× slower **and** disagrees on the objective beyond 1e-6 |

So cold-start dual is not uniformly worse — it is *unpredictable*, and on a cold start that unpredictability buys nothing, because there is no warm basis whose retained dual feasibility is the entire reason to prefer it. This measurement supports the cold-start branch of the table above; it says **nothing** about the warm-start branch, which cannot be measured until warm starts exist.

Two classification details in that benchmark, both of which change the numbers if got wrong:

- **16 instances admit no dual-feasible start at all** and were excluded. `setup_dual_feasible_start` requires every cost to point toward a *finite* bound; where it does not, `DUAL` runs the primal path and comparing it against itself measures nothing.
- **Whether a model admits that start is a property of the model the simplex actually receives, so presolve changes it.** `pilot87` has no dual-feasible start unreduced but does after presolve fixes columns and tightens bounds. An earlier version of this benchmark drove `Simplex` directly, classified `pilot87` as "no-start", and reported dual as a 0.78× *win* overall — the opposite conclusion, from measuring a configuration that does not ship.

### 2.2 Dual simplex correctness work (Phase 3)

Three defects were found and fixed by wiring the dual path in and measuring it. All three shared a shape worth naming: each produced a *plausible* answer rather than an obvious crash.

1. **Single-pass dual ratio test → divergence and false infeasibility.** The test took the minimum ratio with only a 1e-12 tie-break toward larger pivots, so any pivot above `kPivotTol` (1e-9) was acceptable. Since the primal step is $t = \delta / \alpha_{\text{enter}}$, one such pivot converts a small bound violation into an enormous one, which becomes the next iteration's leaving row. **MEASURED on `grow15`:** worst basic infeasibility reached 1.4e+12 over 5,219 iterations, dual feasibility was lost entirely (max dual infeasibility 7.0 against a 1e-9 tolerance), and the solver then reported `INFEASIBLE` **on a feasible model** — one of 10 such false infeasibility claims across the 89-instance set. Fixed by the Harris two-pass form (Koberstein 2005 §3.3, the dual counterpart of §3's primal test): pass 1 relaxes each ratio by a dual-feasibility tolerance, pass 2 takes the largest pivot in that set. False infeasibility claims went 10 → 0. Pinned by `dual_simplex_ratio_test_rejects_tiny_pivots`.
2. **Termination measured in scaled units, verification in original units.** The dual stopped when every basic variable was within tolerance of its bounds *in Ruiz-scaled space*, while `finalize_result` verifies in original units — a 1e-9 scaled violation is $C_j$ times that once unscaled. Fixed by applying `unscale_factor` in the leaving-row test, so the termination criterion and the verification gate measure the same thing.
3. **Termination decided on drifted values.** `value_` is carried forward by incremental pivot updates between refactorizations. Both terminal conclusions — `OPTIMAL` *and* `INFEASIBLE` — were being decided on those drifted numbers. **MEASURED on `grow15`:** the drifted values passed the feasibility test while values re-derived from the same basis were 1.2e-05 infeasible in original units. Both exits now refactorize, re-derive, and re-scan before concluding; an infeasibility claim decided on drift would be a wrong answer, not merely a slow one.

**Open item, not fixed:** under explicit `DUAL`, `pilot87` reports an objective differing from the primal path's by more than 1e-6 relative while passing its own verification gate. The shipping path is unaffected (`AUTO` is primal), so this is recorded rather than worked around.

This is intentionally a small, explicit decision table rather than a learned or heuristic-scored selector (cf. SOTA.md §1.3's rejection of learned branching/selection for v1 on auditability grounds) — every selection is traceable to a stated structural reason. **RESEARCH HYPOTHESIS, not yet validated:** that this simple rule captures most of the achievable benefit versus a more elaborate model-characteristic classifier; revisit if benchmarking (Level 6) shows a large fraction of solves taking the "wrong" branch.

## 3. Degeneracy Handling (mandatory, not optional)

Per SOTA.md §1.4.2 and the project's own ASSUMPTION that refinery scheduling/blending models are frequently degenerate (interchangeable units/periods), the v1 simplex core includes anti-degeneracy machinery from the start rather than as a later optimization:

- **Ratio test:** Harris two-pass ratio test (SOTA.md §1.4.2, ESTABLISHED METHOD) — first pass computes a relaxed step-length bound tolerating a small, bounded constraint violation; second pass selects, among rows within that bound, the largest-magnitude pivot. This directly counters the numerically fragile "exact tightest ratio" tie-breaking that a naive ratio test would use.
- **Pricing:** Devex pricing (SOTA.md §1.4.2, ESTABLISHED METHOD) as the v1 default — a cheaper approximation to exact steepest-edge that still substantially reduces degenerate-tie iteration counts relative to naive Dantzig pricing, at lower per-iteration bookkeeping cost than full steepest-edge. **IMPLEMENTATION DECISION:** start with Devex; steepest-edge (Goldfarb & Reid 1977 / Forrest & Goldfarb 1992) is a candidate upgrade if benchmarking shows Devex's approximation quality is insufficient on refinery-structured degenerate instances — not built preemptively.
- **Anti-cycling fallback:** Bland's rule (smallest-index tie-breaking, SOTA.md §1.4.2) as the guaranteed-termination fallback when the Harris/Devex combination still stalls beyond a configured iteration count — Bland's rule is slow but is the only one of the surveyed methods with a finite-termination proof, making it the correct last resort rather than the default (which would sacrifice speed everywhere to guard against a failure mode Harris+Devex already handles in the common case).

## 3.1 Initial basis (crash) — implemented

**ESTABLISHED METHOD:** Bixby, "Implementing the Simplex Method: The Initial Basis," *ORSA Journal on Computing* 4(4), 1992.

The naive start seats an artificial in every row, so phase 1 must drive out one artificial per row — at least $m$ pivots before phase 2 can begin, and in practice far more. Netlib's own index records Bixby measuring 465,810 phase-1 iterations on `dfl001` ($m = 6071$) from exactly this cause.

An inequality row's slack already has room to absorb its own residual (`L` row: $[0,+\infty)$; `G` row: $(-\infty,0]$). Wherever the required residual falls inside the slack's bounds, that row starts **feasible** with the slack basic and needs no artificial. Only equalities and violated inequalities get one. The resulting basis is still one unit column per row, so $B$ stays diagonal and the initial factorization stays trivial.

Two conditions are enforced, both discovered by test failures rather than anticipated:

- **The slack must have room to move, not merely cover the residual.** An equality row's slack is fixed at $[0,0]$; crashing that into the basis seats a variable that can never change, so every ratio test through it forces a zero step and the basis is degenerate from iteration one. This made `grow7` fail verification (`NUMERICAL_FAILURE`) until the freedom test was added.
- **An artificial displaced by a crashed slack is frozen at $[0,0]$ for phase 1.** Under the all-artificial start every artificial began *basic*; under a crash basis many begin *nonbasic* and therefore remain eligible to **enter**, reintroducing infeasibility into a row that started feasible. Freezing is sound because the ratio test keeps the basic slack within its own bounds, so the row stays satisfied throughout phase 1.

## 4. Basis Management and Factorization

- **Representation:** LU factorization of the basis matrix, maintained via sparsity-preserving updates (Bartels & Golub 1969 / Forrest & Tomlin 1972 style — SOTA.md §1.1, ESTABLISHED METHOD) rather than full re-factorization at every pivot.
- **Periodic re-factorization:** triggered on a fixed iteration count *and* on the condition-number-monitoring signal from `NUMERICS.md` §5 — whichever comes first. This bounds both the fill-in growth from repeated updates and the numerical drift that motivates `NUMERICS.md`'s continuous condition monitoring.
- **Pivoting for the factorization itself:** Markowitz threshold pivoting (SOTA.md §1.4.1, ESTABLISHED METHOD) balancing fill-in against numerical stability.

### 4.1 Implementation status (Phase 3) — `src/lp/BasisFactorization.{hpp,cpp}`

Implemented and integrated into `Simplex`. Three deviations from the specification above are deliberate and are recorded here rather than left as silent gaps:

| Specified | Implemented | Why |
|---|---|---|
| Bartels–Golub / Forrest–Tomlin update | **Product form of the inverse** (Dantzig & Orchard-Hays 1954), eta file discarded at each refactorization | PFI is markedly easier to make numerically correct, and its known weakness — eta-file growth — is already bounded by the periodic-refactorization policy this section mandates. Forrest–Tomlin is the correct upgrade **if** measurement shows eta application, not factorization, dominates. That is an experiment to run, not an assumption to act on. |
| Markowitz threshold pivoting | **Gilbert–Peierls left-looking LU** (SIAM J. Sci. Stat. Comput. 9(5), 1988) with threshold partial pivoting (factor 0.01) and a row-count sparsity tie-break; columns pre-ordered by ascending nonzero count | Gilbert–Peierls costs time proportional to the arithmetic actually performed. The ascending-nnz column order makes the many unit columns (slacks, artificials) pivot immediately, which triangularizes most of a simplex basis before any general elimination — a cheap stand-in for the explicit triangularization phase production codes run first. |
| Refactorize on iteration count | Refactorize on **accumulated eta count** | PFI solve cost and numerical drift both track the eta file. Bound-flip iterations advance the iteration counter but push no eta, so an iteration-based trigger refactorizes at the wrong times. |

**Basis repair** (`NUMERICS.md` §5) is implemented here rather than deferred: a dependent basis column is reported by the factorization (not thrown), the unmatched row's artificial is substituted, and the basis is refactorized once. A second failure is reported as `NUMERICAL_FAILURE` rather than retried.

**EXPERIMENTAL RESULT — what this replaced and what it bought.** The prior implementation stored $B^{-1}$ explicitly as a dense $m \times m$ array: $O(m^2)$ memory and per-pivot work, $O(m^3)$ refactorization. Measured effect of the replacement, same machine, same instances, Netlib feasible set at row cap 1300, **77/77 passing in both cases** (so this is a pure speed comparison at equal correctness):

| Instance | rows | dense inverse | sparse LU | speedup |
|---|---|---|---|---|
| `pilot.ja` | 940 | 12.344 s | 0.869 s | 14.2× |
| `25fv47` | 821 | 4.742 s | 0.380 s | 12.5× |
| `d6cube` | 415 | 46.936 s | 4.433 s | 10.6× |
| `pilot.we` | 722 | 4.222 s | 0.501 s | 8.4× |
| `maros` | 846 | 0.679 s | 0.118 s | 5.8× |
| `pilotnov` | 975 | 1.884 s | 0.380 s | 5.0× |

The `d6cube` row also closes an open regression previously recorded in `NUMERICS.md` §2 (Ruiz scaling had pushed it 5.7 s → 46.9 s); the sparse factorization removes it, which indicates the regression was fill/inverse-update cost rather than anything to do with scaling.

The asymptotic point matters more than any single row above: at $m = 6072$ (`dfl001`) the dense inverse alone is 295 MB with ~$2.2\times10^{11}$ flops per refactorization, and at $m = 105127$ (`ken-18`) it is 88 TB. Those instances were not slow — they were unreachable in principle.

**Reachability, measured on the five instances that had been excluded by the 1300-row cap:**

| Instance | rows | cols | dense inverse | sparse LU |
|---|---|---|---|---|
| `bnl2` | 2324 | 3489 | not solved in 900 s | **1.37 s**, verified optimal |
| `greenbea` | 2392 | 5405 | not solved in 900 s | **1.97 s**, verified optimal |
| `d2q06c` | 2171 | 5167 | not solved in 900 s | **4.66 s**, verified optimal |
| `pilot87` | 2030 | 4883 | not solved in 900 s | **14.76 s**, verified optimal |
| `dfl001` | 6071 | 12230 | not solved in 900 s | `ITERATION_LIMIT` at 765 s (~549k iterations, ~1.4 ms/iter) |

Baseline: a single 900 s run over all five completed **0 of 5** (output was still buffered when the timeout killed it — nothing had finished). After: **4 of 5** verified optimal, three of them in under 5 s.

**`dfl001` is an open failure and is recorded as one.** It is not a surprise: Netlib's own index file documents it as pathological — Bixby reports 465,810 phase-I iterations under reduced-cost pricing, 94,337 with CPLEX defaults, and 25,803 only after switching to steepest-edge pricing *and* a different scaling, calling it "a nasty problem." Our run exhausted its iteration cap rather than stalling numerically, and per-iteration throughput was healthy, so the deficit is iteration *count*, not iteration cost. The two documented levers — presolve (`SYSTEM.md` §2.3, designed but not implemented) and pricing strength beyond Devex's steepest-edge approximation — are exactly the ones Bixby's numbers implicate. Neither is yet built; until one is, `dfl001` stays a known failure rather than a tuned-around one.

Two further observations, recorded rather than resolved: `d2q06c` (1.99e-07) and `pilot87` (3.26e-07) pass but sit an order of magnitude closer to the 1e-6 acceptance threshold than the rest of the set, which is consistent with `pilot87`'s documented bad scaling (Lustig, quoted in the Netlib index) and warrants attention when accuracy targets are tightened toward `NUMERICS.md` §3's 1e-8 goal.

## 5. Interface

```cpp
struct LPWarmStartContext {
    std::optional<Basis> parent_basis;
    bool has_parent_basis;
    bool parent_basis_dual_feasible;
};

struct LPResult {
    SolveStatus status;          // NUMERICS.md §6
    std::vector<double> x, y_dual, s_reduced_cost;
    Basis basis;                 // handed to children as their warm start
    int iterations;
    Residuals residuals;         // NUMERICS.md §3, in original-model units
};

class LPEngine {
public:
    LPResult solve(const SparseMatrixCSR& A, const SparseMatrixCSR& A_T,
                    const Bounds& bounds, const Objective& c,
                    const LPWarmStartContext& warm_start);
private:
    // owns: current basis, LU factors, Devex reference weights,
    // scratch vectors from the node-scratch arena (MEMORY.md §3.1 tier 2)
};
```

`LPEngine` does not own the matrix — it receives a view into the device/host-resident `SparseMatrixCSR`/`CSC` pair owned by `SolveContext` (`SYSTEM.md` §3), since the same matrix (or a node-local restriction of it) is shared across every node in the B&B tree. This avoids the zero-allocation-during-solve violation that would result from copying the matrix per node.

## 6. What v1 Explicitly Does Not Include

QP support (prompt.md's stated long-term target includes QP) is **not** part of the v1 LP engine. A QP extension requires either an active-set method built on this same simplex machinery or a barrier method handling the Hessian term — both are deferred alongside barrier/IPM (§1) until the LP core is validated. This is stated explicitly rather than left as a silent gap, per prompt.md's instruction to distinguish what is built from what is claimed.
