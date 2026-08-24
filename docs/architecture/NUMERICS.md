# Numerical Reliability Architecture

**Status:** PHASE 2 architecture. Per prompt.md §2.5, numerical reliability is a first-class subsystem, not an afterthought bolted onto the LP/MILP engines. This document specifies, for every numerical algorithm in the system, its precision, stopping criterion, residual test, failure condition, and recovery mechanism — prompt.md's explicit per-algorithm requirement.

---

## 1. Precision Policy

**FP64 is the default and certified path for v1.** Every quantity that feeds a feasibility, optimality, or reduced-cost decision is computed and checked in FP64. This is an **IMPLEMENTATION DECISION**, not a performance choice: prompt.md §2.5 requires deterministic, reproducible optimization, and cuSPARSE's FP64 SpMV path (`CSR_ALG2`) is the deterministic one (`CPU_GPU.md` §2.1); the faster `ALG1` path is explicitly *not* bit-reproducible run-to-run (SOTA.md §1.3b) and is therefore excluded from any code path whose output feeds a correctness decision.

**FP32 and mixed precision are deferred, not rejected.** SOTA.md §1.3b documents mixed-precision iterative refinement (Haidar/Tomov/Dongarra/Higham, SC18 2018) as an ESTABLISHED METHOD for dense systems, with sparse extensions emerging (ACM TOMS 2023). This is a legitimate future acceleration path for factorization-heavy stages (e.g., a future IPM's normal-equations solve) — but introducing it into v1 would mean validating a second numerical code path before the first one is even benchmarked, which violates prompt.md's own incremental-implementation rule. **Deferred to a milestone after the FP64 core is validated at Level 6+ of the validation hierarchy.**

## 2. Scaling

**Method:** Ruiz equilibration (SOTA.md §1.4.1, ESTABLISHED METHOD), applied once per solve during the Scaling stage (`SYSTEM.md` §2.4).

- **Precision:** FP64.
- **Stopping criterion:** iterate the row/column rescaling until $\max_i |{\|A'_{i,:}\|_\infty - 1}| < \epsilon_{\text{ruiz}}$ and the equivalent column condition, or a fixed iteration cap is hit (whichever comes first — the cap exists because Ruiz's convergence is linear but not instantaneous, and an unbounded loop is not acceptable in a system with a hard reliability mandate).
- **Residual test:** post-scaling, verify $\max$ row/column norm of $A' = RAC$ is within $[1-\delta, 1+\delta]$ of 1; if not, the scaling did not converge to spec.
- **Failure condition:** convergence check fails after the iteration cap.
- **Recovery:** fall back to the unscaled model with a diagonal identity scaling ($R=C=I$) and flag the solve as running without equilibration — this is safer than proceeding with a partially-converged, potentially misleading scale factor.
- **Unscaling (mandatory at Solution Output):** primal solution $x = C x'$; dual solution $y = R\, y'$; reduced costs $s = C^{-1} s'$. This inverse must be applied *before* any residual is reported in original-model units — a residual computed in scaled space is not directly interpretable in the units the caller specified their model in (SOTA.md §1.4.1's documented failure mode: scaling can hide a physically large violation behind a numerically small scaled residual).

**Implementation status (Phase 3):** implemented in `src/lp/Scaling.cpp` (`compute_ruiz_scaling`, `apply_ruiz_scaling`) and integrated into `Simplex` (`src/lp/Simplex.cpp`) — scaling runs once at construction, every pivot operates in scaled space, and `finalize_result` unscales $x$, $y$, and reduced costs (via $s = C^{-1}s'$ for structural columns, $s = R\,s'$ for slack columns) before computing the §3 residuals, per this section's unscaling-before-reporting requirement. `use_ruiz_scaling` defaults to on but can be disabled per-solve to isolate scaling-related regressions. **EXPERIMENTAL RESULT:** on the Netlib feasible set, this fixed a genuine `ITER_LIMIT` failure on `maros` (846×1443) — 37.3s/68,670 iterations unscaled vs. 0.68s to a verified optimum scaled (docs/research/SOTA.md §5, H2) — the first real (not synthetic) evidence this project has for H2. `finnis`'s pre-existing near-tolerance error was unaffected (still open, undiagnosed). `d6cube`'s solve time regressed at the time (5.7s → 46.9s); that regression was subsequently **closed** by the sparse LU basis factorization (`LP.md` §4.1), which brought it to 4.4s — indicating the cost was fill-in/inverse-update work, not scaling.

## 3. Residuals, Tolerances, and Stopping Criteria

All residual formulas below use the standard relative-tolerance convention (Gill, Murray, Wright-style normalization) so that tolerances are meaningful independent of problem scale:

$$\text{primal residual} = \frac{\|Ax - b\|_\infty}{1 + \|b\|_\infty}, \qquad \text{dual residual} = \frac{\|A^Ty + s - c\|_\infty}{1 + \|c\|_\infty}, \qquad \text{complementarity} = \frac{|x^Ts|}{1 + |c^Tx|}$$

| Quantity | Precision | Default tolerance | Failure condition | Recovery |
|---|---|---|---|---|
| Primal residual $\|Ax-b\|$ | FP64, computed via the GPU SpMV primitive when device-resident (`CPU_GPU.md` §2.3), else CPU | $10^{-8}$ relative (configurable) | Exceeds tolerance after claimed termination | Re-factorize basis and re-solve; if repeated, report `NUMERICAL_FAILURE` |
| Dual residual $\|A^Ty+s-c\|$ | FP64 | $10^{-8}$ relative | Exceeds tolerance after claimed termination | Same as above |
| Complementarity $x^Ts$ | FP64 | $10^{-8}$ relative | Exceeds tolerance | Same as above |
| Feasibility tolerance (bound/row violations) | FP64 | $10^{-7}$ absolute in *original* (unscaled) units | Any variable/row outside $[l,u]$ by more than tolerance | Reject candidate solution; continue search or report infeasible |
| Optimality tolerance (objective gap, LP) | FP64 | $10^{-8}$ relative | N/A — this is a stopping criterion, not a failure mode | — |
| Reduced-cost tolerance (dual feasibility at a claimed optimum) | FP64 | $10^{-7}$ | A nonbasic variable's reduced cost has the wrong sign beyond tolerance | Basis is not actually dual-feasible — continue simplex, do not report optimal |
| Reduced-cost tolerance (MILP integrality) | — | $10^{-6}$ absolute distance from nearest integer | Fractional value exceeds tolerance | Not integer-feasible; continue B&B |

### 3.1 Implementation findings (Phase 3)

**The Harris ratio-test expansion was silently spending accuracy, and measurement — not theory — settled its width.** The two-pass Harris test (`LP.md` §3) relaxes each basic variable's bound by `kHarrisExpand` in pass 1 so pass 2 has near-tied rows to pick the largest pivot from. Nothing in this implementation ever removes the infeasibility that permits, so a basic variable can finish the solve that far outside its bound and the terminal basis is not quite primal feasible.

| expansion | Netlib pass rate | total iterations | total sec | `pilot87` rel. err |
|---|---|---|---|---|
| 1e-7 | 88 / 89 | 195,966 | 34.72 | 1.07e-06 **FAIL** |
| 1e-9 | **89 / 89** | 181,418 | 34.15 | 2.16e-07 PASS |

At 1e-7, `pilot87` terminated with a structural variable 1.83e-07 below its lower bound. That is *inside* the §6 verification gate, so it was reported `OPTIMAL` — but the objective error it produced exceeded the 1e-6 threshold the Netlib validation applies, which is how it was caught at all. Tightening to 1e-9 cost nothing: iteration count **fell** 7.4% across the set and no instance regressed more than 25%. The classical stability argument for a wide expansion (Gill, Murray, Saunders & Wright 1989) does not govern at this scale — the two-pass structure itself supplies the pivot choice, and the extra width mainly bought accumulated infeasibility. The EXPAND shift-and-reset machinery remains the fallback if a future instance stalls at this width.

**A tolerance that stops the algorithm must not equal the tolerance that verifies it.** The dual simplex originally terminated at `kFinalTol` (1e-6) — the same width §6's gate rejects at — leaving nothing between "the algorithm stopped" and "the result failed verification". Termination thresholds are now set at 1e-9, two orders inside the gate.

**OPEN FINDING — the row-residual normalization is scale-blind, and it currently rejects sound bases.** §3's primal residual normalizes row violations by $1 + \|b\|_\infty$ alone. On a model whose *row activities* are far larger than its rhs, that denominator is not representative of the arithmetic actually performed. **MEASURED on `grow15`:** the dual path terminates with every basic variable exactly within bounds (violation 0.0 in original units), yet `finalize_result` reports a 1.24e-05 primal residual, all of it from one row whose terms are ~1e6 and cancel down to a small rhs. Relative to that row's own activity the error is ~1e-11 — double-precision round-off, unimprovable by any algorithmic change — but the gate rejects it, the result is downgraded to `NUMERICAL_FAILURE`, and `AUTO` falls back to the primal path. No wrong answer is produced; work is wasted and a sound basis is discarded.

The fix is a per-row relative denominator (normalizing by $1 + |(Ax)_i|$, so the test stays absolute-strict on small-activity rows and only relaxes where the row's own magnitude justifies it). It is **not applied here**, deliberately: it changes the acceptance criterion for every result the engine reports, and per prompt.md's development rule that belongs in its own increment with its own measurement, not folded into the dual simplex's. Recorded as a known defect in the metric rather than left as an unexplained fallback.

**Every tolerance is configurable** (`SolverOptions`), but the *defaults* above are the ones this architecture certifies against — any solve run with looser tolerances than default must have that fact recorded in the solution's metadata, so a benchmark result is never silently compared against a different accuracy target (directly serving prompt.md's Benchmark Strategy requirement to report accuracy alongside speed).

### 3.2 Validation coverage, extended to 20,000 rows

The tolerance results in §3.1 were established at a 2,600-row cap, which excluded the largest models in the Netlib set. Re-running `validate_netlib` at a **20,000-row cap**, single process with nothing else running:

**92 / 93 validated instances pass** (18 instances have no published reference value; 3 remain above the cap). The single failure is `dfl001`, which reaches the iteration limit after 572,139 iterations and 792 s.

The instances this cap newly admits all pass, and several are substantially larger than anything §3.1 covered:

| instance | rows | cols | solve time | rel. error |
|---|---|---|---|---|
| `stocfor3` | 16,675 | 15,695 | 14.34 s | 1.07e-09 |
| `dfl001` | 6,071 | 12,230 | *iteration limit* | — |
| `maros-r7` | 3,136 | 9,408 | 4.01 s | 1.40e-11 |
| `fit2p` | 3,000 | 13,525 | 6.60 s | 9.03e-10 |
| `pilot87` | 2,030 | 4,883 | 16.70 s | 2.16e-07 |
| `d6cube` | 415 | 6,184 | 5.23 s | 2.71e-11 |

Two things worth recording. First, **the 1e-9 Harris expansion chosen in §3.1 holds at four times the model size it was tuned on** — `pilot87`, the instance that forced the tolerance change, still passes at 2.16e-07, and no newly admitted instance produced a tolerance-related failure. Second, `dfl001` is the engine's one genuine correctness gap: not a wrong answer, but no answer. It is also the one instance where the first-order path (`PDLP.md` §5) succeeds where the simplex does not, reaching a verified KKT error of 8.65e-07 in 1.13 s.

## 4. Iterative Refinement

**ESTABLISHED METHOD** (classical Wilkinson-era residual correction, extended in modern mixed-precision work per SOTA.md §1.3b/§1.4.1). v1 uses **same-precision** iterative refinement only (FP64 factorization, FP64 refinement) — this is a robustness mechanism for basis conditioning issues, not a performance mechanism, in v1:

- **Precision:** FP64 throughout.
- **Stopping criterion:** residual norm decreases by at least a fixed factor per refinement step, up to a small fixed cap of steps (typically ≤ 3 — refinement that hasn't converged in a handful of steps indicates a conditioning problem refinement alone won't fix).
- **Residual test:** recompute $Ax - b$ (via the same SpMV primitive, `CPU_GPU.md` §2.3) after each correction step.
- **Failure condition:** residual fails to decrease, or decreases too slowly, across the step cap.
- **Recovery:** escalate to full re-factorization (fresh LU/basis factorization rather than an update) — a stale, ill-updated factorization is a common cause of refinement stalling; if re-factorization also fails to restore residual quality, escalate further to basis repair (§5).

## 5. Numerical Failure Detection and Basis Repair

**Failure detection is continuous, not a one-time end check.** Every simplex iteration checks: (a) whether the current basis factorization's estimated condition number (Hager-style estimator, SOTA.md §1.4.1) has crossed a configurable threshold, and (b) whether the Harris ratio test (`LP.md`) is being forced into degenerate zero-length pivots beyond a stall-count threshold.

- **Basis repair mechanism:** when condition-number monitoring or repeated refinement failure (§4) indicates the current basis factorization is unreliable, the engine performs a full re-factorization from the current basis (not from scratch — the *basis choice* is preserved, only its numerical factorization is redone). If the basis itself is suspected structurally degenerate (persistent cycling/stalling despite anti-degeneracy pivoting, `LP.md` §3), the engine falls back to a perturbed restart (small, bounded perturbation to break exact ties, per SOTA.md §1.4.2's discussion of perturbation methods) rather than an unbounded retry loop.
- **Fallback algorithm chain (implemented, `Simplex::solve`):** dual simplex → primal simplex → reported failure. Implemented exactly as specified, with one clarification measurement forced: the dual path is entered only when `LpAlgorithm::DUAL` is selected, because `AUTO` resolves to primal on cold starts per `LP.md` §2.1's decision table and its supporting benchmark. When the dual path is entered and its terminal basis fails §6's verification, the primal path runs and produces the reported answer; `LpResult::used_dual_simplex` records which algorithm the reported result came from, and `dual_iterations` still counts the discarded work so a benchmark cannot understate the cost of a fallback. Original spec text follows: dual simplex (default) → primal simplex (if dual simplex's Phase 1 cannot establish a feasible starting basis, or repeatedly fails numerically) → report `NUMERICAL_FAILURE` with full diagnostic state (last residuals, iteration count, condition estimate) rather than silently returning a degraded/unverified result. There is no v1 fallback to a first-order method (PDLP-style) — that path is deferred per SOTA.md's KS-7 decision, so the fallback chain terminates in an explicit failure status, not a silent accuracy downgrade.

## 6. The Hard Invariant

Restated because it is the single most important rule in this document: **a solver status of `OPTIMAL` (or `INFEASIBLE`, or `UNBOUNDED`) is only ever reported after the relevant candidate result has passed the residual checks in §3, in original (unscaled) model units.** Reaching an iteration limit, or an algorithm's internal termination flag, is necessary but never sufficient. A termination that fails verification is reported as `NUMERICAL_FAILURE` or `ITERATION_LIMIT` — distinct statuses from `OPTIMAL`, both of which are legitimate, honestly-reported outcomes rather than something to be hidden from the caller or from benchmark reporting.

```cpp
enum class SolveStatus {
    OPTIMAL,            // terminated AND passed §3 verification
    INFEASIBLE,         // proven infeasible AND verified
    UNBOUNDED,          // proven unbounded AND verified
    NUMERICAL_FAILURE,  // termination reached but verification failed, or repair exhausted
    ITERATION_LIMIT,    // stopped by iteration/time budget, unverified — NOT optimal
};
```
