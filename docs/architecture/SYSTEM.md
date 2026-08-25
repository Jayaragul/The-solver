# System Architecture — Core Optimization Pipeline

**Status:** PHASE 2 — architecture only; no implementation exists yet. This document defines module ownership and interfaces; Phase 3 will implement them incrementally, one internally-consistent subsystem at a time, per prompt.md's development rule.

**Dependency discipline (carried forward from Phase 1 / prompt.md §1):** every module below is implemented from first principles. The only external dependencies permitted anywhere in this pipeline are CUDA Runtime/Driver, cuSPARSE, cuBLAS, cuSOLVER (as numerical *primitives*, never as solvers), and the C++ standard library. No existing LP/MILP/QP solver, wrapper, or optimization-engine library is used at any stage.

---

## 1. Pipeline Overview

```text
Model Input
    ↓
Model Validation
    ↓
Presolve
    ↓
Scaling
    ↓
Matrix Transformation
    ↓
LP Engine
    ↓
MILP Engine
    ↓
Branch-and-Bound
    ↓
Cut Management
    ↓
Primal Heuristics
    ↓
Incumbent Management
    ↓
Numerical Verification
    ↓
Solution Output
```

This is a directed pipeline for a single top-level solve, but three stages are re-entrant: **LP Engine** is invoked once per B&B node (thousands of times per solve), **Cut Management** and **Primal Heuristics** are invoked per node or per node-batch, and **Numerical Verification** runs both per-LP-solve (cheap check) and once at the end (full certificate, per prompt.md §2.5: "no result accepted solely because an iteration count was reached"). The diagram shows data flow for model-to-solution; it is not a call-once linear schedule.

---

## 2. Module Definitions

Each module is specified as: **owns** (data it is the sole mutator of), **consumes**, **produces**, **CPU/GPU residency**, **interface sketch**. Interface sketches are architecture-level signatures for Phase 3 implementation — not implementation.

### 2.1 Model Input

**Owns:** the raw problem description as given by the caller (variable bounds, objective, constraint matrix in triplet/COO form, integrality flags, QP Hessian if present).
**Consumes:** external model description (format TBD in a later modeling-layer decision — out of scope for the numerical core).
**Produces:** `RawModel` — a value type holding dimensions ($m$ rows, $n$ columns), COO-format $A$, dense $b$, $c$, bounds $l \le x \le u$, integrality mask, optional QP Hessian $Q$ (COO, upper-triangular).
**Residency:** host, pageable memory (one-time, low-reuse — no motivation for pinning here per the reuse-opportunity criterion in `CPU_GPU.md` §2).

```cpp
struct RawModel {
    Index n_rows, n_cols;
    std::vector<Triplet> A_coo;         // (row, col, value)
    std::vector<double>  b, c, l, u;
    std::vector<VarKind> integrality;   // CONTINUOUS | INTEGER | BINARY
    std::optional<std::vector<Triplet>> Q_coo; // QP Hessian, upper triangular
};
```

### 2.2 Model Validation

**Owns:** nothing persistent — a pure checking pass.
**Consumes:** `RawModel`.
**Produces:** `ValidationResult` (pass, or a specific structural error: dimension mismatch, $l > u$, duplicate triplet entries policy, non-finite coefficients, dangling row/column references).
**Rationale:** per prompt.md §2.5 ("no result accepted solely because an iteration count was reached"), the same discipline applies at the front door — the solver must never silently proceed on a structurally invalid model. This is an **IMPLEMENTATION DECISION**: fail fast, with a specific diagnostic, rather than attempting to repair an invalid model.

### 2.3 Presolve

**Owns:** the presolve transformation log (`PresolveMapping`) — the reversible record required by prompt.md §2.6 to reconstruct the original-space solution.
**Consumes:** validated `RawModel`.
**Produces:** `PresolvedModel` (reduced $A, b, c, l, u$) + `PresolveMapping` (append-only log of applied reductions, in application order, each carrying exactly the data needed to invert it).
**Residency:** CPU only. Presolve is inherently control-flow-heavy and irregular (row/column scans, fixed-point iteration over reduction rules) — no GPU candidacy per the arithmetic-intensity criterion in `CPU_GPU.md`.
**Reversibility contract:** each reduction rule (singleton row/column elimination, bound tightening, fixed-variable substitution, redundant-row detection, empty row/column removal — per prompt.md §2.6) appends one `PresolveStep` variant to the log. Postsolve (§2.13 below) replays the log in reverse to map a `PresolvedModel` solution back to `RawModel` space. This mapping is the correctness-critical contract of this module: **a presolve reduction that cannot be exactly inverted must not be applied**, full stop — this is a hard invariant, not a performance tradeoff.

```cpp
class PresolveEngine {
public:
    struct Result { PresolvedModel model; PresolveMapping mapping; };
    Result run(const RawModel&);
private:
    // fixed-point loop over reduction rules until no rule fires
};
```

### 2.3.1 Implementation status (Phase 3) — `src/lp/Presolve.{hpp,cpp}`

Implemented, with the reversibility contract honoured by construction: `PresolveResult` records `kept_columns`, `kept_rows`, and the fixed value of every removed column *at the moment the reduction fires*, and `postsolve()` reconstructs the original-space solution from that record. `src/lp/LpSolver.cpp` composes presolve → scale → simplex → postsolve and then **re-verifies primal feasibility against the original model**, so a reduction that was not exactly invertible surfaces as `NUMERICAL_FAILURE` rather than as a clean optimum.

Reductions implemented: empty row, empty column, fixed column, singleton row, redundant row, forcing row, and activity-based bound propagation. Deliberately deferred: doubleton-equation and free-column-singleton substitution — both change the sparsity pattern and need a materially harder postsolve, and §2.3 makes exact invertibility a hard invariant rather than a target.

**Three numerical-policy findings, each discovered by a failing test rather than reasoned about in advance.** They are recorded because each is a trap that a naive presolve falls into, and the reasoning generalizes:

1. **Activity must never be cached across a pass.** Fixing a column updates the row bounds of every row it appears in; a cached activity snapshot then disagrees with the updated bounds, and comparing the two over-tightens. Activity is therefore recomputed per row, on demand. (Same O(nnz) cost either way.)

2. **Reduction tolerances and infeasibility tolerances point in opposite directions and must not share a constant.** Declaring INFEASIBLE wrongly destroys a solvable model, so that test is *generous*. Firing a reduction wrongly silently changes the problem, so those tests are *conservative*. Sharing one constant — and scaling it by total activity magnitude, which can be orders larger than the row itself — made `bore3d` fail: forcing-row fired on rows with genuine slack, and eight passes later an emptied row no longer contained zero.

3. **Presolve must never conclude INFEASIBLE from a derived bound.** `rest_min`/`rest_max` subtract one term from a sum accumulated over a whole row; on a row mixing large and small coefficients that cancellation can exceed any fixed tolerance. `maros` fails exactly this way — propagation on row 161 crosses column 822's bounds on a provably feasible model. Since infeasibility detection in presolve is an *optimization* (simplex phase 1 detects it independently from the unreduced row), the resolution is to **decline the reduction and keep the row**. A missed tightening costs speed; a false infeasibility costs correctness.

A `reason` / `reason_row` / `reason_col` / `reason_pass` field on `PresolveResult` reports which reduction concluded infeasibility and where. Without it a buggy reduction and a genuinely infeasible model are indistinguishable — all three findings above were located with it.

### 2.4 Scaling

**Owns:** row/column scale factor vectors $r \in \mathbb{R}^m$, $c \in \mathbb{R}^n$.
**Consumes:** `PresolvedModel`.
**Produces:** `ScaledModel` ($A' = R A C$, $b' = Rb$, $c'_{\text{obj}} = Cc_{\text{obj}}$, $l' = C^{-1}l$, $u' = C^{-1}u$) + the scale factors themselves (needed to unscale the solution — see `NUMERICS.md` §2).
**Method (per SOTA.md §1.4.1, ESTABLISHED METHOD):** Ruiz equilibration — iterative row/column rescaling by the reciprocal square root of the row/column ∞-norm, until convergence (Ruiz, RAL-TR-2001-034, 2001).
**Residency:** CPU. Each Ruiz iteration is $O(\text{nnz})$ and the whole procedure runs a small, bounded number of iterations once per solve — not large enough or repeated enough to amortize a kernel launch; this is an explicit **IMPLEMENTATION DECISION**, not an oversight (contrast with the SpMV hot loop in §2.6, which *is* GPU-resident).

**Implemented** (`src/lp/Scaling.cpp`, integrated into `Simplex`) — see `NUMERICS.md` §2 for the implementation-status note and the `maros` empirical result it produced.

### 2.5 Matrix Transformation

**Owns:** the dual CSR/CSC representation of $A'$ (and $A'^T$ where needed) that the LP engine actually consumes.
**Consumes:** `ScaledModel`.
**Produces:** `SparseMatrixCSR`, `SparseMatrixCSC` (see `src/sparse/` in Phase 3) — both built once, since cuSPARSE SpMV is ~3× faster non-transposed than transposed (SOTA.md §1.3b), so both $Ax$ and $A^Ty$ get a native-orientation representation rather than paying the transposed-SpMV penalty on every call.
**Residency:** built on CPU, then a device-resident copy is created (per `MEMORY.md` §3) for GPU-accelerated SpMV. Host and device copies have independent, explicitly-tracked lifetimes — never implicitly synchronized.

### 2.6 LP Engine

**Owns:** the current basis, factorization state, and simplex iterate for whichever LP is currently being solved (root relaxation or a B&B node relaxation).
**Consumes:** a `SparseMatrixCSR`/`CSC` pair, bounds, objective, and — for a warm start — a parent basis.
**Produces:** `LPResult { status, x, y_dual, s_reduced_cost, basis, iteration_count, residuals }`.
**Full design:** `LP.md`. **Residency:** control flow (pivoting, ratio test, pricing) is CPU-resident per SOTA.md §1.3b's literature evidence against GPU simplex; SpMV-shaped subcomputations (e.g., residual verification, and any future first-order path) may call into the GPU primitive defined in `CPU_GPU.md`.

### 2.7 MILP Engine / 2.8 Branch-and-Bound

**Owns:** the B&B search tree (open-node structure), the global incumbent, and the global best bound.
**Consumes:** repeated `LPResult`s from the LP Engine (one per node), branching decisions, cut pool state.
**Produces:** the final `Solution` or an explicit non-optimal status, plus solve statistics (node count, best bound, gap history) required for benchmarking per prompt.md's Benchmark Strategy. The first implementation is `src/milp/` and uses deterministic best-bound branch-and-bound with reliability branching, certified simplex relaxations, root cover separation, and incumbent verification.
**Full design:** `MILP.md`. **Hard architectural constraint (prompt.md, non-negotiable):** branch-and-bound control flow — node creation, node selection, branching-variable choice, incumbent bookkeeping — is 100% CPU-resident. The GPU never controls the tree; it may only accelerate arithmetic *inside* a node's LP solve, per the responsibility matrix in `CPU_GPU.md`.

### 2.9 Cut Management

**Owns:** the cut pool (generated valid inequalities, active/inactive status, age-based eviction policy).
**Consumes:** a fractional LP solution at a B&B node.
**Produces:** violated cuts to append to the node's local relaxation.
**Residency:** CPU (separation is control-flow-heavy; see SOTA.md kill-shot KS-5). The current implementation separates root-only pure-binary cover inequalities; mixed-row MIR/flow-cover separation remains disabled until its validity transformations and benchmarks are independently validated.

### 2.10 Primal Heuristics

**Owns:** nothing persistent — invoked on demand with a read-only view of the current relaxation and incumbent.
**Consumes:** fractional LP solution.
**Produces:** candidate integer-feasible solutions (offered to Incumbent Management).
**v1 scope note:** ships with a trivial rounding heuristic only; RENS/feasibility-pump/diving-class heuristics are deferred (SOTA.md flags refinery-specific heuristic effectiveness as an untested ASSUMPTION — §1.1).

### 2.11 Incumbent Management

**Owns:** the single global best feasible solution and its objective value; the resulting optimality gap.
**Consumes:** candidate solutions from B&B leaf nodes and from Primal Heuristics.
**Produces:** updated incumbent, updated gap, and the trigger for the "time to first incumbent" / "time to target gap" KPIs in prompt.md's benchmark strategy.
**Concurrency note:** single-writer (the B&B control loop is single-threaded in v1 — see `MILP.md` for the explicit non-goal of intra-tree parallelism in the first milestone); no synchronization primitives are needed until a parallel B&B milestone is scoped.

### 2.12 Numerical Verification

**Owns:** nothing persistent — a stateless checking pass, but a *mandatory* one.
**Consumes:** a candidate `LPResult` or final `Solution`.
**Produces:** a verification verdict: primal residual $\|Ax - b\|$, dual residual $\|A^Ty + s - c\|$, complementarity $x^Ts$, bound feasibility, integrality feasibility (for MILP) — each checked against an explicitly justified tolerance (full detail in `NUMERICS.md`).
**Hard invariant (prompt.md §2.5, restated as an architectural gate):** no status is ever reported as `OPTIMAL` because an iteration limit or algorithmic termination flag was hit — it is reported as `OPTIMAL` only if it *also* passes this module's residual checks. A termination that fails verification is reported as `NUMERICAL_FAILURE`, not silently upgraded.

### 2.13 Solution Output

**Owns:** nothing persistent.
**Consumes:** the verified solution in *presolved* space, plus `PresolveMapping` and the scaling factors from §2.3–2.4.
**Produces:** the final solution in **original model space** — this is postsolve: replay `PresolveMapping` in reverse, then unscale ($x = Cx'$, dual $y = R y'$) per `NUMERICS.md` §2.
**Correctness-critical:** this is the module that makes §2.3's reversibility contract matter. A presolve reduction the mapping can't invert corrupts every solution the solver ever returns — this is why §2.3 treats invertibility as a hard invariant rather than a nice-to-have.

---

## 3. Interfaces Between Modules

All inter-module data (`RawModel`, `PresolvedModel`, `ScaledModel`, `LPResult`, `Solution`) are **value types with explicit ownership** — no module reaches into another module's private state. The `PresolveMapping` and scale factors are the only pieces of state that must outlive their producing module's call (they are needed again at Solution Output) — these are threaded through explicitly as part of a `SolveContext` owned by the top-level driver, not stored as hidden global/static state (prompt.md §3.10: "no hidden global state").

```cpp
class SolveContext {
public:
    Solution solve(const RawModel& model, const SolverOptions& opts);
private:
    PresolveMapping presolve_mapping_;
    ScaleFactors     scale_factors_;
    MemoryArena      arena_;           // see src/memory/MemoryArena.hpp, Phase 3
    // owns device-resident matrix/vector buffers for the solve's lifetime
};
```

This satisfies prompt.md §3.1's requirement that all solve-lifetime allocation happen during initialization: `SolveContext::solve()` sizes and populates `arena_` once (after presolve/scaling determine final problem dimensions), and no module downstream of that point performs unplanned heap or `cudaMalloc` allocation.

---

## 4. What This Document Deliberately Does Not Do

Per prompt.md's "do not optimize for appearance" principle, this document does not include a UML diagram, a plugin-architecture abstraction layer, or speculative extensibility hooks for algorithms not yet built. Every interface above exists because a named module (§2.1–2.13) needs it now, for the v1 scope defined in `LP.md` and `MILP.md`. Extending the pipeline (e.g., adding a first-order LP path, per SOTA.md's deferred KS-7) means adding a new module with the same ownership discipline — not retrofitting speculative generality now.
