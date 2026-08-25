# SANKHYA — Sovereign Optimization Solver Core
## 48-Hour Hackathon Execution Plan — v2 (rescoped after technical review)

*(Name is a placeholder — सांख्य, "analytical reckoning". Change freely.)*

> **v2 changelog.** Scope narrowed to one deep LP core + minimal B&B + PDHG (LP and QP)
> after review found v1 over-scoped. Mehrotra QP-IPM and all cutting planes moved to the
> roadmap. Several quantitative claims removed as undefendable. Verification upgraded from
> "recompute feasibility" to a full primal-dual certificate with exact rational checking.
> v1 preserved at `PLAN.v1.md` for the record.

---

## 0. The honest framing (read this first)

We are **not** going to beat Gurobi in 48 hours. CPLEX is ~38 years of engineering; HiGHS is
a funded research group over many years. Any team claiming otherwise gets dismantled by the
first judge who has actually used a solver.

We win on a different axis:

> **A from-scratch LP core whose optimality claims are independently verified — for many
> instances by an exact rational certificate check that trusts neither our solver nor any
> reference value — extended by a GPU first-order engine for scale, and improved by an
> autonomous experimentation loop in which correctness is a hard gate rather than a scored
> objective.**

The problem statement asks for a "transparent, extensible and sovereign **foundation**."
*Foundation*, not replacement. We answer the question actually asked.

### Claims we will not make

| Will not say | Because |
|---|---|
| "Faster than Gurobi / CPLEX" | False, and instantly disprovable |
| "Production ready" | 48 hours |
| "Dependency-free" | The CUDA path depends on the NVIDIA driver and runtime. See §4. |
| "Solves millions of variables" | We state the measured size of the specific instance class |
| "C++ is 30–50× faster than Python" | Not a property of the languages. See §2. |
| "Autotuning typically returns 1.2–2×" | Asserted without evidence in v1. It is a *target*, not a known fact. |
| "No foreign solver has an India-industrial profile" | Unfalsifiable without surveying every vendor. Removed. |

---

## 1. Hard constraints

| Constraint | Consequence |
|---|---|
| 48 h wall clock | **The dominant constraint.** Drove the v2 rescope. |
| No compiler installed | ~90 min of Hour 0 is toolchain install. Budgeted. |
| 4 GB VRAM (RTX 3050 laptop) | GPU work must be sparse and matrix-free. Rules out GPU dense factorization; rules in first-order methods. |
| Teammates have no GPU | GPU is a single-machine workstream. Everything else is CPU/Python and parallelizable. |
| 33 GB free on C: | Toolchain on C:, all data on R:. Curated MIPLIB subset only. |
| 16 threads, 16 GB RAM | Real multi-core story available; instance size capped, peak RSS instrumented. |
| One primary implementer | The real reason v1 was over-scoped. |

---

## 2. Scope — v2

### Tier 1 — LP core (deep, verified, the centre of the project)

Primal and dual revised simplex, bounded variables · sparse LU with Markowitz ordering and
threshold partial pivoting · product-form update with periodic refactorization · **Harris
two-pass ratio test with bound flipping** · Devex pricing · cost perturbation and Bland
fallback for anti-cycling · full presolve **with postsolve** · geometric + equilibration
scaling.

All robustness evidence comes from here. This tier gets the debugging time.

### Tier 2 — MILP (deliberately minimal, and labelled as such)

**Plain branch and bound. No cutting planes.** Dual-simplex warm start, domain propagation,
pseudocost branching, best-bound with plunging, rounding and diving heuristics.

We say "branch-and-**bound**", never "branch-and-cut". Gomory MIR cuts generated from
tableau rows with imperfect numerics can be *invalid* and cut off the true optimum — a wrong
answer, which would destroy the correctness narrative that is our whole thesis. Not worth it
at hour 30 of 48. Cuts are on the roadmap with that reasoning stated.

### Tier 3 — QP (prototype, via the PDHG engine)

Convex QP by extending PDHG with a proximal step on the quadratic term. Reuses the Tier-4
engine rather than adding a second Newton-based numerical stack. **~1e-4 relative accuracy,
labelled a prototype.** It covers the statement's stated QP scope honestly and demonstrates
the modularity claim instead of merely asserting it.

### Tier 4 — GPU (scaling prototype)

Restarted-averaged PDHG for LP with adaptive step size and primal-weight balancing.
Hand-written CUDA CSR SpMV for A and Aᵀ, plus fused vector kernels. No cuSPARSE, no cuBLAS.

### Tier 5 — Verification, benchmarking, AutoResearch

§5 and §6. This is where the project differentiates.

### Moved to the roadmap (was IN in v1)

Mehrotra predictor-corrector QP-IPM with LDLᵀ · **all cutting planes** (Gomory MIR, knapsack
cover, flow cover, clique) · dual steepest edge · Forrest–Tomlin update · parallel B&B tree ·
RINS/RENS · conflict analysis · symmetry detection · MIQP/NLP/MINLP · PDHG→simplex crossover.

Each gets a roadmap line with an effort estimate. "Knows exactly what is missing and why"
reads as competence.

### Why C++ (the defensible version)

Not a language speed claim. The reasons are: direct control over sparse data structures and
memory layout; the inner loops of a revised simplex (LU update, ratio test, pricing over
sparse columns) are irregular, branchy, pointer-chasing code that does **not** reduce to a
handful of vectorized array calls; explicit control over factorization internals, threading
and SIMD; predictable performance without interpreter or GC variance; and single-binary
deployment. A NumPy/SciPy implementation would be competitive only where the work decomposes
into large dense kernels, which is precisely what sparse simplex does not do.

---

## 3. Architecture

```
        +--------------------------------------------------------+
        |  CLI  sankhya solve model.mps --profile refinery        |
        |  C API sankhya.h  ->  Python ctypes shim                |
        +----------------------------+---------------------------+
                                     |
        +----------------------------v---------------------------+
        |  MODEL LAYER                                            |
        |  MPS (fixed+free) / QPS reader          [from scratch]  |
        |  min c'x + 0.5 x'Qx,   L <= Ax <= U,   l <= x <= u      |
        |  integrality flags, CSC/CSR sparse storage              |
        +----------------------------+---------------------------+
                                     |
        +----------------------------v---------------------------+
        |  PRESOLVE + SCALING            (+ postsolve stack)      |
        +----------------------------+---------------------------+
                                     |
        +--------------------+-------+------------------+
        |                    |                          |
+-------v----------+  +------v-----------+   +----------v-----------+
| SIMPLEX  (Tier 1)|  | B&B      (Tier 2)|   | PDHG      (Tier 3/4) |
| primal + dual    |  | plain, NO cuts   |   | CPU (OpenMP) and GPU |
| Markowitz LU     |  | warm-started dual|   | own CUDA SpMV        |
| Harris ratio test|  | propagation      |   | LP  ->  ~1e-4        |
| Devex pricing    |  | dive + round     |   | QP  ->  prototype    |
+-------+----------+  +------------------+   +----------------------+
        |
        | emits primal-dual certificate (x, y, z)
        v
+-------------------------------------------------------------------+
|  INDEPENDENT VERIFIER   (separate process, protected, never tuned) |
|  float pass:  primal feas · dual feas · complementary slackness    |
|  EXACT pass:  same checks in rational arithmetic over Q            |
+-------------------------------------------------------------------+
```

---

## 4. Sovereignty — the defensible claim

**What we claim:**

> No third-party optimization solver or numerical library appears in the solve path. Every
> optimization algorithm and every numerical kernel — LU, LDL, ratio test, SpMV, all vector
> operations — is our own implementation.

**What we do not claim:** that the binary has no external platform dependency. It links the
C++ runtime and OpenMP, and the GPU path requires the NVIDIA driver and CUDA runtime. Those
are platform infrastructure, not optimization software. Pretending otherwise invites an easy
and deserved takedown.

**Excluded from the solve path:** COIN-OR (CBC/CLP/Cgl/Osi), HiGHS, GLPK, SCIP, SoPlex,
lp_solve, Gurobi/CPLEX/Xpress, `scipy.optimize`, cuSPARSE/cuSOLVER/cuBLAS, Eigen, SuiteSparse,
MKL, any BLAS/LAPACK.

**Permitted as external reference *processes*** — separate executables invoked by the harness,
never linked, never in our address space: HiGHS, CBC, SCIP. The statement requires comparison
against an established solver; this is how we do it without contaminating the core.

**Evidence, stated as evidence:** `dumpbin /dependents` plus the linker input list plus the
build manifest. This is *supporting evidence* for the claim above — the actual proof is the
source tree, which is open. We will not say a 20-second command "closes the sovereignty
question", because it does not.

---

## 5. Verification and benchmarking

### 5.1 The primal-dual certificate — the strongest thing we build

On LP termination the solver emits `(x, y, z)`: primal solution, row duals, reduced costs.
A **separate, protected verifier process** then checks, reading only the original untouched
MPS:

1. **Primal feasibility** — `L ≤ Ax ≤ U`, `l ≤ x ≤ u`
2. **Dual feasibility** — `c − Aᵀy − z = 0`, with sign conditions on `y` per row type and on
   `z` per bound status
3. **Complementary slackness / zero duality gap** — `cᵀx` equals the dual objective

Together these are a *self-contained proof of optimality*. It trusts neither our solver's
status flag nor any published reference value.

**The exact pass.** For instances within a size budget, the verifier re-runs all three checks
in **exact rational arithmetic** (`fractions.Fraction`) from the reported basis. Where that
completes, "we believe this is optimal" becomes a machine-checked exact proof — under
floating-point-free arithmetic, with zero tolerance fudge.

**The MILP asymmetry, stated openly.** For MILP we can exactly verify the *incumbent*
(feasibility + integrality + objective). We **cannot** cheaply verify the *bound*, because
that means verifying the whole search tree. So we report MILP optimality as *claimed*,
cross-checked against HiGHS and published optima, and we say plainly that it is a weaker
guarantee than the LP certificate. Pretending the two are equivalent would be the most
likely thing to get us caught.

### 5.2 Datasets (staged on `R:\`)

| Set | Role | Notes |
|---|---|---|
| **Netlib LP** | **Immutable correctness and regression baseline.** Not a tuning target. | The canonical 98-instance suite. Literature also uses 89- and 93-instance subsets with exclusions, so we publish `bench/NETLIB_MANIFEST.txt` listing every file used, its checksum, its reference optimum and source, and any exclusion **with the reason**. `N/98` is meaningless without that manifest. |
| **Netlib Kennington + large extras** | Feeds the GPU scaling runs | |
| **MIPLIB 2017 — curated subset** | MILP credibility, modest by design | ~25 instances HiGHS solves quickly. We are a plain B&B with no cuts and we expect to look weak here. We report it anyway. |
| **QPLIB — convex continuous subset** | QP prototype validation | ~12 instances |
| **Mittelmann LP — largest feasible** | GPU scaling | |
| **Industrial families (ours)** | **The AutoResearch tuning target** | §5.3 |

### 5.3 Industrial case studies — 4 committed, 2 stretch

Python generators emitting MPS, each with a `--scale` knob so the same model yields a
small instance for correctness and a large one for the scaling demo. **Split into train and
held-out families before any tuning begins.**

**Committed:** (1) refinery product blending — LP, property blending on octane/sulfur/RVP/
viscosity · (2) crude blending / pooling — Haverly, piecewise-linear MILP relaxation ·
(3) multi-echelon supply chain — min-cost flow with capacities and fixed-charge arcs ·
(4) transportation LP — degenerate by construction, scales to millions of nonzeros, drives
the GPU demo.

**Stretch:** (5) capacitated lot-sizing with a deliberately weak big-M relaxation ·
(6) unit commitment with min up/down and ramp limits.

Weighted toward LP because that is what we are actually good at in 48 hours.

### 5.4 Harness

Subprocess runs of `{sankhya, highs, cbc} × {instance}`, hard timeout, memory cap, median of
3, randomized order, fresh process, cleared caches. Records status, objective, primal/dual
infeasibility, gap, iterations, nodes, factorizations, wall time, peak RSS. Emits SQLite →
Dolan–Moré profiles, per-instance tables, static HTML dashboard. `--fault-inject` rescales
rows by 10^k to drive up condition number and chart where each solver breaks.

**Tolerances** (fixed, protected, never tunable by the agent): optimality
`|obj − obj*|/(1+|obj*|) ≤ 1e-6` · primal feasibility `1e-7` infinity-norm · certificate
checks at `1e-9`, or exact where the exact pass runs.

---

## 6. The AutoResearch layer

Adapted from Karpathy's AutoResearch loop: an agent edits a restricted surface, a **fixed**
evaluator runs, a **protected** metric decides keep-or-revert. We adopt the methodology and
explicitly reject the premise that an agent writes the solver.

### 6.1 What it targets, and what it must never touch

**Netlib is the immutable correctness and regression baseline — never a tuning target.**
Tuning happens on the **industrial families** (§5.3), on a train split, with every reported
number coming from a held-out split.

The output is per-domain configuration profiles:

```
sankhya solve blend.mps --profile refinery
sankhya solve scm.mps   --profile supply-chain
```

**The claim, stated carefully:** *we explicitly tune and validate solver configuration
profiles on Indian industrial benchmark families, and report held-out measurements.* Nothing
about what other vendors do or do not ship — we have not surveyed them and cannot defend a
universal claim.

### 6.2 The loop

```
  agent proposes a change
        |
        v
  clean-room incremental rebuild, pinned compiler flags   -> fails? revert
        |
        v
  CORRECTNESS GATE  (hard precondition, NOT a scored term)
     LP  : primal feas + dual feas + complementary slackness  (certificate)
           + agreement with reference optimum where one exists
     MILP: integrality + feasibility + incumbent objective
           + claimed bound/gap cross-checked against reference
     round-trip: presolve/postsolve reconstructs the original solution
        |
     any failure -> REVERT immediately, regardless of speed
        |
        v
  PERFORMANCE (median of 3, pinned cores, randomized order)
        |
        v
  improvement beyond the stated noise floor? -> KEEP, commit, new incumbent
  otherwise                                  -> REVERT, log the negative result
```

Correctness is a **gate**, never a term in the objective. A candidate 10× faster and wrong on
one instance scores nothing. This one decision removes the entire "get fast by being wrong"
hack class — and it is the sentence to preserve if everything else in this document changes.

### 6.3 The mutation surface

**Tier A — parameters (committed).** One file, `config/tuning.json`: LU Markowitz threshold
and pivot tolerance, refactorization frequency, Harris pivot tolerance, bound-flip threshold,
Devex reset frequency, partial-pricing list size, cost-perturbation magnitude, stall threshold
before Bland fallback, presolve rule flags/order/pass limit, scaling choice and iterations,
pseudocost initialization, reliability threshold, plunging depth, best-bound vs best-estimate
weight, diving frequency, PDHG restart criterion and primal-weight update rate.

Note these are *algorithmic strategy* knobs. **Feasibility and optimality tolerances live in
the protected manifest, not here** — otherwise the agent could tune the definition of correct.

**Tier B — frozen-signature strategy bodies (stretch).** The agent may rewrite the body, never
the interface, of `chooseEnteringVariable`, `chooseBranchingVariable`, `selectNextNode`,
`divingHeuristic`. Karpathy-faithful (code, not just numbers) and safe because the verifier
catches wrongness regardless. **Tier B merges additionally require human diff review.**

### 6.4 Protection — broader than v1 had it

v1 hashed source files. That is insufficient: the agent could influence measurement through
generated files, build configuration, environment, or caches without touching a protected
source file. The manifest therefore covers:

- evaluator and verifier code, and the manifest checker itself
- benchmark instance files and reference solutions/optima
- **build scripts, CMake configuration, and pinned compiler/optimization flags**
- tolerance definitions
- train/held-out split definitions, fixed before the loop starts
- **captured environment**: compiler version, CUDA version, thread pinning, relevant env vars

Enforcement: agent write-scope limited to `src/strategies/` and `config/tuning.json`;
`bench/PROTECTED.sha256` verified before every evaluation, **eval refuses to run on any
mismatch**; clean-room build directory per candidate; fresh process, cleared caches,
randomized order, timeout and memory caps.

### 6.5 Anti-gaming

| Attack | Defence |
|---|---|
| Loosen internal tolerances to declare optimality early | External verifier recomputes from the original MPS at protected tolerances; tolerances are not in the mutation surface |
| Return a fast wrong answer | Certificate check — dual feasibility and complementary slackness, not just primal feasibility |
| Special-case on instance name, size, or hash | Held-out split + mandatory human diff review for Tier B |
| Influence timing via build flags or environment | Flags and environment are in the protected manifest; clean-room build per candidate |
| Cache results between runs | Fresh process, cleared working dir, randomized order |
| Claim MILP optimality without proving the bound | Bound cross-checked against reference; reported as *claimed*, never as certified |
| Win on timing noise | Median of 3, pinned cores, improvement must exceed a stated noise floor |

"We assumed our own agent would cheat and designed against it" reads as engineering maturity,
which is rarer than a good benchmark number.

### 6.6 Loop economics — stated as arithmetic, not as a promise

Incremental rebuild ~10–25 s; fast proxy eval on a curated train split ~60–90 s; so roughly
**2 min per candidate**. An 18-hour background window has a hard ceiling of ~540 candidates,
so 400–500 is the optimistic end after crashes, rebuild stalls and timeouts, and **150 is what
we commit to**.

**We do not claim this "meaningfully searches" the space.** Dimension count says little about
search difficulty — parameter interactions, conditioning and discrete/continuous ranges matter
far more. We report candidates evaluated, the best-so-far trajectory, and the held-out delta.
Nothing about coverage.

**Resource conflict:** the loop and the benchmark sweep both want CPU, and contention corrupts
the timings being measured. The loop runs pinned to 6 cores with the solver single-threaded
(less noisy, and parameters transfer); it is **paused entirely** for the final sweep.

### 6.7 Expected outcome — a goal, not a known fact

A **1.2–2× domain-specific improvement on the held-out split would be a strong outcome. We
report whatever the held-out measurements actually show**, including no improvement. A
negative result with 150+ logged candidates and a protected, reproducible harness is still a
real methodology result — and reporting it honestly is worth more than a number we cannot
defend.

---

## 7. Schedule

### H+0 → H+2 · Foundation
Toolchain install (VS Build Tools C++ workload → CUDA 12.x → CMake/Ninja) — **~13 GB on C:,
needs approval**. Repo skeleton, build, protected-manifest scaffolding, `SOVEREIGNTY.md`.
*Team: start every dataset download now — multi-GB and slow.*

### H+2 → H+8 · Model layer + LU
Sparse CSC/CSR, MPS fixed+free reader, QPS reader, CLI, writers. Scaling. **Sparse LU with
Markowitz ordering and threshold partial pivoting**, tested against a dense reference LU on
random and pathological matrices *before any simplex code exists*. Non-negotiable ordering.
*Team: case studies 1–2; Netlib manifest with checksums, reference optima and exclusions.*

### H+8 → H+16 · LP core
Primal revised simplex → dual simplex. PFI update and refactorization policy, Harris two-pass
ratio test, bound flipping, Devex pricing, cost perturbation, Bland fallback.
**Gate H+16: ≥ 60/98 Netlib. If missed, freeze features and debug.**
*Team: harness + first reference runs against HiGHS.*

### H+16 → H+20 · Presolve + control solver
Full presolve/postsolve with round-trip tests. Naive textbook simplex control (Dantzig, no
scaling, no Harris) + instrumentation.
**Gate H+20: ≥ 85/98 Netlib.**

### H+20 · Launch AutoResearch
Freeze `PROTECTED.sha256` and the train/held-out split. Loop starts on 6 pinned cores with
the float verifier, LP parameters only. Runs in background from here.

### H+20 → H+24 · Exact certificate verifier
Primal-dual certificate emission; float verifier; **exact rational pass**. Python, protected,
teammate-parallelizable.
*Team: case studies 3–4.*

### H+24 → H+29 · MILP (plain B&B, no cuts)
Dual-simplex warm start, propagation, pseudocost branching, best-bound + plunging, rounding
and diving. *Background: widen tuning surface to branching and node-selection parameters.*

### H+29 → H+34 · GPU
CUDA CSR SpMV (A, Aᵀ) + fused vector kernels, correctness-checked against CPU. PDHG on CPU
(OpenMP) first, then GPU. Measure 1-thread vs 16-thread vs GPU.

### H+34 → H+37 · QP prototype
PDHG proximal extension for convex QP. QPLIB convex subset.

### H+37 → H+42 · Freeze and sweep
**Pause the loop**, freeze per-domain profiles. Full sweep on the **held-out split**: Netlib
× 3 solvers, MIPLIB subset, QPLIB, case studies at 2 scales, default vs tuned profile.
Exact-certificate pass. Profiles, robustness charts, dashboard.

### H+42 → H+47 · Demo build
Live-demo script, seven moments rehearsed end to end, deck, README, API docs, roadmap.
**Everything runs offline from a local copy.** Assume venue Wi-Fi fails.

### H+47 → H+48 · Buffer
Untouched. It will get used.

---

## 8. Team split (no GPU required for anything below)

| Who | Owns |
|---|---|
| **Me + this machine** | C++ core, CUDA, all numerics, build system |
| **Teammate A** | Case studies 1–2 (refinery blending, crude pooling) → MPS |
| **Teammate B** | Case studies 3–4 (supply chain, transportation) + MPS validation |
| **Teammate C** | Harness, Netlib manifest with checksums and exclusions, reference runs, results DB |
| **Teammate C (from H+20)** | AutoResearch babysitting: splits, protected manifest, trajectory plots, rejected-candidate log |
| **Teammate D** | **Exact rational verifier** (highest-value teammate task), plots, dashboard, deck, docs |

---

## 9. The demo — seven moments

1. **"It is ours."** Linker inputs and `dumpbin /dependents` — no third-party optimization or numerical library. Presented as evidence for an open source tree, not as proof-by-command.

2. **"It is correct — and here is the proof."** The exact rational certificate check running live on a Netlib instance: primal feasibility, dual feasibility, complementary slackness, in exact arithmetic over ℚ. **No solver is trusted, no reference value is trusted, no tolerance is involved.** This is the strongest thing we have and it leads.

3. **"It is robust."** Naive textbook simplex cycles or returns a wrong objective on a degenerate ill-conditioned instance; ours converges. Per-iteration condition-estimate and residual traces show *why*. Then the fault-injection sweep, 10²→10¹⁰, charting where each solver breaks.

4. **"It scales."** Transportation LP at multi-million nonzeros: CPU 1-thread vs 16-thread vs GPU PDHG, measured. Stated with the caveat that PDHG reaches ~1e-4 without crossover, making it a bound-and-warm-start engine, not a simplex replacement.

5. **"It solves Indian industrial problems."** Four case studies end to end: model → MPS → SANKHYA → interpreted result. Default config vs tuned profile, on held-out instances.

6. **"It improves itself — safely."** AutoResearch trajectory: candidates proposed, rejected on the correctness gate, best-so-far curve, held-out delta. Live tamper demo — edit a protected tolerance, watch the eval refuse to run. Close on the anti-gaming table.

7. **"We know exactly where we stand."** Performance profile vs HiGHS, our honest position on it, **including that our plain B&B is weak on MIPLIB and why we chose that over risking invalid cuts**. Then the roadmap with effort estimates.

---

## 10. Targets

Measurement targets, not promises. Missed numbers get reported as measured.

| Metric | Commit | Stretch |
|---|---|---|
| Netlib LP solved to 1e-6 *(against the published manifest)* | **≥ 85 / 98** | 95 / 98 |
| Netlib instances passing the **exact rational** certificate | **≥ 40** | ≥ 70 |
| Netlib instances passing the float certificate | **≥ 85** | all solved |
| MIPLIB curated subset (25 easy, 300 s, plain B&B) | **≥ 8** proven optimal | ≥ 15 |
| QPLIB convex subset, prototype at 1e-4 | **≥ 6 / 12** | 12 / 12 |
| Largest LP solved by GPU PDHG | **≥ 2M nonzeros** | ≥ 20M |
| GPU vs our 16-thread CPU PDHG | **measure and report** (hope ≥ 3×) | ≥ 8× |
| Case studies end to end | **4 / 4** | 6 / 6 |
| AutoResearch candidates evaluated | **≥ 150** | ≥ 400 |
| Held-out profile improvement | **measure and report** (1.2–2× would be strong) | — |
| Correctness regressions shipped by the loop | **0** — gate is absolute | 0 |
| Third-party optimization/numerical libs in solve path | **0** | 0 |

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Over-scoping (v1's failure)** | **Was high** | v2 cut QP-IPM and all cutting planes. If H+16 slips, cut dual simplex next, then MILP entirely. LP + certificate + GPU + AutoResearch alone is a complete, credible submission. |
| LU subtly wrong → everything downstream is garbage | High | Built first, tested against dense reference on pathological matrices before any simplex exists |
| Simplex cycles on degenerate instances | High | Harris + bound flipping + perturbation + Bland fallback after N stalls |
| Exact rational verifier too slow | Med | Size budget per instance, hard timeout; instances that exceed it get the float certificate and are reported separately |
| AutoResearch overfits the tuning split | High | Held-out split fixed before the loop starts and in the protected manifest; all reported numbers from held-out |
| Agent finds an unanticipated reward hack | Med | Correctness as a hard gate kills the class; Tier B needs human diff review |
| Tuning contends with benchmark timing | Med | Loop pinned to 6 cores, single-threaded solver, paused for the final sweep |
| Toolchain install fails or overruns | Med | Fallback A: MinGW-w64, no CUDA. Fallback B: CuPy `RawKernel` — own kernels, no toolkit needed. **Decide by H+2.** |
| MIPLIB results look weak | **Expected** | We chose plain B&B deliberately over risking invalid cuts. Say so — it is a defensible engineering decision, not a shortfall. |
| Venue has no internet | Med | Everything vendored and rehearsed offline from H+42 |

---

## 12. What ships

```
The-solver/
├─ src/           C++17 core (model, presolve, LU, simplex, B&B, PDHG)
├─ cuda/          hand-written CSR SpMV + fused vector kernels
├─ include/       sankhya.h (C API)
├─ python/        ctypes binding
├─ apps/          CLI
├─ verify/        certificate verifier — float and exact rational  [PROTECTED]
├─ tests/         unit (LU, ratio test, presolve/postsolve round-trip) + regression
├─ bench/         harness, NETLIB_MANIFEST.txt, PROTECTED.sha256, profiles, dashboard
├─ cases/         4 committed + 2 stretch industrial generators
├─ autoresearch/  loop, tuning.json, trajectory logs, discovered profiles
├─ docs/          SOVEREIGNTY.md, ALGORITHMS.md, VERIFICATION.md, API.md, ROADMAP.md, RESULTS.md
└─ demo/          offline demo script + deck
```

---

## 13. Roadmap

- **0–3 mo:** cutting planes done safely (Gomory MIR with numerical safeguards and validity checks, knapsack cover, flow cover) · dual steepest edge · Forrest–Tomlin · Mehrotra QP-IPM with LDLᵀ · PDHG→simplex crossover
- **3–9 mo:** parallel B&B tree · conflict analysis · symmetry handling · RINS/RENS · MIQP via outer approximation
- **9–18 mo:** NLP (interior point with exact Hessians) · MINLP · spatial B&B for true bilinear pooling
- **Throughout:** an Indian industrial instance library with published reference solutions — plausibly worth more in the long run than the solver itself
