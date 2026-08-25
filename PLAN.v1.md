# SANKHYA — Sovereign Optimization Solver Core
## 48-Hour Hackathon Execution Plan

*(Name is a placeholder — सांख्य, "analytical reckoning". Change freely.)*

---

## 0. The honest framing (read this first)

We are **not** going to beat Gurobi in 48 hours. Nobody has ever done that. CPLEX is
~38 years of engineering; HiGHS is ~10 years of a funded research group. Any team that
stands up and claims otherwise will be dismantled by the first judge who has actually
used a solver.

So we win on a different axis. Our claim is precisely this:

> **A working, from-scratch, dependency-free optimization core that is *provably* ours,
> demonstrably numerically robust on problems that break naive implementations, and
> already scaling to industrial sizes via GPU — delivered with an honest, measured
> benchmark against HiGHS rather than a marketing number.**

That is a claim we can defend line-by-line, and it is exactly what the problem statement
asks for: a "transparent, extensible and sovereign **foundation**". The word is
*foundation*, not *replacement*.

**Three things we will never say:** "faster than Gurobi", "production ready", and
"solves problems with millions of variables" (we will say: *this specific class* of LP at
*this measured size*).

---

## 1. Hard constraints we are planning around

| Constraint | Consequence for the plan |
|---|---|
| 48 h wall clock | Ruthless scope cuts. Every module has a documented simpler fallback. |
| No compiler installed | ~90 min of Hour 0 is toolchain install. Non-negotiable, budgeted. |
| 4 GB VRAM (RTX 3050 laptop) | GPU work must be **sparse and matrix-free**. Rules out GPU dense factorization. Rules *in* first-order methods — which is the right choice anyway. |
| Teammates have no GPU | GPU is a single-machine workstream (mine). Teammates get CPU-only + Python work. |
| 33 GB free disk | Toolchain on C:, all benchmark data on R:. Curated MIPLIB subset, not the full set. |
| 16 threads (i7-12650H) | A real multi-core story is available and should be used. |
| 16 GB RAM | Caps in-memory instance size; instrument peak RSS in the harness. |

---

## 2. Architecture

```
                 +------------------------------------------------+
                 |  CLI   sankhya solve model.mps --gap 1e-4       |
                 |  C API sankhya.h  ->  Python ctypes shim        |
                 +----------------------+-------------------------+
                                        |
                 +----------------------v-------------------------+
                 |  MODEL LAYER                                    |
                 |  MPS (fixed+free) / QPS reader     [from scratch]|
                 |  min c'x + 0.5 x'Qx,  L <= Ax <= U,  l <= x <= u |
                 |  integrality flags, CSC/CSR sparse storage       |
                 +----------------------+-------------------------+
                                        |
                 +----------------------v-------------------------+
                 |  PRESOLVE + SCALING       (+ postsolve stack)   |
                 +----------------------+-------------------------+
                                        |
        +-------------------------------+--------------------------------+
        |                               |                                |
+-------v---------+          +----------v---------+          +-----------v--------+
| SIMPLEX ENGINE  |          |  IPM ENGINE        |          |  GPU PDHG ENGINE   |
| revised, bounded|          |  Mehrotra pred-corr|          |  (PDLP-style)      |
| Markowitz LU    |          |  regularized LDL^T |          |  matrix-free       |
| PFI update      |          |  -> LP and QP      |          |  own CUDA SpMV     |
| Harris ratio    |          |                    |          |  -> huge sparse LP |
| Devex pricing   |          +--------------------+          +--------------------+
+-------+---------+
        |  warm-started dual simplex
+-------v-----------------------------------------------------------------+
|  MILP: branch and cut                                                    |
|  pseudocost / reliability branching  ·  best-bound + plunging            |
|  cuts: Gomory MIR, knapsack cover  ·  propagation and bound tightening   |
|  heuristics: rounding, diving, feasibility pump                          |
|  OpenMP: parallel root cuts, parallel dives, concurrent-LP race          |
+--------------------------------------------------------------------------+
```

**Language:** C++17 core (MSVC), CUDA C++ for GPU kernels, OpenMP for CPU parallelism.
Python **only** for: benchmark harness, case-study model generators, plots, dashboard.
No Python in the solve path.

**Why C++ and not Python/NumPy:** the entire value proposition is "numerically robust
engine that scales". A Python core cannot be demonstrated at industrial scale, and the
90 minutes of toolchain install buys a ~30–50x constant factor. This is the single
highest-leverage decision in the plan.

---

## 3. Scope — in, out, stretch

### IN (committed)

- **Model / IO:** MPS fixed + free, QPS quadratic extension, solution and log writers, CLI, C API + Python binding
- **Presolve:** empty/singleton rows and columns, forcing and redundant constraints, bound tightening, duplicate row/column removal, fixed-variable removal, coefficient tightening — **with full postsolve**
- **Scaling:** geometric mean + equilibration (Curtis–Reid if time permits)
- **LP:** bounded-variable revised simplex, primal **and** dual; Markowitz LU with threshold partial pivoting; product-form update with periodic refactorization; **Harris two-pass ratio test with bound flipping**; Devex pricing; Bland's rule anti-cycling fallback
- **QP:** convex QP via primal-dual interior point, own regularized LDL^T on the quasi-definite KKT system (the same code path also gives us an LP IPM for free)
- **MILP:** branch and cut — reliability/pseudocost branching, best-bound + plunging node selection, Gomory MIR and knapsack cover cuts at the root and shallow nodes, domain propagation, rounding/diving/feasibility-pump heuristics
- **Parallelism (16 threads):** concurrent-LP race (primal ‖ dual ‖ PDHG on the root relaxation, first to finish wins), parallel root cut separation, parallel diving heuristics
- **GPU:** restarted-averaged PDHG for LP with adaptive step size and primal-weight balancing; **hand-written CUDA CSR SpMV** (A and A^T) plus fused vector kernels. No cuSPARSE, no cuBLAS.
- **Benchmarking:** full Netlib LP, curated MIPLIB 2017 subset, QPLIB subset, 6 industrial case studies, Dolan–Moré performance profiles, robustness instrumentation
- **The naive control solver:** a deliberately textbook simplex (Dantzig rule, no scaling, no Harris test) used *only* to demonstrate what our robustness machinery buys. ~1 h to write, and it produces the most persuasive slide in the deck.

### OUT (explicitly — and we put this on a slide)

Dual steepest-edge pricing · Forrest–Tomlin update (PFI + frequent refactorization instead) ·
parallel B&B tree search · flow-cover / clique / lift-and-project cuts · RINS / RENS / local
branching · conflict analysis · symmetry detection · MIQP / NLP / MINLP · crossover from
PDHG to a basis · distributed multi-node.

Every one of these gets a line in the roadmap slide with an effort estimate. Judges read
"knows exactly what is missing and why" as competence, not as weakness.

### STRETCH (only if we are ahead at H+30)

Forrest–Tomlin update · dual steepest edge · flow-cover cuts · PDHG→simplex crossover ·
GPU batched diving.

---

## 4. Sovereignty compliance — this is a graded criterion, treat it as one

The statement says: *"shall not be built upon any existing open source solver library."*
We enforce this mechanically, not by promise.

**Allowed in the shipped binary:** our source, the C++17 standard library, OpenMP, the CUDA
runtime and nvcc (a compiler/runtime, not a solver).

**Forbidden in the shipped binary:** COIN-OR (CBC / CLP / Cgl / Osi), HiGHS, GLPK, SCIP,
SoPlex, lp_solve, Gurobi / CPLEX / Xpress, `scipy.optimize`, cuSPARSE / cuSOLVER / cuBLAS,
Eigen, SuiteSparse, any BLAS or LAPACK. We write our own LU, LDL^T, SpMV and every vector
kernel.

**Allowed as *external reference processes only*** — invoked as separate executables by the
benchmark harness, never linked, never in our address space: HiGHS, CBC, SCIP, and Gurobi's
size-limited free license. The problem statement *requires* comparison against an
established solver; this is how we do that without contaminating the core.

**Deliverable:** `docs/SOVEREIGNTY.md` plus a CI script that dumps the linker input list and
runs `dumpbin /dependents` on the built binary, failing on any forbidden name. We run this
live in the demo. It takes 20 seconds and it settles the question permanently.

---

## 5. Benchmark and experimentation setup

### Datasets (all staged on `R:\` to protect C: space)

| Set | What | Approx size | Why |
|---|---|---|---|
| **Netlib LP** | all 98 classic LPs + published optima | ~50 MB | The correctness gate. Small, brutally degenerate and ill-conditioned. This is where naive simplex dies. |
| **Netlib Kennington** + large extras | large sparse LPs | ~200 MB | Feeds the GPU PDHG story. |
| **MIPLIB 2017 — curated subset** | ~40 instances HiGHS solves in <60 s, plus ~8 known-hard for gap-only reporting | ~2 GB | MILP credibility. The full 240-instance set is neither downloadable nor solvable in our budget, and we say so plainly. |
| **QPLIB — convex continuous subset** | ~10–15 instances | ~100 MB | QP credibility. |
| **Mittelmann LP set** — 5–8 largest feasible | very large sparse LP | ~2 GB | The GPU headline number. |
| **Industrial case studies** | 6 generators, ours, parameterized | — | The India story. See below. |

### The 6 industrial case studies

Python generators emitting MPS, each with a `--scale` knob so the *same* model produces a
5k-variable version for correctness and a 5M-nonzero version for the scaling demo.

1. **Refinery product blending** (LP, then MILP with min-run / mode-switch binaries) — gasoline and diesel pools, property blending on octane, sulfur, RVP, viscosity
2. **Crude blending / pooling** — the classic Haverly pooling problem; bilinear terms handled by a piecewise-linear MILP relaxation (McCormick envelopes + SOS2)
3. **Multi-period production planning** — capacitated lot-sizing with setups, using a *deliberately weak big-M formulation*, which is exactly the "weak LP relaxation" robustness case the statement names
4. **Unit commitment / power dispatch** — thermal units, minimum up/down times, ramp limits, spinning reserve. Highly degenerate MILP.
5. **Multi-echelon supply chain** — min-cost network flow with capacities and fixed-charge arcs, sized on an Indian plant → CFA → distributor geography
6. **Rail / road transportation** — a large transportation LP, degenerate by construction (many cost ties), scaling to millions of nonzeros for the GPU demo

### Harness (`bench/run.py`)

- Runs `{sankhya, highs, cbc}` × `{instance}` in subprocesses with a hard wall-clock timeout, a memory cap, and 3 repeats for timing
- Records: status, objective, primal/dual infeasibility, relative gap, iterations, nodes, factorizations, wall time, peak RSS
- Verifies our objective against published optima (Netlib) or against HiGHS (everything else) to a stated tolerance
- Emits a SQLite results DB, then **Dolan–Moré performance profiles**, per-instance tables, and a static HTML dashboard
- **`--fault-inject` mode** for the robustness story: rescale constraint rows by 10^k to drive up the condition number and chart where each solver breaks

### Metrics, and the tolerances we hold ourselves to

- Optimality: `|obj − obj*| / (1 + |obj*|) <= 1e-6`
- Primal feasibility: `max(0, Ax−U, L−Ax)` in infinity norm `<= 1e-7`
- Dual feasibility / complementarity (IPM): `<= 1e-8`
- MILP: proven gap at timeout, plus time-to-first-incumbent and the gap-closure curve
- Robustness: basis condition estimate and `‖A_B x_B − b‖_inf` traced **per iteration**

---

## 6. The AutoResearch layer — autonomous algorithmic experimentation

Adapted from Karpathy's AutoResearch loop: an agent edits code, a **fixed** experiment runs,
a **protected** metric decides keep-or-reject. We adopt the *methodology*. We explicitly
reject the premise that an agent writes the solver.

### 6.1 Why this belongs in *this* project specifically

The problem statement's sharpest criticism of existing open-source solvers is that they
"have not been developed, validated or **optimized specifically for Indian industrial use
cases**." AutoResearch is the direct answer to that sentence.

So we do **not** point the loop at Netlib chasing a generic speedup — that is a small,
boring number. We point it at our industrial case studies and let it discover
**domain-tuned profiles**:

```
sankhya solve blend.mps --profile refinery
sankhya solve uc.mps    --profile unit-commitment
sankhya solve scm.mps   --profile supply-chain
```

Each profile is a configuration *discovered automatically* by the loop on that domain's
instance family, then validated on a held-out split of the same family. No foreign solver
ships an India-industrial profile, because no foreign vendor has tuned on these problems.
This is the sharpest differentiator in the project, and it is defensible because every
number behind it is measured.

### 6.2 The loop

```
  agent proposes a change
        |
        v
  incremental rebuild (ninja)          <-- fails? reject, log, next
        |
        v
  CORRECTNESS GATE  (hard precondition, NOT part of the score)
    objective matches reference within 1e-6?
    primal feasibility within 1e-7?
    integrality within 1e-6?
    postsolve round-trip exact?
        |
     any failure -> REJECT immediately, no matter how fast it was
        |
        v
  PERFORMANCE MEASUREMENT (median of 3, pinned cores)
    wall time · simplex iterations · B&B nodes · factorizations · peak RSS
        |
        v
  significant improvement over incumbent?  -> KEEP, commit, new incumbent
  otherwise                                -> REJECT, log the negative result
```

Correctness is a **gate**, never a term in the objective. A candidate that is 10x faster
and wrong on one instance scores nothing at all. This single design choice removes the
entire class of "get fast by being wrong" reward hacks.

### 6.3 What the agent may modify — two tiers

**Tier A — strategy parameters (safe; committed for 48 h).**
One file, `config/tuning.json`, exposing roughly 35 knobs that real solver developers tune
by hand:

- LU: Markowitz threshold, pivot tolerance, refactorization frequency
- Ratio test: primal/dual feasibility tolerances, Harris pivot tolerance, bound-flip threshold
- Pricing: Devex reference-framework reset frequency, partial pricing candidate-list size
- Anti-degeneracy: cost perturbation magnitude, stall threshold before Bland fallback
- Presolve: rule on/off flags, rule ordering, pass limit, bound-tightening aggressiveness
- Scaling: algorithm choice, iteration count
- Branching: pseudocost initialization, reliability threshold, lookahead depth
- Node selection: plunging depth, best-bound vs best-estimate weight
- Cuts: rounds at root, violation/parallelism/rank thresholds, cuts-per-round cap
- Heuristics: diving frequency, feasibility-pump iteration cap, restart policy
- PDHG: restart criterion, primal-weight update rate, step-size adaptivity

This is a ~35-dimensional mixed discrete/continuous space. It is exactly what CPLEX and
Gurobi tune internally and ship as opaque defaults. Ours is transparent, versioned, and
reproducible in the repo.

**Tier B — strategy function bodies (higher variance; stretch).**
A small set of pluggable functions with **frozen signatures**, where the agent may rewrite
the body but not the interface:

```cpp
int  chooseEnteringVariable(const PricingCtx&);   // pricing rule
int  chooseBranchingVariable(const NodeCtx&);     // branching rule
int  selectNextNode(const TreeCtx&);              // node selection
bool divingHeuristic(const LPState&, Solution&);  // primal heuristic
```

This is Karpathy-faithful — the agent modifies *code*, not just numbers — and it stays safe
because the interface is frozen and the verifier catches wrongness regardless of what the
body does.

### 6.4 What the agent may never touch — and how that is *enforced*

Declaring files off-limits is not enforcement. Judges will probe this, so we make it
mechanical:

1. **Write-scope restriction.** The agent's filesystem write access is limited to
   `src/strategies/` and `config/tuning.json`. Nothing else is writable.
2. **SHA-256 manifest.** Every protected file — harness, verifier, tolerances, reference
   objectives, instance data, the manifest checker itself — is hashed at baseline into
   `bench/PROTECTED.sha256`. **The eval refuses to run if any hash has changed.** This is
   about 30 lines of code and it is a superb live demo: edit a tolerance, watch the loop
   halt and refuse to score.
3. **Independent external verification.** The verifier is a *separate process* that reads
   the solution file and recomputes feasibility and objective from the original untouched
   MPS, at fixed tolerances. It never trusts the solver's own status claim.
4. **Held-out split.** The agent tunes on a train split; every reported number comes from a
   held-out split it has never been evaluated against. Without this we will absolutely
   overfit 30 Netlib instances and the headline number will be meaningless.
5. **Fresh process per run**, cleared caches, timeout and memory caps, so a pathological
   candidate cannot hang or poison the loop.

### 6.5 Anti-gaming — we red-team our own agent

| Attack the agent could find | Defence |
|---|---|
| Loosen internal tolerances to declare optimality early | External verifier recomputes from the solution file at fixed tolerances |
| Special-case on instance name, dimensions, or hash | Held-out split + mandatory human diff review before any Tier-B merge |
| Cache solutions between runs | Fresh process, cleared working dir, randomized run order |
| Report "optimal" without proof | Objective checked against published optima independently of solver status |
| Tune the tolerance knobs themselves into meaninglessness | Feasibility tolerances are in the protected manifest, not in `tuning.json` |
| Win on timing noise | Median of 3, pinned cores, improvement must exceed a stated noise floor (>5% median, non-overlapping IQR) |

That table is a slide by itself. "We assumed our own agent would cheat and we designed
against it" reads as engineering maturity, which is rarer than a good benchmark number.

### 6.6 Loop economics under a 48-hour budget

The binding constraint is eval cost, not agent cleverness.

- Incremental Ninja rebuild of one strategy file: ~10–25 s
- Fast proxy eval — a curated ~30-instance train split chosen to run in ~60–90 s total
- **Realistic iteration: ~2 min. A 16-hour background window gives ~400–500 iterations.**

That is genuinely enough to search a 35-dimensional space meaningfully.

**Resource conflict — flagging this explicitly:** the tuning loop and the benchmark sweep
both want the CPU, and contention corrupts exactly the timings we are measuring. Mitigation:
the loop runs **pinned to 6 cores with the solver in single-thread mode** (single-thread
timing is far less noisy anyway, and the parameters transfer), leaving 10 threads for
foreground work. The final sweep runs alone, with the loop paused.

### 6.7 Honest expected gain

Autotuning in real solver development typically returns **1.2–2x on a targeted instance
family**. It will *not* close a 25x gap to HiGHS, and we will not imply that it might. What
we present is: a measured 1.3–1.6x on our industrial families, the search trajectory plot,
the rejected-candidate log, and the methodology. For a 48-hour build, an honest 1.4x plus a
reproducible autonomous experimentation harness is a far stronger result than a fabricated
10x.

### 6.8 Scheduling — this is *not* a week-3 afterthought

Your instinct to build the baseline first is correct, but the calendar was drawn for weeks
and we have hours. Compressed: the loop starts the moment the LP core and harness are
trustworthy (the H+19 Netlib gate) and then runs **continuously in the background for ~18
hours** while we build MILP, GPU and QP in the foreground. It costs us almost no foreground
time, which is exactly why it fits.

---

## 7. The 48-hour schedule

Times are from H+0. Team-parallel work in *italics*.

### H+0 → H+2 · Foundation
- Install VS Build Tools (Desktop C++ workload) → CUDA Toolkit 12.x → CMake + Ninja. **Needs your approval: ~13 GB on C:.**
- Repo skeleton, CMake build, CI script, `SOVEREIGNTY.md` and the dependency-audit script
- *Team: start every dataset download to R: now — they are slow, and starting them at H+20 is a classic hackathon death.*

### H+2 → H+7 · Model layer and LP scaffolding
- Sparse CSC/CSR, MPS fixed+free reader, QPS reader, CLI, solution and log writers
- Scaling (geometric + equilibration)
- Markowitz LU with threshold partial pivoting and triangular solves — **the single most important piece of numerics in the project**
- *Team: case-study generators 1–3*

### H+7 → H+15 · LP core
- Bounded-variable revised primal simplex, then dual simplex
- PFI update and refactorization policy, Harris two-pass ratio test, bound flipping, Devex pricing, Bland fallback
- **Gate at H+15: must solve ≥ 60 / 98 Netlib. If not, freeze feature work and debug.**
- *Team: benchmark harness plus the first Netlib run against HiGHS*

### H+15 → H+19 · Presolve and the naive control solver
- Full presolve/postsolve stack — the biggest single win per hour spent, on both Netlib and MIPLIB
- Naive textbook simplex control implementation, plus instrumentation hooks
- **Gate at H+19: ≥ 85 / 98 Netlib.**
- **Launch the AutoResearch loop** (§6) on 6 pinned cores, LP parameters only. From here it runs
  in the background continuously. Freeze `bench/PROTECTED.sha256` before it starts.

### H+19 → H+25 · MILP
- B&B with dual-simplex warm start, propagation, pseudocost/reliability branching, node selection
- Rounding, diving, feasibility pump
- Gomory MIR and knapsack cover cuts at the root
- *Team: case-study generators 4–6, MIPLIB subset reference runs*
- *Background: widen the AutoResearch search space to branching, node-selection and cut parameters as soon as B&C is stable*

### H+25 → H+30 · GPU
- CUDA CSR SpMV (A, A^T) and fused vector kernels, correctness-checked against the CPU path
- Restarted-averaged PDHG with adaptive step size and primal-weight balancing — CPU version first (OpenMP), then GPU
- Measure CPU 1-thread vs CPU 16-thread vs GPU on the large instances

### H+30 → H+34 · QP
- Mehrotra predictor-corrector IPM, regularized LDL^T on the KKT system
- QPLIB subset run; the same IPM gives us an LP mode for the concurrent-LP race

### H+34 → H+40 · Parallelism and the full benchmark sweep
- Concurrent-LP race, parallel root cuts, parallel dives
- **Pause the AutoResearch loop**, freeze the best configuration per domain, emit the `--profile` set
- Full sweep: Netlib × 3 solvers, MIPLIB subset, QPLIB, all 6 case studies at 2 scales, default config vs tuned profile, **all on the held-out split**
- Performance profiles, robustness charts, dashboard

### H+40 → H+46 · Demo build
- Live-demo script, all six showcase moments rehearsed end to end, deck, README, API docs, roadmap
- **Everything must run offline from a local copy.** Assume the venue Wi-Fi fails, because it will.

### H+46 → H+48 · Buffer
Untouched. It will get used.

---

## 8. Team split (accounting for no GPU on their machines)

| Who | Owns |
|---|---|
| **Me + this machine** | C++ core, CUDA, all numerics, build system |
| **Teammate A** | Case-study generators 1–3 (refinery blend, crude pooling, lot-sizing) — pure Python → MPS |
| **Teammate B** | Case-study generators 4–6 (unit commitment, supply chain, transportation) + MPS validation cross-checks |
| **Teammate C** | Benchmark harness, dataset curation, reference-solver runs (HiGHS / CBC — CPU only) , results DB |
| **Teammate D** | Dolan–Moré plots, HTML dashboard, deck, README, `SOVEREIGNTY.md`, demo rehearsal |
| **Teammate C (from H+19)** | Babysits the AutoResearch loop: train/held-out splits, protected-hash manifest, trajectory plots, rejected-candidate log. CPU-only, no GPU needed. |

Nothing on that list needs a GPU. The GPU is a single-node dependency, and it is mine.

---

## 9. What we actually show the judges — seven moments

1. **"It is ours."** Run the dependency-audit script live. `dumpbin /dependents` on the binary shows nothing but system DLLs and the CUDA runtime. Twenty seconds, and the sovereignty question is closed for the rest of the session.

2. **"It is correct."** The Netlib table: N/98 solved, objectives matched to published optima within 1e-6, side by side with HiGHS. A real number, not a claim.

3. **"It is robust — and here is the proof."** Take a degenerate, badly scaled instance. Run the naive textbook simplex: it cycles, stalls, or returns a wrong objective. Run SANKHYA: it converges. Then show the instrumented chart — basis condition estimate and residual per iteration, naive vs ours — so the audience *sees* the Harris ratio test and the refactorization policy doing their job. Then the `--fault-inject` sweep: scale the matrix by 10^2 … 10^10 and chart exactly where each solver breaks.
   **This is the most important slide in the deck.** It is the literal text of the problem statement.

4. **"It scales — and the GPU is earning its place."** The transportation / supply-chain LP at ~5M nonzeros. CPU 1-thread PDHG vs CPU 16-thread vs GPU PDHG, measured wall clock, on a 4 GB laptop card. We show the speedup *and* state plainly that PDHG reaches ~1e-4 relative accuracy without crossover, so it is a bound-and-warm-start engine rather than a replacement for simplex. That caveat is the point, not a weakness.

5. **"It solves Indian industrial problems."** All six case studies end to end: model → MPS → SANKHYA → solution → interpreted result (blend recipe, commitment schedule, dispatch plan). Not toy sizes.

6. **"And it improves itself — safely."** Show the AutoResearch trajectory: N candidates proposed, M rejected on the correctness gate, best-so-far curve, and the resulting per-domain profiles with their measured speedup on the **held-out** split. Then the live tamper demo — edit a tolerance in a protected file and watch the loop refuse to score. Close on the anti-gaming table. The claim is deliberately modest: *we adapted Karpathy's AutoResearch methodology to optimization, with correctness as a hard gate rather than a scored term, and it found a measured 1.Nx on Indian industrial instance families.*

7. **"We know exactly where we stand."** One slide: the performance profile against HiGHS on MIPLIB, with our honest position on it. Beside it, the gap analysis — *what* is missing (dual steepest edge, Forrest–Tomlin, parallel tree, cut families), *why* each costs us, and what closing it takes in engineer-months. Ending on a credible roadmap beats ending on an inflated number, every time.

---

## 10. Targets we are willing to be held to

| Metric | Commit | Stretch |
|---|---|---|
| Netlib LP solved to 1e-6 | **≥ 85 / 98** | 95 / 98 |
| Netlib time vs HiGHS (geometric mean) | within **25x** | within 8x |
| MIPLIB curated subset (40 easy/medium, 300 s cap) | **≥ 15** proven optimal | ≥ 25 |
| QPLIB convex subset | **≥ 8 / 12** | 12 / 12 |
| Largest LP solved by GPU PDHG | **≥ 2M nonzeros** | ≥ 20M |
| GPU PDHG vs our own 16-thread CPU PDHG | **≥ 3x** | ≥ 8x |
| Industrial case studies solved end to end | **6 / 6** at demo scale | 6/6 at 10x scale |
| Forbidden dependencies in the binary | **0** | 0 |
| AutoResearch candidates evaluated | **≥ 150** | ≥ 400 |
| Tuned-profile speedup on held-out industrial split | **≥ 1.25x** | ≥ 1.6x |
| Correctness regressions shipped by the loop | **0** (gate is absolute) | 0 |

If we miss a commit number, the slide reports the real number. We do not move goalposts.

---

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Toolchain install exceeds 2 h or fails | Med | Fallback A: MinGW-w64 via winget (no CUDA). Fallback B: CuPy `RawKernel`, which ships its own nvrtc, needs no toolkit, and still lets us write our own CUDA kernels. **Decide by H+2, no later.** |
| LU factorization subtly wrong, so everything downstream is garbage | **High** | Build it first (H+2). Test against a dense reference LU on random and pathological matrices *before any simplex code exists*. This ordering is non-negotiable. |
| Simplex cycles on degenerate Netlib instances | High | Harris ratio test + bound flipping + Bland's-rule fallback after N stalled iterations + cost perturbation. Budgeted, not improvised at 3 a.m. |
| MILP too slow to be interesting | Med | Presolve returns more per hour than B&B tuning. Cut cut-families before cutting presolve. |
| 4 GB VRAM overflow | Low | Matrix-free by design; 20M nonzeros is roughly 240 MB. Instrument and cap. |
| MIPLIB download too slow or too large | Med | Start downloads at H+0. Curated subset only. Mirror to R: early. |
| Venue has no internet | Med | Everything vendored and rehearsed offline from H+40. |
| Scope creep into MIQP / NLP | Med | Explicitly OUT. It is a slide, not code. |
| AutoResearch overfits the tuning split, so the headline number is fake | **High** | Held-out split is mandatory and every reported number comes from it. Train/test separation is fixed before the loop starts and is in the protected manifest. |
| Loop finds a reward hack we did not anticipate | Med | Correctness as a hard gate (not a scored term) kills the whole class. Tier-B code changes additionally require human diff review before merge. |
| Tuning loop contends with benchmark timing and corrupts both | Med | Loop pinned to 6 cores, solver single-threaded during tuning; loop paused for the final sweep. |
| Loop consumes hours and returns nothing | Med | It runs in the *background* and costs ~0 foreground time. If it returns nothing we report that honestly — a negative result with 400 logged candidates is still a real methodology result. |

---

## 12. What ships at H+48

```
The-solver/
├─ src/       C++17 core (model, presolve, LU, simplex, IPM, branch-and-cut, PDHG)
├─ cuda/      hand-written kernels: CSR SpMV, fused vector ops
├─ include/   sankhya.h  (C API)
├─ python/    ctypes binding
├─ apps/      CLI
├─ tests/     unit (LU, ratio test, presolve/postsolve round-trip) + regression
├─ bench/     harness, dataset scripts, Dolan-More profiles, dashboard
├─ cases/     6 industrial model generators
├─ autoresearch/  agent loop, tuning.json, PROTECTED.sha256, trajectory logs, profiles
├─ docs/      SOVEREIGNTY.md, ALGORITHMS.md, API.md, ROADMAP.md, RESULTS.md
└─ demo/      offline demo script + deck
```

---

## 13. Post-hackathon roadmap (the "foundation" argument)

- **0–3 months:** dual steepest edge, Forrest–Tomlin update, crossover from PDHG to a basis, flow-cover and clique cuts, RINS / RENS
- **3–9 months:** parallel B&B tree, conflict analysis, symmetry handling, MIQP via outer approximation
- **9–18 months:** NLP (interior point with exact Hessians), MINLP, spatial branch and bound for true bilinear pooling
- **Throughout:** an Indian industrial instance library — the strategic asset nobody else has, and arguably worth more in the long run than the solver itself
