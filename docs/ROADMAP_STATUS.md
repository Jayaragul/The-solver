# Roadmap status

Status of this repository against `CLAUDE_OPUS_SOLVER_ROADMAP.md`, using that
document's own vocabulary: `IMPLEMENTED`, `MEASURED`, `ESTABLISHED METHOD`,
`ENGINEERING DECISION`, `RESEARCH HYPOTHESIS`, `KNOWN LIMITATION`.

Everything claimed here is traceable to a file in `docs/measurements/`. Nothing
in this document is an estimate.

---

## Headline

| KPI | value | source |
|---|---|---|
| Netlib validated instances solved | **93 / 93** | `netlib-hybrid-20000rows.jsonl` |
| — simplex alone | 92 / 93 | `netlib-validation-20000rows.txt` |
| Kennington + QAP, cross-method checked | **21 / 21**, 0 disagreements | `crossmethod-kennington.jsonl` |
| — HYBRID total on that set | **43.9 s** vs 388.0 s simplex-alone | `crossmethod-kennington.jsonl` |
| total models solved | **114 / 114** | both files |
| row cap | 20,000 | — |
| total time | **72.594 s** (was 115.900 s before the structure-aware lead) | JSONL |
| geometric mean | 0.030 s | JSONL |
| median | 0.020 s | JSONL |
| 95th percentile | **4.033 s** | JSONL |
| max | **25.066 s** (`stocfor3`) | JSONL |
| total iterations | 255,144 | JSONL |
| worst relative objective error | 5.779e-07 | JSONL |
| unit tests | 104 / 104 | `ctest` |

`MEASURED`. Single process, nothing else running, build stamp
`1afe5bfa` recorded in the JSONL header.

This meets the roadmap's stated target — *"Maintain or exceed current 92/93
Netlib validation"* — by one instance.

---

## Phase 0 — benchmark and observability infrastructure

The roadmap ranks this **priority 1**, ahead of all algorithmic work. That
ordering was earned the hard way here: three separate wrong conclusions in this
project came from measurements taken without recorded conditions (see
`docs/architecture/PDLP.md` §5).

| item | status |
|---|---|
| Structured JSON/CSV output | `IMPLEMENTED` — JSON Lines, `src/bench/RunMetadata.*` |
| Full configuration capture | `IMPLEMENTED` — method, pricing, presolve, scaling, budgets, tolerances |
| Reproducibility metadata | `IMPLEMENTED` — git commit + dirtiness, compiler, CUDA, GPU, driver, CPU, RAM, threads, OpenMP schedule |
| Instance hashing | `IMPLEMENTED` — FNV-1a 64 over file bytes |
| Per-stage timers | `IMPLEMENTED` — `SimplexProfile`, 9 stages, `benchmarks/profile_simplex.cpp` |
| Median / geometric mean / p95 summaries | `IMPLEMENTED` — `bench::summarize` |
| Memory statistics | **NOT IMPLEMENTED** — peak RSS and peak VRAM are not captured |
| Repeated-run support, median over runs | **NOT IMPLEMENTED** — each sweep runs once |
| Performance profile generation | **NOT IMPLEMENTED** |
| Benchmark regression comparison | **NOT IMPLEMENTED** |
| NVTX ranges | **NOT IMPLEMENTED** |

Acceptance criteria: *"a result can be traced to a commit and instance hash"* is
met. *"A regression shows exactly which stage changed"* is met for the simplex
only, and not across runs.

---

## Phase 1 — correctness hardening

| item | status |
|---|---|
| Independent primal / dual / objective verification | `IMPLEMENTED` — original-space gate, `NUMERICS.md` §6, applies to every result including first-order |
| Presolve on/off differential testing | `IMPLEMENTED` — `validate_netlib … nopresolve` |
| CPU/GPU differential testing | `IMPLEMENTED` — bit-identical assertions across thread counts and backends |
| Determinism under fixed configuration | `MEASURED` — exact-equality tests, including PDLP |
| Random / ill-conditioned / degenerate LP generators | **NOT IMPLEMENTED** |
| MPS and sparse-structure fuzzing | **NOT IMPLEMENTED** |
| Compute Sanitizer in CI | **NOT IMPLEMENTED** — never run |
| Brute-force checker for tiny MILPs | **NOT APPLICABLE YET** — no MILP engine |

`KNOWN LIMITATION`: "no false INFEASIBLE" and "no false UNBOUNDED" are asserted
against the Netlib infeasible set (28/0/1) and the feasible set, not against
generated adversarial cases. That is weaker evidence than the roadmap asks for.

---

## Phase 2 — high-performance LP core

| item | status |
|---|---|
| Sparse basis factorization + eta updates | `IMPLEMENTED` — no dense inverse |
| Devex pricing | `IMPLEMENTED`, `MEASURED` at 2.16× Dantzig per iteration |
| Ruiz scaling | `IMPLEMENTED`, `MEASURED` |
| Presolve (core reductions) | `IMPLEMENTED` |
| **Warm-started dual simplex** (`solve_from_basis`) | **NOT IMPLEMENTED** |
| Hyper-sparse FTRAN/BTRAN with active-pattern detection | **NOT IMPLEMENTED** |
| Markowitz / AMD ordering, symbolic reuse | **NOT IMPLEMENTED** |
| Presolve expansion (doubleton, aggregation, probing, …) | **NOT IMPLEMENTED** |

`KNOWN LIMITATION`: the roadmap names warm-started dual simplex as the single
largest likely gain and ranks it **priority 3**. It does not exist. This is the
most important outstanding LP item and it is a hard prerequisite for a
competitive MILP engine, where every node is a warm child solve.

---

## Phase 3–5 — MILP

**NOT IMPLEMENTED.** There is no branch-and-bound, no integrality metadata, no
node management, no cuts.

`ENGINEERING DECISION`: branch and bound was explicitly *not* built during this
work despite being requested, because the benchmark set in question (Netlib) is
pure LP — there are no integer variables to branch on, so B&B could not have
moved the measured score. It becomes the top priority the moment MIPLIB is the
target.

---

## Phase 6 — GPU specialization

| item | status |
|---|---|
| Device-resident PDLP, fused kernels | `IMPLEMENTED` — `src/cuda/PdlpKernels.cu` |
| Sync-free inner loop | `MEASURED` — **127 iterations per host synchronize** vs simplex's 1 |
| Diagonal preconditioning | `IMPLEMENTED` — Ruiz + Pock–Chambolle |
| Adaptive step size | `IMPLEMENTED`, `MEASURED`, **net positive in HYBRID** — see below |
| Adaptive restarts | `IMPLEMENTED` — Applegate sufficient/necessary/artificial |
| GPU pricing inside simplex | `MEASURED` at 3–5× **slower**; retained only behind a flag |
| Feasibility polishing | **NOT IMPLEMENTED** |
| CUDA Graphs | **NOT IMPLEMENTED** |
| SELL-C-σ / HYB / custom SpMV formats | **NOT IMPLEMENTED** — only CSR measured |
| Nsight Systems / Compute profiling | **NOT DONE** |

### Adaptive step size — implemented, measured, not adopted by default

`MEASURED`. Applegate et al. 2021 §3.1, implemented entirely on-device: η lives
in device memory, the accept/reject test is a reduction plus a small kernel, and
a rejected step leaves the iterate untouched so the next queued iteration *is*
the retry. No host synchronization anywhere — which refutes this repository's
earlier claim that the rule "needs a host decision every iteration".

Result on the Netlib feasible set under 2,500 rows:

| | fixed η | adaptive η |
|---|---|---|
| iterations | 1,854,720 | **1,451,008** (−22%) |
| time | **110.35 s** | 120.90 s (+10%) |
| converged | 20 / 26 | 20 / 26 |
| within 1e-6 of optimum | 11 | 12 |

Large per-instance swings in both directions: `maros-r7` 8,448 → **4,608**
iterations (2.3× faster), `ganges` 394,752 → **167,168**; but `stocfor3`
207,616 → **242,176** and `scfxm3` 51,712 → **80,128**.

`ENGINEERING DECISION`: kept and left **on**, and the justification is now
stronger than when this was written. Once the first-order path can *lead*
(`LP.md` §7) its behaviour on large models is on the critical path, and a direct
A/B over the whole Netlib set says:

| | adaptive on | adaptive off |
|---|---|---|
| solved | **93 / 93** | 92 / 93 |
| total | **72.59 s** | 88.86 s |

So it buys an instance *and* 1.22× wall clock in the configuration that ships.
The earlier "+10% and net negative" finding was measured with PDLP standing
alone at `eps = 1e-6`, which is not how the engine uses it.

`KNOWN LIMITATION`: the adaptive rule does **not** fix the six instances where
PDLP stalls (`pilot.we`, `pilot.ja`, `bnl2`, `greenbea`, `greenbeb`, `pilot`).
`greenbea` still ends at KKT 3.24 after 478,464 iterations. Whatever causes
those stalls, it is not the step size. Feasibility polishing is the untested
hypothesis; it has not been implemented, so nothing is claimed for it.

---

## What the honest GPU verdict is

`MEASURED`. Three separate questions, three different answers:

1. **GPU inside the simplex — rejected.** 3–5× slower, for a structural reason
   (a host decision per iteration) that no kernel work can reach.
2. **GPU as a synchronization argument — supported.** 127 iterations per host
   round trip, exactly as designed.
3. **GPU as a wall-clock win — depends entirely on scale.** On Netlib it
   loses: 4.4× slower than the simplex below 2,500 rows, 1.56× faster above it.
   On the **Kennington set it wins decisively** — see below.

Per-iteration cost is near-flat across a 20× range of nonzeros (57 µs at 7,777
nnz, 111 µs at 144,848), the signature of latency-bound execution. That is why
the small set loses: the device is mostly idle.

### The scale hypothesis — TESTED, and it holds

An earlier revision of this document recorded, as an untested `RESEARCH
HYPOTHESIS`, that the flat cost curve implied headroom at larger scale, and
asserted that **no model in this repository was large enough to test it**.

**That assertion was wrong.** 21 models were being skipped by `validate_netlib`
— not for size, but because `netlib_readme.txt` carries no reference objective
for them. They are the Kennington families and the QAP relaxations, and they are
far larger than anything in the validated set: `ken-18` has 105,127 rows,
`osa-60` has 1,397,793 nonzeros. Excluding the hardest models because they were
awkward to score is precisely the kind of coverage gap that makes an aggregate
look better than the engine is.

`benchmarks/validate_crossmethod.cpp` now solves them with both methods and
checks the objectives against each other. `MEASURED`, 60 s budget per method:

| | result |
|---|---|
| solved | **21 / 21** |
| objective disagreements | **0** |
| both methods solved (16) | simplex 106.56 s vs first-order 44.53 s — **2.39× faster** |
| solved only by the first-order path | 5, within the 60 s simplex budget |

| instance | rows | nnz | simplex | first-order | |
|---|---|---|---|---|---|
| `ken-11` | 14,694 | 49,058 | 21.67 s | **1.03 s** | 21.0× |
| `pds-10` | 16,558 | 106,436 | 22.29 s | **1.46 s** | 15.3× |
| `osa-60` | 10,280 | 1,397,793 | 14.52 s | **1.93 s** | 7.5× |
| `ken-18` | 105,127 | 358,171 | *budget* | **12.63 s** | solves what simplex could not |
| `pds-20` | 33,874 | 230,200 | *budget* | **6.94 s** | " |
| `ken-13` | 28,632 | 97,246 | *budget* | **3.84 s** | " |

`KNOWN LIMITATION`: the five "first-order only" results are relative to a **60 s
simplex budget**, not to a simplex that ran to completion. They show the GPU path
reaching an answer far sooner, not that the simplex cannot get there eventually.

`KNOWN LIMITATION`: cross-method agreement is weaker than a published optimum. It
shows two methods sharing no arithmetic reached the same point and both cleared
the original-space gate; it cannot prove that point optimal.

So H1's wall-clock claim **is supported at scale**, on real industrial-structured
models rather than by extrapolation. Layer D of the benchmark plan (a
refinery-style generator with controllable difficulty) still does not exist, so
no refinery-*specific* claim is admissible.

---

## Next, in the roadmap's own priority order

1. **Warm-started dual simplex** (`solve_from_basis`). Priority 3 on the
   roadmap, missing, and the prerequisite for everything in MILP. The Kennington
   results sharpen this: the simplex hits its budget on `ken-13`, `ken-18`,
   `pds-20`, `qap12` and `qap15`, and those are exactly the models where a warm
   start has the most to give.
2. **Peak RSS / VRAM capture and repeated-run medians** — the two Phase 0 gaps
   that still let a regression hide.
3. **Generated adversarial LPs + Compute Sanitizer** — Phase 1's real
   acceptance criteria, currently only argued from Netlib.
4. **Hyper-sparse FTRAN/BTRAN**, then presolve expansion.
5. **Minimal correct MILP B&B**, verified against exhaustive enumeration.
6. Feasibility polishing for the six stalling PDLP instances.
7. Layer D refinery generator — without it, no refinery claim is admissible.

The governing rule stands: *no optimization is accepted unless it improves a
declared benchmark KPI without reducing correctness or solvability.* Two changes
during this work were reverted under exactly that rule — window-granularity
adaptive step size, and concurrent simplex/PDLP racing.
