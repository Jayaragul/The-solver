# CPU/GPU Division of Labor and PCIe Optimization

**Status:** PHASE 2 architecture. Every GPU placement decision below is justified against SOTA.md's Phase 1 findings and is explicitly marked as validated-by-literature or pending-empirical-validation — per prompt.md §2.2, no operation is moved to GPU automatically.

**Target hardware, measured (not assumed):** NVIDIA RTX 3050 Laptop, compute capability 8.9, 4.29 GB VRAM, 14 SMs, PCIe-connected (laptop MXM/soldered, shared with the OS desktop compositor — `nvidia-smi` shows 0.8–1.7 GB already resident from other GPU clients before this solver runs). This is a materially smaller and more contended GPU than every benchmark cited in SOTA.md §1.3b, and every claim below about expected GPU benefit is qualified accordingly.

---

## 1. Responsibility Matrix

| Responsibility | Residency | Rationale |
|---|---|---|
| Model parsing / representation | CPU | Control-flow-heavy, one-time, not arithmetic-intensity-bound |
| Presolve | CPU | Irregular row/column scans, fixed-point iteration (SYSTEM.md §2.3) |
| Scaling decision (Ruiz factors) | CPU | $O(\text{nnz})$, small bounded iteration count, doesn't amortize launch overhead |
| Structural analysis (block structure, symmetry candidates) | CPU | Graph/combinatorial, control-flow-heavy |
| Branch-and-bound control flow | CPU — **hard constraint** | prompt.md: GPU must never control the global tree |
| Node creation / selection | CPU | Sequential priority-queue logic |
| Branching (variable selection) | CPU | Strong/pseudo-cost/reliability branching is inherently sequential per-candidate evaluation |
| Cut management | CPU | Separation heuristics are control-flow-heavy (SOTA.md KS-5) |
| Incumbent management | CPU | Single global mutable state, low volume |
| Heuristic control | CPU | Control-flow-heavy |
| Numerical policy decisions | CPU | Decision logic, not bulk arithmetic |
| Memory management | CPU | Arena/pool ownership logic |
| Solver state management | CPU | `SolveContext`, all module state |
| **CSR SpMV** ($Ax$, $A^Ty$, repeated) | **GPU** (cuSPARSE) | Validated GPU candidate — see §2.1 below |
| Vector arithmetic (axpy, dot, norm) on SpMV-adjacent vectors | **GPU** (cuBLAS Level-1), *conditionally* | Only when already co-resident with a GPU SpMV result — see §2.2 |
| Numerical residual calculation ($Ax-b$, $A^Ty+s-c$) | **GPU**, reusing the SpMV primitive | Same kernel as the forward solve — a genuine reuse opportunity (§2.3) |
| Sparse factorization (LU/Cholesky for basis or normal equations) | **CPU in v1** | cuDSS is preview-status/non-deterministic, cusolverSp is partially host-bound (SOTA.md §1.3b) — not adopted yet |
| Pricing / pivoting kernels | **CPU** | Literature evidence against GPU simplex is strong (SOTA.md §1.3b) — not attempted in v1 |
| Batched decomposition subproblems | **Deferred** | Second-phase research per SOTA.md KS-9, contingent on subproblem homogeneity not yet characterized |
| First-order LP iterations (PDLP-style) | **Deferred** | SOTA.md KS-7 explicit non-v1 decision |

---

## 2. GPU Candidate Analysis

Per prompt.md §2.2, each candidate is evaluated on arithmetic intensity, memory bandwidth, parallelism, synchronization, branch divergence, occupancy, launch overhead, PCIe transfer cost, and reuse opportunity — not assumed beneficial.

### 2.1 CSR SpMV via cuSPARSE — the only unconditional v1 GPU placement

- **Arithmetic intensity:** for CSR, each nonzero contributes one multiply-add (2 FLOP) and requires reading one value (8 bytes, FP64) + one column index (4 bytes, `int32`) from the matrix, plus one read of $x_j$ (8 bytes, but reused across rows with locality) and one write of $y_i$. Roughly 2 FLOP per ~12–16 bytes moved — this is **memory-bandwidth-bound**, not compute-bound, on any modern GPU (roofline argument: FP64 peak FLOP/s on this device vastly exceeds what 2 FLOP/12 bytes can sustain against ~a few hundred GB/s of memory bandwidth).
- **Memory bandwidth:** this is the binding constraint. The RTX 3050 Laptop's memory bandwidth is a small fraction of a datacenter GPU's — the *literature's* reported SpMV speedups (SOTA.md §1.3b) were measured on hardware with far more bandwidth and far more SMs (14 here vs. often 80–132 on datacenter parts); a proportionally smaller speedup, or a larger crossover nnz threshold before GPU wins, is the expected (not yet measured) outcome — this is exactly what **H1** in SOTA.md §5 is designed to test.
- **Parallelism:** row-parallel decomposition is embarrassingly parallel across rows; cuSPARSE's internal algorithm selection (`CSR_ALG1`/`ALG2`) handles load balancing for irregular row-length distributions (a real concern for refinery-structured matrices, which are not uniformly sparse).
- **Synchronization:** none required *within* one SpMV call; synchronization is only needed at the point where the CPU next needs the result (an explicit `cudaStreamSynchronize` or event wait — never an implicit blocking call in the hot path per prompt.md §3.6).
- **Branch divergence:** none in the arithmetic itself (SpMV has no data-dependent control flow); this is precisely why SpMV is GPU-appropriate where simplex pivoting (data-dependent, divergent) is not.
- **GPU occupancy:** depends on row-length distribution and matrix size; for very small matrices (the common case inside deep B&B trees, where node-local relaxations can shrink via presolve/bound propagation), occupancy will be poor — this bounds where in the solve SpMV offload is actually worth invoking, and motivates §2.1's "conditionally" framing for smaller in-node work.
- **Launch overhead:** a fixed, non-trivial cost (typically single-digit microseconds) that must be amortized against the SpMV's own runtime; for small $n$/nnz this can dominate — a candidate case where the adaptive strategy (see `LP.md`) should keep the operation on CPU.
- **PCIe transfer cost:** the matrix itself is loaded once and stays device-resident for the life of a solve (§3 below) — it is **not** re-transferred per call. Only the dense vector $x$ (H2D) and result $y$ (D2H) cross PCIe per call, each $O(n)$ — negligible relative to the $O(\text{nnz})$ compute/bandwidth cost of the SpMV itself, provided $\text{nnz} \gg n$ (true for realistic sparse refinery models).
- **Reuse opportunity — the strongest argument for this candidate:** the *same* device-resident CSR matrix and the *same* cuSPARSE descriptor/workspace serve (a) the forward LP computation, and (b) the residual verification pass in Numerical Verification (`SYSTEM.md` §2.12), which needs exactly $Ax-b$. `cusparseSpMV_preprocess` (SOTA.md §1.3b) amortizes analysis cost across this repeated use.
- **Verdict:** ESTABLISHED as the primary v1 GPU placement, **empirical validation still required** (H1) before any throughput claim is made on this hardware.

### 2.2 Vector arithmetic (cuBLAS Level-1: axpy, dot, nrm2)

- Only justified when the vectors involved are *already* device-resident as a byproduct of an SpMV call (§2.1) — i.e., no additional PCIe transfer is introduced solely to run a vector op on GPU. Standalone vector ops on host-resident data are **not** moved to GPU: their arithmetic intensity is even lower than SpMV's, and the H2D/D2H round-trip for an $O(n)$ vector operation would typically cost more than a CPU-side pass over the same data. **IMPLEMENTATION DECISION:** GPU vector ops are opportunistic, chained after a GPU SpMV, never a standalone dispatch.

### 2.3 Numerical residual calculation

Identical computational shape to §2.1 ($Ax-b$ is literally an SpMV plus an axpy) — this is not a separate kernel design, it is a reuse of §2.1's primitive. Listed separately in the responsibility matrix only because prompt.md's architecture template calls it out explicitly (§2.5's numerical-reliability mandate makes residual computation a first-class, always-run operation, not an optional diagnostic).

### 2.4 Sparse factorization (deferred)

cuDSS is documented by NVIDIA as **preview**: API subject to change, non-deterministic results between runs (SOTA.md §1.3b). prompt.md §2.5 requires deterministic, reproducible numerics as a first-class property. These are in direct conflict for any v1 use — **IMPLEMENTATION DECISION: excluded from v1.** `cusolverSp`'s sparse LU/Cholesky path is additionally reported to be partially host-bound, undermining the case for GPU placement even setting determinism aside. Revisit when cuDSS reaches GA with a documented determinism guarantee.

### 2.5 Pricing kernels (implemented and measured); pivoting (still excluded)

SOTA.md §1.3b's literature review (Ploskas & Samaras and related work) found GPU-parallelized simplex has repeatedly underperformed sequential CPU simplex on sparse LPs, for structural reasons that apply regardless of implementation quality: irregular, data-dependent branching (warp divergence) and basis updates that are inherently sequential with fine-grained irregular memory access.

That conclusion is upheld for **pivoting** — the basis update, the ratio test, and all branch-and-bound control flow remain CPU-resident, and nothing here proposes changing that. It is a structural argument about a sequential dependence chain, and no measurement was needed to accept it.

**Pricing is a different operation and was treated as a separate question.** Unlike pivoting, pricing is one wide, uniform pass over every nonbasic column followed by an argmax — the shape that maps well to a GPU. prompt.md §3.2 explicitly lists "pricing-related kernels where appropriate" among the GPU candidates, and §3.8 forbids reporting performance that has not been measured. Declining to implement it on the strength of a literature result about a *different* operation would have been an assertion, not a finding. **IMPLEMENTATION DECISION: implemented, measured, and left non-default — §4 has the numbers that decided it.**

What was built (`src/cuda/PricingKernels.cu`, `src/cuda/GpuPricer.hpp`):

- **cuSPARSE still owns the SpMV.** prompt.md §3.5 forbids a hand-written SpMV, and $A^\top v$ is still `cusparseSpMV` with `CSR_ALG2`. The custom kernels are the fused reduced-cost/eligibility/argmax pass and the Devex weight update — neither of which any library exposes.
- **The argmax runs on the device.** This is the entire architectural argument for GPU pricing, and it is not about arithmetic. The naive placement (SpMV on the GPU, everything else on the host) ships $O(n)$ doubles back per iteration and then does the reduced-cost assembly and entering-variable scan on the host anyway — it moves the cheapest part of pricing and pays a full PCIe round trip for the privilege. Reducing on the device instead means **one 24-byte `PricingCandidate` crosses back per iteration, independent of $n$**.
- **Devex weights never leave the device.** The pivot row is $A^\top B^{-\top} e_r$ — the same $A^\top v$ product pricing already needs — so the weight update reuses the same cuSPARSE operator and the same fusion. Only the BTRAN result ($m$ doubles) goes up; nothing comes back.
- **Nothing that is read once is materialized.** Reduced costs and pivot-row entries each have exactly one consumer, so both are computed inside the kernel that consumes them. That removed two launches per iteration and, measured, ~13% of GPU pricing time — the right order of magnitude precisely because this path is launch-bound rather than bandwidth-bound (§4.1).
- **Determinism is preserved.** The argmax comparator is a total order on (score descending, index ascending), so the reduction returns the same winner for any block count or scheduling, and the lowest-index tie-break matches the CPU's ascending strict-`>` scan. `CSR_ALG2` keeps the SpMV itself reproducible (§2.1).

---

## 3. PCIe Optimization

### 3.1 Design principle

The constraint matrix is the largest, most expensive-to-transfer object in the system and is also the most reused (every LP solve at every B&B node touches it via SpMV). It is therefore made **persistently device-resident for the life of a solve** — transferred once (async, pinned staging buffer), never re-transferred, regardless of how many thousands of B&B nodes subsequently call SpMV against it (or a node-local sub-view of it, if node presolve shrinks the active row/column set — see `MILP.md`).

### 3.2 What crosses PCIe, and how often

| Data | Direction | Frequency | Size |
|---|---|---|---|
| CSR/CSC matrix (values, indices, offsets) | H2D | Once per solve | $O(\text{nnz})$ |
| Dense vector $x$ (SpMV input) | H2D | Once per SpMV call | $O(n)$ |
| Dense vector $y$ (SpMV result) | D2H | Once per SpMV call | $O(m)$ |
| Residual scalars (norms) | D2H | Once per residual check | $O(1)$ |

The one-time $O(\text{nnz})$ transfer is amortized across the full solve; the recurring per-call transfers are $O(n)$/$O(m)$/$O(1)$ — strictly smaller than the $O(\text{nnz})$ compute they support, provided the matrix is not pathologically sparse ($\text{nnz} \sim n$, a near-diagonal matrix) — a case where SpMV offload is not expected to pay off anyway per §2.1's occupancy/launch-overhead caveat.

### 3.3 Mechanisms

- **Pinned host buffers:** the staging buffers for the one-time matrix upload and the per-call vector transfers are pinned (`cudaHostAlloc`/`cudaMallocHost`), enabling true asynchronous DMA transfer rather than the implicit staged copy the driver performs for pageable memory. Pinning is applied *only* to buffers that actually cross PCIe repeatedly — the raw `RawModel` input (§`SYSTEM.md` 2.1) is read once and is not pinned, since pinning has its own allocation cost and pinning a one-time-use buffer would be pure overhead.
- **Asynchronous memcpy + streams:** all H2D/D2H transfers are issued via `cudaMemcpyAsync` on a dedicated stream, never the default (synchronizing) stream.
- **Double buffering:** relevant once the LP engine issues more than one SpMV per "round" (e.g., a future iterative-refinement or first-order path with several vector operations per iteration) — while the GPU computes on buffer A, the CPU prepares the next input into buffer B. For v1's usage pattern (SpMV invoked once per LP iteration, result consumed immediately by CPU-resident pivoting logic before the next SpMV can be issued), double buffering has **limited applicability**, since there is no independent next-input to prepare while the current SpMV runs — the CPU's next action *depends on* the current SpMV's result. This is stated explicitly rather than implemented speculatively, per prompt.md's "do not recommend generically" instruction for PCIe techniques.
- **Event-based synchronization:** the CPU waits on a `CudaEvent` recorded after the SpMV kernel, not a blanket `cudaDeviceSynchronize()` (prompt.md §3.6 explicitly prohibits unnecessary synchronize calls in the hot path).
- **Persistent device residency:** the matrix, and pre-allocated device buffers for $x$/$y$/residual scratch, are allocated once (during `SolveContext` initialization, per `SYSTEM.md` §3) and reused for the life of the solve — no repeated `cudaMalloc`/`cudaFree` inside the per-node solve loop, matching prompt.md §3.1's zero-allocation-during-solve requirement.

### 3.4 When zero-copy helps, and when it doesn't

Zero-copy (mapped pinned host memory, accessed directly by the GPU over PCIe per-transaction rather than via a bulk transfer) is beneficial only for data accessed **once, or very few times**, where the cost of a bulk transfer wouldn't be amortized anyway — e.g., reading a small one-shot configuration vector. It is **actively harmful** for this project's SpMV hot path: the matrix and vectors are accessed repeatedly (once per B&B node, thousands of times per solve), and zero-copy would serialize that access over PCIe on every touch instead of paying the transfer cost once and then operating on fast VRAM for every subsequent access. **IMPLEMENTATION DECISION: zero-copy is not used for any persistent solve-lifetime data structure in this architecture.** This directly answers prompt.md §2.4's instruction not to recommend zero-copy generically — it is evaluated per-object and rejected for every object that matters on the hot path.

### 3.5 Unified memory

Not used in v1. **Rationale (IMPLEMENTATION DECISION):** Unified Memory's page-migration heuristics trade explicit control for convenience, and this project's numerical-reliability mandate (prompt.md §2.5, deterministic/reproducible behavior) is easier to satisfy with explicit, auditable data placement than with a runtime-managed migration policy whose timing can vary run-to-run. On a 4.29 GB device shared with the OS compositor, predictable VRAM budgeting (`MEMORY.md` §1) also matters more than UM's convenience of not having to size buffers up front.

---

## 4. MEASURED: is GPU pricing actually faster? (H1 / H5)

`benchmarks/bench_pricing_backend.cpp`. Full `solve_lp` pipeline on both sides — same presolve, scaling, tolerances, ratio test, basis factorization and pricing rule; **the pricing backend is the only variable**. Netlib feasible set to 400 rows, 41 instances, Devex. Hardware: RTX 3050 Laptop (CC 8.9, 14 SMs, 4.29 GB), 16-thread host, CUDA 13.3 under WSL2. Three runs per configuration; iteration counts were identical across all six.

**Answer: no — not at any size this project can validate against.**

| CPU pricing | iterations | pricing time | per iteration | GPU is |
|---|---|---|---|---|
| single-threaded | 20,545 | 1.81 – 1.84 s | 88 – 90 µs | **3.15 – 3.24× slower** |
| 16 threads | 20,545 | 1.23 – 1.27 s | 60 – 62 µs | **4.64 – 4.83× slower** |
| *(GPU, unaffected by host threads)* | 21,323 | 5.76 – 5.96 s | 270 – 279 µs | — |

GPU pricing was faster on **0 of 41** instances. Objectives agreed on all 41; no status mismatches. Both rows are given because they answer different questions: the single-threaded row isolates *where pricing runs*, and the 16-thread row is the comparison against what actually ships.

### 4.1 Why — measured, not inferred

The benchmark probes this platform's kernel-launch cost directly, because the alternative is guessing at it:

| | |
|---|---|
| empty kernel, submission only | **8.74 µs / launch** |
| empty kernel, launch + synchronize | **26.72 µs / round trip** |

GPU pricing must complete one round trip per iteration — the host cannot choose a pivot without the winning column, so that synchronize is inherent to the algorithm, not an artifact of the implementation. **26.72 µs is therefore a hard floor under every GPU pricing iteration.** On most Netlib models the CPU's *entire* pricing pass costs less than that floor, so no kernel improvement of any kind could have won: the work being accelerated is smaller than the cost of asking for it.

This is §2.1's occupancy caveat arrived at from the other direction, and it is a property of the workload's *granularity* rather than of the kernels. It is also inflated by the platform — WSL2 paravirtualizes GPU submission, so these latencies exceed what a native Linux driver would show. That does not change the conclusion here; it changes where the crossover sits.

### 4.2 The trend is the useful part

The aggregate hides the shape. Per-instance, against **single-threaded** CPU pricing, the gap closes monotonically with problem size:

| instance | nnz | GPU pricing vs CPU pricing |
|---|---|---|
| `sc50b` | 118 | ~100× slower |
| `scagr7` | 420 | ~100× slower |
| `scfxm1` | 2,589 | ~33× slower |
| `grow15` | 5,620 | ~11× slower |
| `fit1d` | 13,404 | ~11× slower |
| `wood1p` | 70,215 | ~5× slower |
| `fit2d` | 129,018 | **~1.8× slower** |

Extrapolating a crossover from seven points would be exactly the unmeasured claim prompt.md forbids, so: **the crossover is not reached by any Netlib instance, and its location is unknown.** What the trend does establish is that the gap is closing with size and that the largest available instance is still short of it — a statement about the test set as much as about the hardware.

### 4.3 What this settles

- **H1 (hybrid CPU-GPU beats CPU-only on refinery-scale LP): NOT SUPPORTED at validated scale.** H1's own failure criterion — "no crossover within representative problem sizes on this hardware" — is met for every instance this project can validate against. H1 is not *refuted* for refinery-scale models, because no refinery-scale model was available to test. Unsupported and refuted are different claims and the difference matters.
- **H5 (selective acceleration beats GPU-ifying the solver): SUPPORTED, now on this project's own evidence rather than by citation.** The literature predicted GPU simplex would underperform; this engine's own pricing backend underperforms, for a reason (launch granularity) consistent with the structural explanation SOTA.md §1.3b gives.
- **`PricingBackend::CPU` remains the default** — which is what every validated result in this repository has always used. The GPU path is retained, correct, tested and deterministic: it is the measuring instrument that produced the finding above, and it is what a larger deployment target would be evaluated with.

### 4.4 A bug worth recording

Making the Devex update fully asynchronous introduced a defect the test suite initially missed. `devex_update` queues DMAs out of its pinned staging buffers and returns without synchronizing — that asynchrony is the point — and the *next* pricing call then overwrote those same buffers on the host while the copies could still be in flight. The only symptom was that GPU iteration counts wandered between runs and one instance intermittently failed verification. No crash, no CUDA error, and no wrong answer that a tolerance-based check would have caught.

Fixed by giving pricing and the Devex update **separate** pinned staging buffers, which makes the hazard impossible rather than unlikely: every Devex copy is followed by the next pricing call's stream synchronize before that buffer can be touched again. Pinned by `gpu_pricing_is_bit_reproducible_across_repeated_solves`, which asserts **exact** equality of iteration count and objective across repeated solves — a tolerance would have passed throughout the bug's lifetime. Verified after the fix: six consecutive benchmark runs, GPU iteration total identical (21,323) every time.

This is the concrete cost of prompt.md §3.4's warning about asynchronous transfers, recorded rather than quietly fixed because the failure signature — nondeterminism with no error — is the one worth recognizing again.

---

## 5. MEASURED: CPU multithreading (OpenMP)

prompt.md §3.2 permits OpenMP "where justified". `src/parallel/Parallel.hpp` states the policy; `benchmarks/bench_parallel.cpp` is the justification.

**Determinism first.** NUMERICS.md §1 requires bit-reproducible results, which rules out `reduction(+:)` over floating point and any scatter-add: both make the summation order a function of scheduling. Only loops where each *output* element is produced by one iteration in serial summation order are parallelized — CSR SpMV rows, per-column reduced costs, per-column pivot-row entries — all with `schedule(static)`. `CSCMatrix::multiply` is deliberately left serial: its scatter-add cannot make the same guarantee. Accepting reassociation on the CPU while rejecting it on the GPU (§2.1's `CSR_ALG2` choice) would have been incoherent.

**Size gating is the whole design.** Entering a parallel region costs a fork and a barrier. The simplex prices once per iteration, thousands of times per solve, on passes often only microseconds long — the same granularity problem §4.1 describes for the GPU. Measured, 16 threads, ungated:

| nnz | parallel speedup |
|---|---|
| 488 | 0.06× |
| 1,990 | 0.65× |
| 3,993 | 1.32× |
| 9,990 | 0.65× |
| 29,983 | **3.84×** |
| 159,975 | **4.85×** |
| 499,951 | **5.08×** |
| 2,399,927 | 3.55× |

Below ~2k nonzeros threading is a clear loss; between ~4k and ~10k the result is inconsistent from run to run; from ~30k it is a reliable 4–5×. `kParallelNnzThreshold = 20000` sits above the inconsistent band and below the reliable win — deliberately not at the first point a speedup appears, because an unreliable win on a pass executed thousands of times per solve is not worth taking. Speedup falls back to 3.55× at 2.4M nnz as the problem leaves cache and the loop becomes bandwidth-bound rather than compute-bound; that is expected, and is why the threshold is a floor rather than a window.

**End-to-end**, full Netlib validation (`validate_netlib`), 1 thread vs 16:

| | 1 thread | 16 threads | |
|---|---|---|---|
| `fit2d` (129,018 nnz) | 1.783 s | 1.012 s | **1.76×** |
| `wood1p` (70,215 nnz) | 0.038 s | 0.028 s | 1.36× |
| aggregate CPU pricing, 41-instance set | 1.81 – 1.84 s | 1.23 – 1.27 s | **~1.45×** |
| instances below the gate | — | unchanged | by design |

Objectives and **iteration counts are identical** at both thread counts on every instance: the thread count changes who does the arithmetic, never the arithmetic. 89/89 passed at both.

The honest scope: most Netlib models fall below the threshold and are untouched by threading, so the aggregate gain is concentrated in a few large instances. That is exactly what a size-gated policy is supposed to produce, and it scales with model size — a refinery-scale model would see more of it than this test set does.

**Building without OpenMP is a supported configuration**, not a degraded one: the pragmas compile out and every result is unchanged. `csr_multiply_is_bit_identical_across_thread_counts` and `lp_solve_is_bit_identical_across_thread_counts` assert exact equality across thread counts, so this is tested rather than asserted.

---

## 6. The GPU path that wins — sometimes: first-order LP

Sections 4 and 5 are a negative result and a modest positive one: GPU pricing loses by 3-5x, CPU threading wins 1.4-1.8x end to end. 4.1 explains why no amount of kernel work would have changed the first one -- a simplex iteration needs a host decision, so it pays a synchronize per iteration, and that floor (26.7 us) exceeds the entire CPU pricing pass on most models.

The constraint is the algorithm, not the implementation. **`docs/architecture/PDLP.md` documents the algorithm that removes it**: restarted average PDHG, whose iteration needs no host decision at all, so a 64-256 iteration window is queued and synchronized once. Measured: **127 iterations per host synchronize**, against simplex's 1. That engineering claim holds cleanly and is the solid result of this work.

Whether removing the synchronize makes the *solver* faster is a separate question, and the answer is **it depends on the model, and on the small Netlib models it usually does not.** Measured in one process with nothing else running:

| sweep | instances | GPU PDLP | CPU simplex | |
|---|---|---|---|---|
| under 2,500 rows | 26 attempted, 20 converged | 110.35 s | 25.11 s | **PDLP 4.4x slower** |
| 2,500-20,000 rows | 4 attempted, 4 converged | 15.60 s | 24.34 s | **PDLP 1.56x faster** |

| instance | nnz | GPU PDLP | CPU simplex | |
|---|---|---|---|---|
| `maros-r7` | 144,848 | **0.93 s** | 3.85 s | 4.1x faster |
| `fit2p` | 50,284 | **1.37 s** | 6.88 s | 5.0x faster |
| `dfl001` | 35,632 | **1.13 s** | 792.16 s *(iteration limit)* | solves what simplex cannot |
| `stocfor3` | 68,627 | 13.29 s | 13.62 s | tie |
| `degen3` | 24,646 | 1.44 s | 1.55 s | tie |
| `d2q06c` | 32,417 | 12.64 s | 4.58 s | 2.8x **slower** |
| `cycle` | 20,720 | 7.28 s | 0.252 s | 29x **slower** |

6 of the 26 small-set instances never converged within 40 s -- all of which the simplex solves in under 5 s. Both directions are real, and the losing direction is larger and more common than the winning one on this benchmark set.

**What predicts the winner is PDLP's iteration count, not problem size and not degeneracy** -- see `PDLP.md` §5. Since that count is unknowable before solving, `LpMethod` is a caller-facing choice rather than a hidden heuristic.

### What this means for H1

H1 asked whether GPU-accelerated sparse linear algebra can materially accelerate repeated LP operations for refinery-scale sparse models. The honest verdict is **split, and mostly negative at this scale**:

- **Inside the simplex: rejected.** 3-5x slower, for a structural reason that more kernel engineering cannot fix.
- **As a synchronization argument: supported.** A sync-free iteration really does amortize 127 iterations per host round-trip, exactly as designed.
- **As a wall-clock claim: not supported on the Netlib set.** 4.4x slower in aggregate under 2,500 rows, 1.56x faster above it, with one instance solved that the simplex cannot solve at all.

The per-iteration cost is nearly flat across a 20x range of nonzeros (57 us at 7,777 nnz, 111 us at 144,848) -- the signature of latency-bound execution. That is the strongest available evidence that the GPU path has headroom at larger scale, and it is also the reason it loses here: on models this small the device is mostly idle. **That headroom is an inference from a cost curve, not a measurement, and this repository does not contain a model large enough to test it.**

An earlier version of this section reported `degen3` at 39x and `d2q06c` at 8x and declared H1 supported. Those numbers came from a contaminated benchmark -- concurrent processes oversubscribing the CPU -- and the mechanism they were used to argue for (degeneracy) was wrong as well. `PDLP.md` §5 records the error and the methodology rules that came out of it.
