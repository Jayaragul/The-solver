# First-Order LP on the GPU (PDLP)

`src/cuda/GpuPdlp.hpp`, `src/cuda/PdlpKernels.cu`, `LpMethod::FIRST_ORDER` in `src/lp/LpSolver.hpp`.

---

## 1. Why this exists

`CPU_GPU.md` §4 measured GPU-accelerated simplex pricing losing to CPU pricing by 3–5× and traced the cause exactly. It is not kernel quality:

| | |
|---|---|
| cuSPARSE SpMV, **with a sync behind it** | 33–52 µs |
| the same SpMV, **queued** | 15–26 µs |
| empty-kernel launch + synchronize | 26.7 µs |

A simplex iteration cannot proceed until the host knows which column entered. That forces one device synchronize per iteration, which puts every SpMV in the expensive column and adds a fixed ~27 µs floor on top. Three rounds of kernel work — fusing the reduced-cost assembly into the argmax, collapsing four Devex launches into one, deleting a host-to-device transfer — moved a 230 µs iteration to 225 µs. The bottleneck was never the kernels.

**The constraint is the algorithm's sequential decision chain, so the response is a different algorithm.**

PDHG's iteration is

```
y  <-  prox( y + sigma * A xbar )
x  <-  proj( x - tau * (c + A^T y) )
```

Two SpMVs and two elementwise updates, with no data-dependent branch the host has to resolve. An entire restart window — 64 to 256 iterations — is queued and synchronized **once**. The measured result is **~128 iterations per host synchronize** against simplex's 1.

prompt.md §3.2 lists "first-order LP iterations" among the GPU candidates; SOTA.md §1.3b warns off GPU simplex. Both are right, and this is why.

---

## 2. Formulation

The LP is put in two-sided form

$$\min\ c^\top x \quad\text{s.t.}\quad r_l \le Ax \le r_u,\quad l \le x \le u$$

which `LpProblem`'s convention ($Ax + s = \text{rhs}$, slack bounds) maps onto directly: $r_l = \text{rhs} - s_u$, $r_u = \text{rhs} - s_l$.

As a saddle-point problem, $\min_x \max_y\ c^\top x + y^\top A x - \sigma_C(y)$ with $C = [r_l, r_u]$. Chambolle–Pock gives the two updates above; the dual prox comes from Moreau's identity,

$$\mathrm{prox}_{s\,\sigma_C}(v) = v - s\,\mathrm{proj}_C(v/s)$$

which is what turns an abstract support function into a **clamp** — the reason a dual step costs one elementwise pass rather than a solve. Infinite bounds need no special case: `fmin`/`fmax` against $\pm\infty$ simply never bind, which is exactly the semantics of a free row or free variable.

---

## 3. What the kernels do

Per iteration, four queued operations and **zero synchronizations**:

| | |
|---|---|
| `cusparseSpMV` | $A\bar{x}$ |
| `k_dual_update` | Moreau prox + running average, one pass |
| `cusparseSpMV` | $A^\top y$ |
| `k_primal_update` | projection, extrapolation $\bar{x} = 2x^{+} - x$, running average, one pass |

Three design choices follow from op count being the binding cost:

- **The averages are folded into the update kernels.** At four operations per iteration a fifth would be a 25% increase in queue depth for one add.
- **`set_vectors` binds cuSPARSE's descriptors to the caller's buffers once per window**, so the extrapolated point is written straight into the buffer the next SpMV reads. No copies, no rebinding inside the loop.
- **$A$ and $A^\top$ are both stored device-resident.** That costs a second copy of the nonzeros and buys two things worth more: cuSPARSE's transpose mode is markedly slower on CSR, and fixed bindings need fixed operators.

Convergence checking costs 2 SpMVs + 1 reduction kernel + a **64-byte** transfer, once per window.

### Determinism

`NUMERICS.md` §1 requires bit-reproducible results, and the threadfence reduction makes *which block retires last* genuinely nondeterministic. That is harmless here because the combine is not order-sensitive: the last block sums the per-block partials in **ascending block index**, and the grid is a pure function of vector length. Same input, same answer, bit for bit — asserted by `pdlp_is_reproducible_across_repeated_solves` with exact equality.

---

## 4. Scaling is part of the algorithm

A first-order method's convergence rate depends directly on the conditioning of $A$, so `LpMethod::FIRST_ORDER` applies **Ruiz equilibration, then Pock–Chambolle preconditioning** on the result (`Scaling.hpp`). They do different jobs: Ruiz equalizes row and column magnitudes, which is what a basis factorization cares about; Pock–Chambolle targets $\|RAC\|_2 \le 1$, which is what the step size cares about. Skipping either turns models that converge in thousands of iterations into models that do not converge.

---

## 5. MEASURED — where it wins and where it does not

Netlib feasible set, `eps = 1e-6`, restart period 256, against the CPU simplex **in the same process, through the same pipeline, with nothing else running**. RTX 3050 Laptop (14 SMs), CUDA 13.3 under WSL2. Two sweeps: models under 2,500 rows (40 s limit) and models from 2,500 to 20,000 rows (60 s limit).

### The aggregate: PDLP loses the small sweep and wins the large one

The two sweeps split by row count because that is how they were run, not because row count is what decides the winner -- it is not, as the next-but-one subsection shows. Read these totals as a summary of two sets of models, not as a threshold.

| sweep | instances | PDLP total | CPU simplex total | |
|---|---|---|---|---|
| under 2,500 rows | 26 attempted, 20 converged | 110.35 s | 25.11 s | **PDLP 4.4× slower** |
| 2,500–20,000 rows | 4 attempted, 4 converged | 15.60 s | 24.34 s | **PDLP 1.56× faster** |

The large-set figure **excludes `dfl001`**, where the simplex does not finish at all; including it the ratio reads 48.8×, which is an artifact of comparing against a failure and is not quoted as a speedup.

**The large end, instance by instance:**

| instance | rows | nnz | PDLP iters | GPU PDLP | CPU simplex | |
|---|---|---|---|---|---|---|
| `maros-r7` | 3,136 | 144,848 | 8,448 | **0.93 s** | 3.85 s | **4.1× faster** |
| `fit2p` | 3,000 | 50,284 | 18,944 | **1.37 s** | 6.88 s | **5.0× faster** |
| `dfl001` | 6,071 | 35,632 | 17,152 | **1.13 s** | 792.16 s *(iteration limit)* | **solves what simplex cannot** |
| `stocfor3` | 16,675 | 68,627 | 207,616 | 13.29 s | 13.62 s | 1.02× — a tie |

`dfl001` is the standing result: this project's long-unsolved instance, which the simplex abandons after 572,139 iterations and 792 seconds, reaches a verified KKT error of **8.65e-07 in 1.13 seconds**. It is the one place the two methods differ in *outcome* rather than in speed.

**The small end, where it loses:**

| instance | rows | nnz | PDLP iters | GPU PDLP | CPU simplex | |
|---|---|---|---|---|---|---|
| `sctap3` | 1,480 | 8,874 | 1,280 | **0.073 s** | 0.103 s | 1.4× faster |
| `degen3` | 1,503 | 24,646 | 24,320 | 1.44 s | 1.55 s | 1.08× — a tie |
| `80bau3b` | 2,262 | 21,002 | 79,360 | 4.80 s | 0.649 s | 7× slower |
| `woodw` | 1,098 | 37,474 | 25,600 | 1.71 s | 0.172 s | 10× slower |
| `cycle` | 1,903 | 20,720 | 123,392 | 7.28 s | 0.252 s | 29× slower |
| `d2q06c` | 2,171 | 32,417 | 215,552 | 12.64 s | 4.58 s | 2.8× slower |

And **6 of the 26 did not converge at all** within 40 s — `pilot.we`, `pilot.ja`, `bnl2`, `greenbea`, `greenbeb`, `pilot` — every one of which the simplex solves in under 5 seconds. `greenbea` is the worst: 500,000 iterations, 29.9 s, final KKT still 2.15. Of the 26, only 11 converged *and* landed within 1e-6 of the published optimum. **A first-order method that stalls is a real failure mode, not a slow success.**

### What actually predicts the winner

Not problem size, and **not degeneracy**. `woodw` at 37k nonzeros loses by 10×; `fit2p` at 50k nonzeros wins by 5×. `degen3` — the textbook degenerate model — is a tie.

The predictor is **PDLP's own iteration count**, which is a property of the model's conditioning after scaling:

- wins: 8,448 / 17,152 / 18,944 iterations
- tie: ~207,000 (`stocfor3`, against a simplex that also has to work)
- losses: 51,712 / 123,392 / 215,552 / 394,752

PDLP wins when it converges in roughly **under 20,000 iterations and the simplex needs many** — both conditions, since `sierra` converges in 22,272 iterations and still loses because the simplex disposes of it in 0.025 s. Neither quantity is knowable before solving. That is precisely why `LpMethod` is a caller-facing choice rather than a hidden size heuristic: **the engine cannot predict which method wins, so it does not pretend to.**

### The per-iteration cost is flat, and that is the real headroom

| instance | nnz | µs / iteration |
|---|---|---|
| `scfxm3` | 7,777 | 57 |
| `cycle` | 20,720 | 59 |
| `stocfor3` | 68,627 | 64 |
| `maros-r7` | 144,848 | 111 |

A 20× increase in nonzeros costs under 2× per iteration. That is the signature of **dispatch-latency-bound execution**: the four queued operations cost what they cost whether or not there is arithmetic to do. On Netlib-scale models the GPU is mostly idle, which is why the small set loses. It also means the arithmetic has room to grow substantially before the per-iteration cost does — the case for this path is at scales past what Netlib contains, and this repository does not contain the evidence to claim that scale has been reached.

### A measurement error, and how it was caught — twice

The first version of this section reported `degen3` at **3.94 s against 153.83 s — a 39× win** — and `d2q06c` at 8×, and concluded that PDLP beats the simplex on degenerate models. **Every part of that was wrong.** `degen3` is a tie; `d2q06c` is 2.8× *slower*; degeneracy is not the mechanism.

The cause was benchmark processes running concurrently — builds, test suites, and a second sweep — each spawning 16 OpenMP threads on a 16-core machine. Models above `kParallelNnzThreshold` really do run their pricing loops on a thread team, so oversubscription inflated the CPU column badly and **asymmetrically**, while PDLP's GPU work was largely unaffected. The result was a fabricated 39×.

The same error had already been caught once, on `stocfor3`, where a contaminated run showed Devex at 259.67 s against Dantzig's 9.77 s and this document inferred a serious pricing defect from it. Building the per-stage profiler (`benchmarks/profile_simplex.cpp`) settled that one: the real Devex/Dantzig ratio is **2.16×**, exactly what the algorithm predicts, since Devex adds one BTRAN and one O(nnz) pivot-row pass to an iteration that already pays for both. There was never a 31× to explain.

**The failure worth recording is not the first error but the second.** Having caught the `stocfor3` artifact and written the lesson down, the obvious next question — *what else was measured while something was running?* — went unasked, and the answer was the entire PDLP sweep. A correction that fixes one instance and leaves the methodology unexamined is not a correction.

Three standing rules came out of it:

- **Every timing in this repository must come from a single process with nothing else running.** On this machine concurrent benchmark processes are not merely noisy, they are actively misleading, because OpenMP oversubscription hits size-gated parallel paths far harder than serial ones.
- **A predicted-vs-measured gap of an order of magnitude points at the measurement at least as often as at the code.** "This number is too big to be real" was correct both times; "therefore the code has a bug" was the unjustified step.
- **When one number is found to be contaminated, re-take every number gathered under the same conditions,** not just the one that looked wrong.

One caveat on the profiler's own output: it wraps nine `ScopedTimer`s around every iteration, and the same `stocfor3` Devex configuration measures 13.6–14.3 s uninstrumented against 20.3 s under the profiler. Its **ratios** are sound; its **absolute** µs/iteration figures carry roughly 40% instrumentation overhead and should not be quoted as per-iteration cost.

---

## 6. TRIED AND REJECTED — window-granularity adaptive step size

$\eta = 0.9/\|A\|_2$ is a **global** bound, usually far more conservative than local curvature requires. PDLP's adaptive rule (Applegate et al. 2021, §3.1) exploits that and is worth several times the iteration count — but it needs an accept/reject decision **every iteration**, and a host-visible decision per iteration is exactly what makes GPU simplex uncompetitive. Adopting it as written would trade away the reason this solver is on the GPU.

The obvious compromise was implemented and measured: adapt once per check window, growing $\eta$ by 1.25× while the merit function falls, halving it when it rises, capped at 8× the global bound.

**It destroyed the solver.** All 14 mid-size Netlib instances tested failed, with KKT errors reaching 1e+50 and objectives reaching 1e+68. The same instances solved in seconds before the change.

No choice of growth factor rescues this, because the failure is structural: PDHG diverges *geometrically* once $\tau\sigma\|A\|^2 \ge 1$, and a window is 64–256 iterations long. By the time a rising merit function reveals the problem, the iterate is astronomically far away and halving $\eta$ recovers nothing. Applegate's rule works precisely because it rejects the single step that violated the bound, before the damage compounds.

**Conclusion, recorded as a constraint rather than a missing feature:** on this design the step size cannot exceed the global bound without per-iteration backtracking, and per-iteration backtracking costs the sync-free inner loop that makes the GPU worth using. $\eta$ stays at the safe value.

---

## 7. Accuracy, and what is *not* claimed

A first-order method converges fast to moderate accuracy and slowly to high accuracy, and returns an interior-ish point rather than a vertex. Both are properties of the method, not defects of this implementation.

So `GpuPdlp` reports a **verified KKT triple** (relative primal residual, relative dual residual, relative duality gap) and never asserts optimality on its own authority. `solve_lp` puts a `FIRST_ORDER` result through the **same original-space verification gate** every simplex result passes (`NUMERICS.md` §6) — converging on PDLP's own relative KKT test is a claim about the scaled reduced problem, not about the model the caller handed in.

Observed at `eps = 1e-6`, objective error against the published Netlib optimum lands around 1e-6 to 3e-6 — i.e. the KKT tolerance is a reliable proxy but not a bound on objective error. Tightening `eps_optimal` tightens both.

**Not yet built: crossover.** Turning a first-order point into a certified optimal *vertex* means constructing a basis from it and finishing with the simplex. Until that exists, `FIRST_ORDER` gives a fast, verified, near-optimal point, and `SIMPLEX` gives an exact vertex — and the caller chooses. Shipping a half-built crossover would be worse than shipping neither, because it would make the exactness claim without being able to support it.

---

## 8. Validation

`tests/lp/test_pdlp.cpp`, 7 tests, all against **hand-computed** optima rather than against this project's own simplex — two implementations agreeing can mean they share an assumption just as easily as it can mean they are right:

- a two-variable LP whose optimum ($x=2$, $y=6$, objective 36) is solvable on paper
- an equality row (`row_lower == row_upper`, where the projection collapses to a point)
- a two-sided row *range* with wide-open variable bounds, which catches a one-sided or wrong-bound Moreau step
- variable upper bounds binding while the row does not
- the sync-count invariant: `host_syncs < iterations/8`, so a future per-iteration sync fails loudly instead of silently costing all the performance
- bit-exact reproducibility across repeated solves
- agreement with the simplex on afiro, once both have been independently validated

`benchmarks/bench_pdlp.cpp` scores against the **published** Netlib optima, accepting agreement with any of the readme's reference columns (summary/CPLEX/MINOS), which disagree in their last digits.
