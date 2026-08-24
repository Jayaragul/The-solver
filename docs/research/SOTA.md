# SOTA & Research Report — Indigenous Hybrid CPU-GPU LP/MILP/QP Solver

**Target workload:** refinery planning, scheduling, crude/product blending, pipeline logistics (MRPL context)
**Status:** PHASE 1 — DEEP RESEARCH & SOTA ANALYSIS (research-only; no solver code has been written)
**Scope discipline:** every non-trivial technical claim below is labeled ESTABLISHED METHOD / ENGINEERING TECHNIQUE / PROPOSED MODIFICATION / RESEARCH HYPOTHESIS / IMPLEMENTATION DECISION, and every factual claim is labeled VERIFIED FACT / LITERATURE EVIDENCE / ASSUMPTION / HYPOTHESIS. No benchmark numbers in this document were fabricated; numbers quoted from third-party papers are attributed and are **not** claims about this project's own solver, which does not exist yet.

---

## 0. Executive Technical Assessment

Two facts dominate everything else in this report.

**First**, the target hardware for this project's own benchmarking is a single NVIDIA RTX 3050 Laptop GPU (compute capability 8.9, 4.29 GB VRAM, 14 SMs — measured directly on the dev machine, not a vendor spec sheet). Every GPU speedup number cited below from PDLP/cuOpt/cuPDLP literature was measured on datacenter-class GPUs (A100-class or better) against large-scale public benchmark suites (Mittelmann's LP set, MIPLIB). Those numbers are evidence that GPU-accelerated first-order LP methods are a real, publicly documented phenomenon — they are **not** a prediction of what this project will measure on this hardware, at refinery-instance scale. Any claim that this project's solver will reproduce those speedup ratios is explicitly out of scope until measured directly (Level 6+ of the validation hierarchy).

**Second**, the research below converges on one clear architectural conclusion, consistent with prompt.md's own constraint that branch-and-bound control flow stays on the CPU: **the highest-confidence GPU opportunity in this domain is not "GPU simplex" or "GPU branch-and-bound."** Multiple independent literature threads (§1.1, §1.3b) report that GPU-parallelized simplex has repeatedly underperformed sequential CPU simplex on sparse LPs, for structural reasons (irregular pivoting, warp divergence, small basis-update kernels that can't amortize launch/PCIe overhead) — this is a **RESEARCH HYPOTHESIS with strong literature support**, not yet independently reproduced by this project. The better-supported GPU opportunities are: (a) cuSPARSE SpMV as a primitive for a PDLP-style first-order LP engine used for large, well-scaled standalone relaxations or as a fast warm-start generator; (b) dense/condensed-system factorization (cuSOLVER/cuBLAS) for IPM normal-equations solves once fill-in densifies them; (c) GPU-batched independent subproblem solves inside decomposition methods (Benders/Dantzig-Wolfe), which map naturally onto refinery scheduling's repeated-LP structure. All three are candidate "kill shots" (§5) requiring direct empirical validation before any production claim.

Refinery-specific claims (ill-conditioning from mixed physical units, degeneracy from interchangeable parallel units/time periods, weak big-M relaxations in scheduling models) remain **largely unverified against actual refinery data** — MRPL's ~15 MMTPA capacity is confirmed (§1.4b), but no published condition-number or degeneracy statistic specific to refinery LP/MILP models was located in this research pass. These remain ASSUMPTIONs carried forward from prompt.md's own framing, pending direct measurement once representative test instances exist.

---

## 1. Literature Review & Solver Comparison

### 1.1 Open-Source Solver Analysis: HiGHS and SCIP

This section analyzes the internal architecture of **HiGHS** ([ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS)) and **SCIP** ([scipopt.org](https://scipopt.org)) against the specific structural properties of large sparse refinery-scale MILPs: crude-blending models with many interchangeable parallel units and repeated time periods (a source of combinatorial symmetry), scheduling formulations with highly degenerate LP relaxations, and weak big-M disjunctive constraints for mode/state switching. Claims are labeled per the report's taxonomy; where a specific benchmark on refinery-type instances could not be located, the claim is marked as a HYPOTHESIS requiring experimental validation rather than asserted as fact.

**Presolve.** Both solvers implement the now-standard family of bound tightening, singleton/duplicate-row elimination, coefficient strengthening, and dual fixing reductions (ESTABLISHED METHOD; LITERATURE EVIDENCE: Achterberg, Bixby, Gu, Rothberg & Weninger, "Presolve Reductions in Mixed Integer Programming," *INFORMS Journal on Computing* 32(2), 2020). HiGHS's presolve/crash/postsolve stack is documented in Galabova's 2022 PhD thesis (ENGINEERING TECHNIQUE). SCIP's presolve loop runs to convergence across dozens of presolver plugins (Bestuzheva et al., 2021). RESEARCH HYPOTHESIS: for crude-blending models where identical unit/period blocks appear as literal duplicate or near-duplicate rows and columns, generic duplicate-detection presolve may collapse *some* redundancy, but is not designed to detect *permutation* symmetry between non-identical-but-isomorphic blocks — that is a distinct problem handled (if at all) by symmetry detection, not presolve; the interaction between the two phases is unverified for this problem class.

**LP Algorithms: Simplex and Interior-Point.** HiGHS defaults to dual simplex (VERIFIED FACT: Huangfu & Hall, "Parallelizing the dual revised simplex method," *Mathematical Programming Computation* 10(1), 2018, describing the PAMI/SIP parallel dual-simplex variants), with primal simplex, the IPX interior-point solver (Schork, 2020), and a newer parallel direct-factorization IPM ("HiPO") as alternatives. SCIP does not implement its own simplex; it delegates to a pluggable LP solver, defaulting to SoPlex (Wunderling, PhD thesis, TU Berlin, 1996) — HiGHS can also serve as SCIP's LP backend. RESEARCH HYPOTHESIS: refinery scheduling LPs re-solved thousands of times inside branch-and-bound have LP relaxations that are highly degenerate at the optimum (many tied reduced costs from symmetric units/periods); warm-started dual simplex should in principle amortize this well, but the degree to which repeated degenerate re-optimization erodes the practical benefit of warm starts on this problem class is unverified and would need direct profiling.

**Basis Management and Factorization.** Both lineages use LU factorization of the basis matrix with sparsity-preserving updates (Bartels & Golub, *CACM* 12(5), 1969; Forrest & Tomlin, *Mathematical Programming* 2, 1972 — ESTABLISHED METHOD). HiGHS also offers direct multifrontal/up-looking factorizations with METIS/AMD/RCM fill-reducing orderings for its IPM path. RESEARCH HYPOTHESIS: refinery-scale sparse constraint matrices with block-diagonal-plus-coupling structure (unit balances coupled loosely across time periods) may or may not exploit generic fill-reducing orderings as well as a structure-aware ordering tailored to the block-angular pattern — open question, not a demonstrated deficiency of either solver.

**Degeneracy Handling.** Both use Harris-type two-pass ratio tests with bound-flipping and Devex/steepest-edge-style pricing (ESTABLISHED METHOD; Huangfu & Hall 2018; Maros, *Computational Techniques of the Simplex Method*, 2003). HYPOTHESIS: refinery blending/scheduling LPs are frequently degenerate because many feasible unit-assignment or blend-ratio bases yield identical objective value; this can cause dual-simplex stalling and destabilize pseudocost estimates carried across B&B nodes — no refinery-specific degeneracy benchmark was found; remains an ASSUMPTION pending experiments.

**Cut Generation.** SCIP implements a broad cutting-plane arsenal (Gomory MI cuts, MIR cuts per Marchand & Wolsey 2001, knapsack-cover/clique cuts, dedicated cut-selection since SCIP 8.0; Bestuzheva et al. 2021/2023 — ESTABLISHED METHOD). HiGHS's branch-and-cut MIP solver also generates cuts, but the specific families and their maturity relative to SCIP's could not be confirmed from available documentation — flagged UNVERIFIED. RESEARCH HYPOTHESIS: generic MIR/Gomory cuts derived from single/aggregated rows are largely blind to unit/time-period symmetry; on symmetric blending models they may generate many near-duplicate cuts rather than a single symmetry-reduced cut — needs direct instrumentation.

**Branching Strategies.** Both implement pseudocost/reliability branching (ESTABLISHED METHOD: Achterberg, Koch & Martin, *Operations Research Letters* 33, 2005). HYPOTHESIS: on weak big-M formulations typical of refinery mode-switching, fractional LP solutions provide little branching signal because the relaxation is far from any integer solution's basin, and symmetric variable classes may vote inconsistently for near-identical branching candidates (PROPOSED MODIFICATION territory: symmetry-aware branching-candidate selection).

**Node Selection.** SCIP defaults to best-estimate search with plunging (Achterberg thesis, 2007); HiGHS uses best-bound-oriented selection (ENGINEERING TECHNIQUE). HYPOTHESIS: on degenerate refinery LPs, the pseudocost-derived "estimate" used to rank nodes is itself built on the same unreliable degenerate signal noted above — untested.

**Primal Heuristics.** SCIP's suite (rounding, diving, RENS, feasibility pump; Berthold, TU Berlin diploma thesis, 2006) is comprehensively documented (ESTABLISHED METHOD). HiGHS's heuristic set is comparatively less documented in peer-reviewed literature (ENGINEERING TECHNIQUE). ASSUMPTION: rounding/diving heuristics that ignore blend-ratio coupling across units may frequently produce LP-feasible-but-not-implementable roundings in crude blending — plausible, not measured.

**Symmetry Handling — the sharpest documented architectural asymmetry between the two solvers.** SCIP has included dedicated symmetry-handling machinery since version 5.0 (bliss, then nauty/traces/SASSY/dejavu), with orbital fixing, orbitope/symresack symmetry-breaking constraints, and orbital branching per the broader literature (VERIFIED FACT + ESTABLISHED METHOD: Bestuzheva et al. 2021/2023; Pfetsch & Rehn, *Mathematical Programming Computation* 11, 2019; Ostrowski, Linderoth, Rossi & Smriglio, *Mathematical Programming* 126(1), 2011). No documented equivalent subsystem for HiGHS was found (UNVERIFIED absence, inferred from lack of documentation, not a source-code audit). RESEARCH HYPOTHESIS, high relevance to this project: crude-blending models with interchangeable parallel units/repeated periods plausibly generate automorphism groups whose order grows combinatorially with unit/period count; a solver without dedicated symmetry handling would be expected to explore many symmetric-equivalent subtrees, while SCIP's machinery should in principle prune these — but the *actual* magnitude on refinery-realistic (as opposed to synthetic combinatorial) symmetry is an experimental question, since real blending symmetry is often only *approximate* (near-identical but not exactly identical capacities/costs), which weakens or defeats exact automorphism detection entirely.

**Numerical Stability.** Both rely on scaling, Harris ratio tests, and periodic refactorization (ESTABLISHED METHOD; Maros 2003; Bixby, *Operations Research* 50(1), 2002). HiGHS's IPX path uses a Krylov inner solve for the normal-equations system, a different numerical failure mode than simplex-based paths. HYPOTHESIS: weak big-M refinery formulations combined with wide-ranging blend-quality coefficients (ppm-level sulfur/octane specs alongside bulk flow variables in the same rows) create ill-conditioned scaling problems that generic row/column scaling handles only partially — a structural formulation concern, not a solver defect, but interacting with solver-level scaling in ways needing instance-specific testing.

**Parallelism.** HiGHS's parallel dual simplex (Huangfu & Hall 2018) parallelizes within a single LP solve; its MIP solver is "largely single-threaded" per current documentation (multithreaded prototype reported in progress). SCIP supports parallelization primarily via external frameworks (UG/ParaSCIP) rather than a built-in shared-memory B&B parallelizer. RESEARCH HYPOTHESIS: for refinery MILPs whose tree size is inflated by unresolved symmetry, tree-parallelism is a poor substitute for symmetry reduction — parallel workers would simply explore symmetric subtrees concurrently rather than eliminating the redundant work.

**Memory Behavior.** LU-update-based simplex keeps memory roughly proportional to nonzeros in the factors plus fill-in, periodically refactorized to bound growth (ESTABLISHED METHOD). ASSUMPTION: on trees inflated by unhandled symmetry, memory pressure from a very large open-node set could become a practical limiter before time does — plausible, not benchmarked.

### 1.2 Commercial Solver SOTA (Gurobi, CPLEX)

**Scope note.** Gurobi and CPLEX are closed-source. Everything below is drawn from vendor reference manuals, vendor conference talks, and peer-reviewed retrospectives (notably Bixby and Achterberg/Wunderling). No proprietary source code, file format, or internal heuristic constant was inspected or inferred.

**Presolve — ESTABLISHED METHOD / VERIFIED FACT.** Both apply bound strengthening/propagation, coefficient strengthening, variable aggregation/substitution, and removal of redundant rows/columns, tunable via aggressiveness parameters [Gurobi Optimizer Reference Manual; Achterberg & Wunderling 2013].

**Primal/dual simplex — VERIFIED FACT.** Both implement revised primal and dual simplex, with algorithm choice exposed to the user; CPLEX documents dual simplex as generally best for the majority of LPs, with primal, network, barrier, and sifting as alternatives.

**Barrier/interior-point — ESTABLISHED METHOD.** Both offer primal-dual predictor-corrector barrier as a distinct method, well suited to large sparse LPs, returning a non-vertex solution.

**Crossover — VERIFIED FACT.** Gurobi documents crossover as primal push, dual push, and simplex cleanup phases recovering a basic solution from a barrier solution; CPLEX documents an analogous selectable crossover mode.

**Concurrent optimization — VERIFIED FACT.** Gurobi's concurrent optimizer races multiple algorithm instances on separate threads (documented default for LP: one thread dual simplex, several parallel barrier, one primal simplex) and returns whichever finishes first; `ConcurrentMIP` runs independent B&B searches in parallel for MIP. This is the clearest public description of vendor "algorithm racing" and is directly transferable as a design pattern.

**Cut generation / branch-and-cut — VERIFIED FACT.** Gurobi's manual lists separable cut families — Gomory, MIR, cover, clique, flow-cover, flow-path, implied-bound, GUB-cover, zero-half, mod-k, StrongCG, network — each tunable, globally scaled by a `Cuts` parameter, interleaved with branching in a branch-and-cut loop (Achterberg & Wunderling 2013).

**Strong/pseudo-cost/reliability branching, node selection — LITERATURE EVIDENCE.** Reliability branching (strong branching for unreliable pseudo-costs, pseudo-cost fallback otherwise) is documented as the de facto default in modern solvers including CPLEX; Gurobi documents an automatic `VarBranch` default without disclosing exact internals.

**Symmetry handling — VERIFIED FACT.** Gurobi documents a `Symmetry` parameter and states it uses orbital fixing/probing to break MIP symmetry [Gurobi support docs; Wunderling, Gurobi Days 2022].

**Primal heuristics — VERIFIED FACT.** Gurobi documents a `Heuristics` runtime-fraction parameter and a distinct `NoRelHeur` heuristic that searches for feasible solutions without solving any LP relaxation.

**Parallelism, numerical stabilization — VERIFIED FACT.** Both barrier and branch-and-cut are multi-threaded; Gurobi's "Solver Parameters to Manage Numerical Issues" page documents scaling and tolerance/pivoting-stability controls jointly as numerical-stability tools.

**Aggregate historical evidence.** Bixby (2012) and Achterberg & Wunderling (2013) are the canonical retrospectives quantifying commercial MIP progress; Koch et al. (2022, arXiv:2206.09787) corroborates a widely cited ~2× algorithmic speedup every ~13 months on fixed hardware, separate from and multiplicative with hardware gains — LITERATURE EVIDENCE, best read as directionally correct industry consensus rather than a precisely reproducible figure (partial vendor provenance in the original curation).

### 1.3 Academic/Industrial Breakthroughs (PDLP, Learn2Branch)

**PDLP — what it is.** Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient," NeurIPS 2021 (arXiv:2106.04756). Applies Chambolle & Pock's primal-dual hybrid gradient method to the LP saddle-point formulation, with diagonal preconditioning, presolve, adaptive step sizes, and adaptive restarting. Per-iteration cost is a matrix-vector multiply against the constraint matrix — why it scales past simplex/barrier's factorization-based linear algebra and why it's naturally GPU-friendly. VERIFIED FACT (paper's own benchmark): on 383 MIPLIB-2017-derived LP instances at 10⁻⁸ accuracy, PDLP achieves ~6.3× geometric-mean speedup over SCS and reduces unsolved instances from 227 to 49 — a comparison against another first-order method, **not** against mature simplex/barrier solvers on typical-accuracy, typical-size LPs.

**GPU lineage — VERIFIED FACT.** OR-Tools shipped a CPU multi-threaded implementation (2022); `cuPDLP.jl` (Lu & Yang, arXiv:2311.12180, published *Operations Research* 73(6), 2025) and COPT's `cuPDLP-C` are GPU implementations. This confirms GPU acceleration is an actively pursued direction for this algorithm family, not speculative.

**Is GPU acceleration appropriate here, and does it belong in v1? — IMPLEMENTATION DECISION with RESEARCH HYPOTHESIS reasoning.** PDLP's strength is asymptotic scalability on very large, well-structured LPs solved to moderate/high accuracy as a standalone problem. Refinery LP/MILP workloads are typically (a) moderate-scale relative to PDLP's target regime, and (b) solved repeatedly as the B&B relaxation, where the dominant requirement is *warm-starting from a parent node's basis* — a natural fit for simplex, not for a first-order method whose iterates are not basic and which lacks a mature warm-start story inside branch-and-cut. First-order methods are also generally more sensitive to poor conditioning/degeneracy than factorization-based methods. **Recommendation: defer from core B&B relaxation-solving in v1.** Reasonable to revisit for a narrow, verified use case (standalone huge LP without repeated warm-starting). **Risk if adopted prematurely:** silent accuracy/robustness regressions on ill-conditioned refinery models, and a second numerical code path to validate and certify.

**Learn2Branch — what it is.** Gasse, Chételat, Ferroni, Charlin & Lodi, "Exact Combinatorial Optimization with Graph Convolutional Neural Networks," NeurIPS 2019 (arXiv:1906.01629). Trains a GCNN branching policy via imitation learning from a strong-branching oracle over the bipartite variable-constraint graph. VERIFIED FACT: the learned policy matches/improves prior ML branching baselines and generalizes to larger instances *within the same training distribution*.

**Risks and production suitability — LITERATURE EVIDENCE + RESEARCH HYPOTHESIS.** Follow-on literature documents recurring open problems: (1) degraded performance outside the training distribution — a policy trained on one refinery model family isn't guaranteed to transfer to another; (2) expensive imitation-learning data collection (requires running strong branching, the thing being approximated); (3) a GPU/ML-runtime dependency and per-node inference latency in the innermost B&B loop; (4) B&B soundness is preserved regardless of branching quality, but auditability of *why* a branch was chosen is harder to certify than a documented deterministic rule — relevant for a refinery tool that may need explainable, reproducible decisions for operational sign-off. **Recommendation: do not include in v1.** Reliability branching should be the v1 baseline; learned branching remains a plausible long-term research hypothesis worth prototyping offline once the core engine is stable.

### 1.3b GPU Sparse Linear Algebra and Batched/First-Order LP Research

**cuSPARSE: SpMV as an arithmetic primitive — ESTABLISHED METHOD.** The generic API (`cusparseSpMV`, `_bufferSize`, `_preprocess`) is the mature production primitive for sparse matrix-vector products. `CSR_ALG1`/`ALG2` variants trade determinism for speed (ALG1 is not bit-reproducible run-to-run; ALG2 is deterministic at a documented performance cost). FP64/FP32 uniform precision is supported; mixed-precision paths (INT8/FP16/BF16-in, FP32-accumulate) target ML workloads, not LP-grade accuracy. Non-transposed SpMV is ~3× faster than transposed, which matters because primal-dual/simplex-adjacent kernels alternate `Ax` and `Aᵗy` — storing both CSR and CSC (or CSR of Aᵗ) is the standard workaround. `cusparseSpMV_preprocess` amortizes analysis cost across repeated calls on the same matrix — exactly the access pattern of an iterative first-order method. **IMPLEMENTATION DECISION rationale:** SpMV is memory-bandwidth-bound (~2 FLOP/nonzero read), so it maps well to GPU only when the matrix is large enough to amortize launch overhead and saturate memory bandwidth — favoring PDLP-style repeated-SpMV methods over classical simplex's many tiny, structurally-changing basis operations.

**cuSOLVER: dense and sparse factorization — ESTABLISHED METHOD, with caveats.** `cusolverDn` (dense Cholesky/LU/QR) and `cusolverSp` (sparse CSR Cholesky/LU/QR) are current production APIs (cuSOLVER 13.3, confirmed on this project's dev machine). `cusolverSp*csrlsvchol` supports only a single RHS per call, and the sparse LU path historically routes symbolic/numeric factorization partly through host-side code, limiting it as a scalable basis-factorization engine. Dense `cusolverDn` factorizations suit normal-equations/Schur-complement systems once fill-in densifies them — compute-bound, high-arithmetic-intensity, GPU-appropriate. Batched sparse QR solves exist for many small independent systems but are more expensive than LU/Cholesky.

**cuDSS — RESEARCH HYPOTHESIS / not yet production-grade.** NVIDIA's dedicated GPU sparse direct-solver library (LDU/LDLᵀ/LLᵀ, multi-GPU/multi-node) is architecturally exactly what IPM normal-equations Cholesky or sparse LU basis factorization needs, but is explicitly shipped as **preview**: API subject to change, and documented as non-deterministic (no bit-wise reproducibility) between runs. Independent benchmarking has observed symbolic-analysis/factorization setup cost dominating total runtime and earlier out-of-memory behavior than iterative alternatives on very large problems. **IMPLEMENTATION DECISION: track and prototype against, but do not adopt as a v1 dependency** — preview-API breakage risk and non-determinism are incompatible with reproducibility requirements for a production refinery solver.

**GPU simplex/IPM literature — largely negative results, RESEARCH HYPOTHESIS with strong support.** GPU simplex has been studied for over a decade (Ploskas & Samaras; multi-GPU implementations; hybrid CPU-B&B/GPU-simplex schemes) with a consistent finding: **for sparse LPs, sequential CPU simplex still tends to outperform parallelized GPU implementations**, for structural reasons — irregular, data-dependent pricing/ratio-test branching causes warp divergence; basis updates are inherently sequential with fine-grained irregular sparse memory access poorly suited to SIMT execution; typical LP/basis sizes are often too small to amortize kernel-launch and PCIe overhead. Interior-point methods fare somewhat better (Cholesky on normal equations is more regular than simplex pivoting) but reported end-to-end IPM speedups from GPU-accelerated factorization remain modest, since factorization doesn't dominate total runtime and fill-in can erode sparsity advantage. Condensed-space IPM reformulations (e.g., GPU-accelerated optimal power flow work) mitigate this via a smaller dense SPD system, at a conditioning cost. **PROPOSED MODIFICATION:** this argues against naively porting simplex to CUDA, and toward either (a) GPU-accelerating only the regular high-arithmetic-intensity kernels while keeping pivoting/pricing control flow on CPU, or (b) adopting a GPU-native algorithm (PDLP family).

**PDLP family as a GPU-native first-order method — ESTABLISHED METHOD for a specific regime.** `cuPDLP.jl` (*Operations Research* 73(6), 2025), `cuPDLP-C`, and `cuPDLPx` replace factorization with repeated sparse `Ax`/`Aᵗy` — exactly cuSPARSE's sweet spot. This is the central reason GPUs help LP at all: trading irregular sequential factorization for regular parallel matrix-vector products, at the cost of needing more iterations and typically only moderate accuracy without careful restart/scaling heuristics. Results are regime-dependent, **not uniformly favorable**: NVIDIA cuOpt reports 10×–5000× speedup over CPU solvers on a subset of Mittelmann's benchmark, while independently tracked benchmarks have also shown cuPDLP-C running ~1.8–2.3× **slower** than the best CPU solver for reaching an optimal *basic* (vertex) solution — first-order methods don't naturally produce a basic feasible solution, which matters for simplex-based sensitivity analysis or B&B warm-starts. These different figures come from different benchmark subsets/tolerances and must not be conflated into one "GPU is Nx faster" headline. **PROPOSED MODIFICATION:** a GPU-native PDLP-style engine is a credible first-release candidate for large, well-scaled LP relaxations at moderate accuracy, used standalone or as a fast warm-start/bound generator feeding a CPU simplex crossover step — mirroring how COPT/HiGHS/Xpress/Gurobi have integrated cuPDLP-derived methods.

**Mixed-precision iterative refinement — ESTABLISHED METHOD.** FP32/FP16(tensor-core) factorization with FP64 residual-correction refinement (Haidar, Tomov, Dongarra & Higham, SC18, 2018: 4–5× speedup while retaining FP64 backward-error guarantees; extended to sparse factorizations, ACM TOMS 2023) is well established for dense systems and being extended to sparse. **ENGINEERING TECHNIQUE** appropriate for IPM normal-equations/KKT solves, where conditioning is known to degrade near the optimum — must be paired with monitored refinement convergence and fallback to full FP64 when the condition estimate is too large.

**Decomposition methods (Benders, Dantzig-Wolfe) — ESTABLISHED METHOD; GPU mapping is PROPOSED MODIFICATION.** Benders (Rahmaniani, Crainic, Gendreau & Rei, *EJOR* 259, 2017) and Dantzig-Wolfe decomposition are classical for block-structured LPs, with documented application to real-time oil-field production optimization (Gunnerud, Foss & Torgnes, *Journal of Process Control* 20(9), 2010) and integrated refinery planning. Each iteration solves many independent, often similarly-shaped recurring subproblem LPs — a natural batch target. **RESEARCH HYPOTHESIS:** whether batching is GPU-appropriate depends on subproblem size/homogeneity; if subproblems are numerous, small, and structurally similar, batched block-diagonal SpMV/PDLP or many concurrent CUDA-stream small solves are plausible; if subproblems are simplex-like and control-flow-heavy, the same divergence/irregular-access problems documented above apply per-subproblem. **IMPLEMENTATION DECISION:** treat as a second-phase research prototype requiring validation against representative refinery subproblem shapes, not a v1 commitment.

### 1.4 Numerical Pathology Research

**1.4.1 Ill-Conditioning.** A constraint matrix $A$ is ill-conditioned when $\kappa(A) = \|A\|\|A^{-1}\|$ is large; refinery models mixing barrels/day, weight fractions, sulfur in ppm, and dollar costs in the same matrix is a structural driver of poor scaling (**ASSUMPTION**, not yet measured on a concrete instance). **Ruiz equilibration (ESTABLISHED METHOD:** Ruiz, RAL-TR-2001-034, 2001) iteratively rescales rows/columns by the reciprocal square root of their norm, converging linearly (asymptotic rate 1/2 for ∞-norm) and preserving symmetry — useful for QP Hessians/KKT systems, and stronger than one-shot geometric-mean diagonal scaling (**ENGINEERING TECHNIQUE**). **Condition estimation (ESTABLISHED METHOD):** Hager's 1-norm estimator (SIAM J. Sci. Stat. Comput. 5, 1984) avoids explicit inversion via matrix-vector products against an existing factorization; folded into LAPACK's `condest`-style routines. **IMPLEMENTATION DECISION:** implement Hager-style estimation as a post-factorization diagnostic, not full SVD. **Iterative refinement/mixed precision (ESTABLISHED/ENGINEERING TECHNIQUE):** classical Wilkinson-era residual correction in extended precision; refinery-specific benefit is a **RESEARCH HYPOTHESIS**. **Pivoting/rank deficiency (ESTABLISHED METHOD):** Markowitz threshold pivoting (*Management Science*, 1957) balances fill-in against stability; redundant refinery mass-balance rows (component balance implied by overall + other component balances) typically handled via presolve redundancy detection rather than hoping factorization survives. **Limitation:** scaling changes numerical representation, not true conditioning, and can hide feasibility signals or interact badly with tolerance-based stopping criteria — directly connects to degeneracy handling below.

**1.4.2 Degeneracy.** Primal degeneracy: a basic feasible solution with a zero-valued basic variable (multiple bases, one vertex). Dual degeneracy: a nonbasic variable with zero reduced cost (non-unique optimal face). Structurally interchangeable units/time periods plausibly create many bases mapping to the same physical solution (**RESEARCH HYPOTHESIS**, general-LP-theory grounded, not refinery-confirmed). **Bland's rule (ESTABLISHED METHOD:** *Math. of OR*, 1977) guarantees finite termination via smallest-index tie-breaking, at the cost of speed. **Harris two-pass ratio test (ESTABLISHED METHOD:** *Math. Programming* 5, 1973): first pass computes a relaxed step-length bound allowing small tolerance-bounded violation; second pass picks, among rows within that bound, the largest pivot magnitude — trading small controlled infeasibility for stability, directly countering degenerate ties. **EXPAND procedure (ESTABLISHED METHOD:** Gill, Murray, Saunders, Wright, *Math. Programming* 45, 1989) generalizes this via a dynamically expanding feasibility tolerance window, provably preventing stalling from becoming cycling. **Steepest-edge/Devex pricing (ESTABLISHED METHOD:** Goldfarb & Reid 1977; Forrest & Goldfarb 1992; Devex per Harris 1973) reduce iteration counts on degenerate problems by weighting reduced cost by edge length, at added per-iteration bookkeeping cost. **IMPLEMENTATION DECISION:** prioritize Harris-ratio-test-style anti-degeneracy pivoting and Devex/steepest-edge pricing over naive Dantzig pricing and strict-ratio tie-breaking for the from-scratch simplex core.

**1.4.3 Symmetry.** Detected by encoding the MILP as a colored graph and computing its automorphism group (**ESTABLISHED METHOD/TOOL**: nauty, bliss, saucy). **Orbital branching (ESTABLISHED METHOD:** Ostrowski, Linderoth, Rossi, Smriglio, IPCO 2007/*Math. Programming*) partitions variables into orbits at each node and restricts search to one representative per orbit, pruning symmetric subtrees without enumerating the orbit structure explicitly. Notably, the authors' follow-up paper on "modified orbital branching for structured symmetry" uses **unit commitment** (identical generating units over a time horizon) as its motivating application — structurally analogous to refinery scheduling with parallel identical units/tanks (**LITERATURE EVIDENCE** supporting a **RESEARCH HYPOTHESIS**, not yet tested on a refinery instance). **Orbitopal fixing (ESTABLISHED METHOD:** Kaibel, Peinhardt, Pfetsch, IPCO 2007) gives linear-time fixing for assignment/partitioning-into-indistinguishable-groups structures. **IMPLEMENTATION DECISION (cheaper first step):** static symmetry-breaking constraints (lexicographic ordering of interchangeable units/tanks imposed a priori) sacrifice some dynamic pruning power for much lower implementation complexity than full automorphism-group computation. **Limitation, important for this project:** orbital branching/orbitopal fixing work best on *exact* symmetry; real refinery data usually has nominally "identical" units with slightly different capacities, ages, or fouling factors that break exact symmetry — whether *approximate* symmetry-breaking is worthwhile is itself a further open research question, since the cited literature addresses exact symmetry only.

**1.4.4 Weak LP Relaxations and Cuts.** Gomory cuts (**ESTABLISHED METHOD**, Gomory 1958) derive a valid inequality purely from integrality on a fractional tableau row; historically numerically fragile until modern safeguards. MIR cuts (**ESTABLISHED METHOD**, Nemhauser & Wolsey 1990; separation heuristic per Marchand & Wolsey 2001) are the base primitive for many structured cuts. Cover cuts (Crowder, Johnson & Padberg 1983) and their lifted strengthening (Gu, Nemhauser & Savelsbergh 1998) target knapsack constraints. Clique cuts (Padberg 1973) target mutual-exclusion/conflict-graph structure. Flow-cover cuts (Padberg, Van Roy & Wolsey 1985; Van Roy & Wolsey 1986) target single-node fixed-charge flow structures — a continuous flow variable gated by a binary "unit active" variable via a big-M link. **RESEARCH HYPOTHESIS (highest-leverage candidate, not yet validated):** big-M time-indexed refinery scheduling formulations (Lee, Pinto, Grossmann & Park, *Ind. Eng. Chem. Res.* 35, 1996) are structurally single-node fixed-charge/flow constraints — precisely what flow-cover and MIR cuts target. Blending-ratio/quality-spec constraints, when linearized via piecewise-MILP relaxations of bilinear blending terms (Wicaksono & Karimi, *AIChE Journal* 54(4), 2008), inherit similar big-M/disjunctive structure. On this basis, MIR and flow-cover-style cuts plausibly offer the highest leverage of the surveyed cut families for refinery blending/scheduling MILPs — ahead of generic Gomory/clique cuts — but this is a hypothesis to validate computationally, not a literature-established conclusion for this specific domain. **Limitation:** all cut families face diminishing returns and can degrade LP-relaxation conditioning (reconnecting to §1.4.1); separation heuristics are themselves approximate and may miss violated cuts.

### 1.4b MRPL Factuality Check

**VERIFIED FACT** (mrpl.co.in, cross-confirmed by Business Standard): MRPL was commissioned in 1988 at 3.69 MMTPA, expanded through multiple phases to a nameplate capacity of **15.0 MMTPA** (Phase III, ~March 2012) — matching the commonly cited "~15 MMTPA" figure.

**Less firmly verified (secondary press sources, 2023):** a debottlenecking project targeting throughput of ~16.6 → 18.2 MMTPA without major new units, and characterization as India's largest single-location refinery in coastal Karnataka. One search-summarized fragment cited an inconsistent "11.82 MMTPA from 9.69 MMTPA" figure that could not be reconciled with other sources — flagged **UNRELIABLE**, likely a summarization artifact, not repeated as fact.

**COULD NOT BE VERIFIED:** no published condition-number estimates or quantified degeneracy statistics specific to refinery LP/MILP planning models (PIMS-style or otherwise) were located. General secondary sources mention refinery planning LPs on the order of hundreds of equations / roughly a thousand activities as a rough industry characterization — itself not a rigorously sourced statistic. **Any claim about the numerical severity of ill-conditioning or degeneracy in real refinery models must be labeled ASSUMPTION or RESEARCH HYPOTHESIS, not literature fact**, until either a specific published study is located or this project measures its own from-scratch solver against a representative refinery instance directly.

---

## 2. GPU Opportunity Analysis (Synthesis)

Cross-referencing §1.1, §1.3b, and prompt.md's constraint that branch-and-bound control flow must stay CPU-resident, the GPU/CPU division of labor that the evidence above actually supports is:

| Candidate operation | GPU-appropriate? | Basis |
|---|---|---|
| SpMV (`Ax`, `Aᵗy`) on the full constraint matrix, repeated many times | **Yes** | Memory-bandwidth-bound, regular, amortizes launch overhead when matrix/iteration count is large enough (§1.3b) |
| Dense/condensed-system factorization (IPM normal equations after fill-in) | **Yes, conditionally** | Compute-bound, high arithmetic intensity, regular access (§1.3b); benefit shrinks if fill-in is small or system stays sparse |
| Batched independent subproblem solves (Benders/DW decomposition) | **Plausible, unvalidated** | Natural SIMD target only if subproblems are numerous, small, and structurally homogeneous (§1.3b); RESEARCH HYPOTHESIS |
| Revised-simplex pivoting/pricing/basis update | **No (strong literature evidence against)** | Irregular, sequential, warp-divergent; sequential CPU simplex has repeatedly outperformed GPU simplex on sparse LPs (§1.3b) |
| Branch-and-bound tree control (node selection, branching decisions) | **No — prohibited by architecture constraint** | Control-flow-heavy, low arithmetic intensity, and explicitly excluded from GPU residency by this project's design principles |
| PDLP-style first-order LP solve (standalone or warm-start generator) | **Yes, for a specific regime** | GPU-native by construction (SpMV-only inner loop); accuracy/basic-solution caveats apply (§1.3) |
| Sparse direct factorization (cuDSS) for basis/KKT solves | **Not yet** | Preview-status API, non-deterministic results — reproducibility risk (§1.3b) |

This table is a synthesis of the labeled claims above, not a new empirical result — every "yes" above still requires the Level 4–7 validation described in prompt.md's hierarchy before being treated as a production decision.

---

## 3. Evidence / Assumption Table

| Claim | Label | Basis |
|---|---|---|
| MRPL nameplate capacity ≈ 15 MMTPA | VERIFIED FACT | mrpl.co.in, Business Standard (§1.4b) |
| MRPL models have condition numbers approaching 10^12 | ASSUMPTION (carried from prompt.md) | No supporting publication found; not yet measured |
| MRPL models are severely degenerate | ASSUMPTION (carried from prompt.md) | No refinery-specific study found; general LP-symmetry theory is suggestive only |
| MRPL scheduling models have large symmetry | ASSUMPTION (carried from prompt.md) | Plausible by analogy to unit-commitment literature (Ostrowski et al.); not refinery-confirmed |
| MRPL LP relaxations are weak | ASSUMPTION (carried from prompt.md) | Plausible given big-M scheduling formulations in general refinery-scheduling literature; not MRPL-specific |
| GPU-parallelized simplex tends to lose to CPU simplex on sparse LPs | LITERATURE EVIDENCE | Ploskas & Samaras; multi-GPU simplex studies (§1.3b) |
| PDLP achieves large speedups over CPU solvers on some large LPs | LITERATURE EVIDENCE (regime-dependent) | cuOpt (10×–5000× on a Mittelmann subset) vs. independent tracking (1.8–2.3× slower for basic-solution accuracy) — both real, neither universal (§1.3) |
| Reliability branching is the modern default in commercial/open solvers | LITERATURE EVIDENCE | Achterberg & Wunderling 2013; vendor docs (§1.2) |
| SCIP has dedicated symmetry handling; HiGHS does not (as documented) | VERIFIED FACT (SCIP) / UNVERIFIED ABSENCE (HiGHS) | SCIP docs/papers vs. absence of HiGHS documentation (§1.1) — not a source-code audit |
| MIR/flow-cover cuts are the highest-leverage cut family for refinery big-M scheduling models | RESEARCH HYPOTHESIS | Structural argument from Lee/Pinto/Grossmann/Park 1996 formulation pattern matching Padberg/Van Roy/Wolsey cut target (§1.4.4) — not computationally validated |
| cuDSS is production-ready for basis/KKT factorization | FALSE per current evidence | NVIDIA's own docs mark it "Preview," non-deterministic (§1.3b) |
| This project's solver will reproduce literature GPU speedups on its own hardware | NOT YET DETERMINED | Target hardware (RTX 3050 Laptop, 4.29GB) is far below the datacenter GPUs used in cited benchmarks; requires direct measurement (§0) |

---

## 4. Candidate Algorithm Matrix — "Kill Shots"

Per prompt.md's requirement, every candidate below is a hypothesis requiring empirical validation, not a claimed result. Fields: mathematical mechanism, computational mechanism, expected benefit, complexity impact, numerical risk, implementation difficulty, CPU/GPU suitability, validation required.

**KS-1: Ruiz equilibration + Hager condition estimation + iterative refinement (numerical scaling pipeline)**
- *Mechanism:* iterative row/column rescaling to unit norm (Ruiz 2001) + post-factorization κ estimate (Hager 1984) + residual-correction refinement.
- *Computational mechanism:* CPU preprocessing pass, O(nnz) per iteration, small fixed iteration count.
- *Expected benefit:* mitigates ill-conditioning from mixed physical units (ASSUMPTION-level refinery relevance); improves numerical reliability of downstream solves.
- *Complexity impact:* negligible relative to solve cost.
- *Numerical risk:* low; well-established technique. Risk is in interaction with feasibility tolerances (§1.4.1 limitation).
- *Implementation difficulty:* low.
- *CPU/GPU suitability:* CPU (control logic); the rescaling arithmetic itself could be GPU-offloaded but is not bandwidth-bound enough to matter at typical sizes.
- *Validation required:* measure κ(A) before/after on real and synthetic refinery-like instances; confirm feasibility-tolerance interaction is controlled.

**KS-2: Harris ratio test + Devex/steepest-edge pricing (anti-degeneracy simplex core)**
- *Mechanism:* §1.4.2 — relaxed-then-tightened ratio test; edge-length-weighted pricing.
- *Computational mechanism:* O(1) extra bookkeeping per pivot for Devex; steepest-edge needs weight-update recurrence.
- *Expected benefit:* fewer degenerate stalls/cycles on symmetric refinery LPs (RESEARCH HYPOTHESIS).
- *Complexity impact:* modest per-iteration overhead, offset by fewer iterations if hypothesis holds.
- *Numerical risk:* low (both are ESTABLISHED, widely deployed).
- *Implementation difficulty:* medium.
- *CPU/GPU suitability:* CPU only — this is exactly the irregular, sequential pivoting logic §1.3b's GPU-simplex literature warns against.
- *Validation required:* iteration counts on degenerate refinery-like test instances vs. naive Dantzig pricing/strict ratio test.

**KS-3: Static symmetry-breaking constraints for interchangeable units/periods**
- *Mechanism:* a priori lexicographic ordering constraints on structurally interchangeable variables (cheaper alternative to full orbital branching, §1.4.3).
- *Computational mechanism:* added at model-construction time; no runtime automorphism computation.
- *Expected benefit:* reduces symmetric subtree exploration in B&B (RESEARCH HYPOTHESIS, unit-commitment analogy).
- *Complexity impact:* small number of added constraints; low.
- *Numerical risk:* low, but wrong ordering choice can interact badly with branching heuristics.
- *Implementation difficulty:* low-medium (requires detecting which units/periods are actually interchangeable — a modeling-layer concern).
- *CPU/GPU suitability:* CPU (model construction/presolve stage).
- *Validation required:* B&B node count with vs. without breaking constraints on refinery-scheduling-style test instances with known interchangeable units.

**KS-4: Orbital branching / orbitopal fixing (dynamic symmetry handling)**
- *Mechanism:* §1.4.3 — orbit-based branching restriction; linear-time orbitope fixing.
- *Computational mechanism:* automorphism-group computation (nauty/bliss/saucy) at (or before) each node.
- *Expected benefit:* potentially larger tree reduction than KS-3 on exact-symmetry structures (LITERATURE EVIDENCE from unit commitment).
- *Complexity impact:* per-node automorphism computation cost; can be significant if done naively.
- *Numerical risk:* low numerically, but algorithmic risk if symmetry is only approximate (real refinery units differ slightly) — automorphism detection may find nothing.
- *Implementation difficulty:* high.
- *CPU/GPU suitability:* CPU only (control-flow, graph algorithms).
- *Validation required:* must first confirm (via KS-3's simpler mechanism) that symmetry reduction matters at all on real refinery instances before investing in this higher-cost approach.

**KS-5: MIR / flow-cover cuts targeted at big-M scheduling structure**
- *Mechanism:* §1.4.4 — cuts derived from single-node fixed-charge flow structure matching the big-M scheduling pattern.
- *Computational mechanism:* separation heuristic (row aggregation → complementation → MIR generation, per Marchand & Wolsey 2001).
- *Expected benefit:* tighter root relaxation on refinery scheduling MILPs (RESEARCH HYPOTHESIS — highest-leverage candidate per §1.4.4, unvalidated).
- *Complexity impact:* moderate — cut pool management, separation cost per node.
- *Numerical risk:* medium — dense/poorly-scaled cuts can degrade relaxation conditioning (interacts with KS-1).
- *Implementation difficulty:* high.
- *CPU/GPU suitability:* CPU (separation logic is control-flow-heavy).
- *Validation required:* root-relaxation gap closed, with vs. without, on refinery-scheduling-structured test instances.

**KS-6: cuSPARSE-based GPU SpMV for repeated LP operations**
- *Mechanism:* §1.3b — `cusparseSpMV` with `_preprocess` amortization.
- *Computational mechanism:* CSR/CSC matrix resident on device; async H2D once, repeated kernel launches.
- *Expected benefit:* speedup on the SpMV-dominated inner loop of a first-order method or any repeated-`Ax` workload (H1, §6).
- *Complexity impact:* low — well-defined library call.
- *Numerical risk:* low for FP64 ALG2 (deterministic); ALG1 trades determinism for speed — must choose deliberately per prompt.md's reproducibility requirement.
- *Implementation difficulty:* low-medium (mostly memory/stream management, §3.3–3.6 of the architecture phase).
- *CPU/GPU suitability:* GPU — this is the primary validated GPU entry point (§0, §2).
- *Validation required:* CPU vs. GPU SpMV throughput crossover point (nnz, dimensions) on this project's actual RTX 3050 hardware — Level 6 benchmarking, not yet performed.

**KS-7: PDLP-style first-order GPU LP engine (standalone/warm-start use)**
- *Mechanism:* §1.3, §1.3b — PDHG with preconditioning/restarts, SpMV-only inner loop.
- *Computational mechanism:* built on KS-6's SpMV primitive; requires its own presolve/scaling/restart logic.
- *Expected benefit:* fast approximate solves for large, well-scaled standalone LPs; potential warm-start generator for CPU simplex crossover.
- *Complexity impact:* high — a second full numerical method requiring its own validation suite.
- *Numerical risk:* medium-high — moderate accuracy by default, sensitive to conditioning/degeneracy, not proven for B&B-node-repeated-solve regime.
- *Implementation difficulty:* very high.
- *CPU/GPU suitability:* GPU for the iteration loop; CPU for presolve, restart-criterion logic, and any crossover to a basic solution.
- *Validation required:* explicitly NOT recommended for v1 (see §1.3 IMPLEMENTATION DECISION). Would need its own dedicated research phase.

**KS-8: Dense/condensed-system GPU factorization for IPM normal equations**
- *Mechanism:* §1.3b — cuSOLVER/cuBLAS dense Cholesky on the condensed KKT system.
- *Computational mechanism:* forms a dense (or denser-than-original) SPD system after fill-in, factorizes on GPU.
- *Expected benefit:* speedup on the factorization step specifically, when fill-in is large enough to justify dense treatment.
- *Complexity impact:* moderate — requires a condensation/Schur-complement step before the GPU call.
- *Numerical risk:* medium — condensation can worsen conditioning (§1.3b).
- *Implementation difficulty:* high.
- *CPU/GPU suitability:* GPU for the factorization; CPU for IPM step control, line search, condensation logic.
- *Validation required:* only relevant if/when an IPM path is built; measure whether fill-in at refinery-scale instances actually reaches the regime where this pays off.

**KS-9: GPU-batched decomposition subproblems (Benders/Dantzig-Wolfe)**
- *Mechanism:* §1.3b — batch many independent, structurally similar recurring subproblem LPs.
- *Expected benefit:* parallelism across the repeated-LP structure inherent to refinery scheduling decomposition (H4, §6).
- *Complexity impact:* high — requires a decomposition-based master/subproblem architecture that doesn't otherwise exist yet.
- *Numerical risk:* medium — depends entirely on subproblem homogeneity (§1.3b).
- *Implementation difficulty:* very high.
- *CPU/GPU suitability:* GPU for batched subproblem solves if small/homogeneous; CPU for master-problem control.
- *Validation required:* second-phase research prototype per §1.3b's own IMPLEMENTATION DECISION — not a v1 commitment.

**KS-10: Concurrent/racing CPU algorithm selection (dual simplex vs. barrier)**
- *Mechanism:* §1.2 — run multiple LP algorithms concurrently on separate CPU threads, keep the first to finish (Gurobi's documented pattern).
- *Computational mechanism:* thread-level parallelism, no GPU involvement.
- *Expected benefit:* removes the need to perfectly predict which algorithm (simplex vs. barrier) is best for a given model class ahead of time.
- *Complexity impact:* moderate — needs multiple algorithm implementations to race, which is itself a large v1 scope question.
- *Numerical risk:* low (each racer is independently validated).
- *Implementation difficulty:* medium, contingent on already having both a simplex and a barrier implementation.
- *CPU/GPU suitability:* CPU.
- *Validation required:* wall-clock comparison of racing vs. single best-guess algorithm selection across a model-characteristic-diverse test set.

---

## 5. Research Hypotheses

**H1 — GPU-accelerated sparse linear algebra can materially accelerate repeated LP operations for refinery-scale sparse models.**
- *Measurable prediction:* GPU SpMV (cuSPARSE, KS-6) throughput exceeds CPU SpMV throughput above some crossover (nnz, dimension) on the project's RTX 3050 hardware.
- *Benchmark:* the CPU-vs-GPU SpMV benchmark suite specified in prompt.md §3.8, run on synthetic and refinery-structured sparse matrices.
- *Control:* single-threaded and OpenMP-parallel CPU SpMV baseline.
- *Metric:* GFLOP/s and wall-clock time, at matched FP64 accuracy.
- *Success criterion:* a measurable, reproducible crossover point exists where GPU SpMV throughput exceeds CPU throughput at problem sizes representative of the target workload.
- *Failure criterion:* no crossover within representative refinery-scale problem sizes on this hardware (plausible given the 4.29GB VRAM / 14 SM constraint relative to literature's datacenter-GPU benchmarks) — if so, H1 is rejected for this hardware target and GPU acceleration must be reserved for a larger deployment target.
- **EXPERIMENTAL RESULT — H1 is PARTIALLY SUPPORTED, and mostly rejected at this scale.** Three distinct measurements, and the gap between them is the finding.
  - *Inside the simplex: NOT supported.* GPU-accelerated pricing (custom fused kernels, device-resident argmax and Devex weights, `src/cuda/PricingKernels.cu`) is **3.2x slower** than single-threaded CPU pricing and **4.7x slower** than the 16-thread CPU across 41 Netlib instances — faster on 0 of 41. The cause was measured rather than guessed (`docs/architecture/CPU_GPU.md` §4): a simplex iteration needs a host decision, so it pays a device synchronize every iteration, and an SpMV behind a sync costs 33–52 us against 15–26 us queued. Three rounds of kernel work took a 230 us iteration to 225 us. The bottleneck is the algorithm's sequential decision chain, not the kernels, so no kernel work reaches it.
  - *As a synchronization argument: supported.* PDHG needs no host decision per iteration, so a whole restart window is queued and synchronized once — measured at **127 iterations per host synchronize**, against simplex's 1. This is the solid engineering result: the sync floor that defeats GPU simplex is genuinely removable by changing algorithm.
  - *As a wall-clock claim: NOT supported on the Netlib set.* Measured single-process, same pipeline, nothing else running: under 2,500 rows PDLP totals **110.35 s against the simplex's 25.11 s — 4.4x slower** (26 attempted, 20 converged, and 6 never converged within 40 s although the simplex solves every one of them in under 5 s). From 2,500 to 20,000 rows it reverses: **15.60 s against 24.34 s, 1.56x faster**, with `maros-r7` at **0.93 s vs 3.85 s** and `fit2p` at **1.37 s vs 6.88 s**. The one unambiguous win is `dfl001`, this project's standing unsolved instance, which the simplex abandons at its iteration limit after **792 s** and PDLP solves to a verified KKT error of 8.65e-07 in **1.13 s** — a difference in outcome, not in speed. `stocfor3` and `degen3` are ties.
  - *What predicts the winner is PDLP's iteration count* — under ~20,000 it wins, above ~50,000 it loses — which is a property of conditioning after scaling, **not of problem size and not of degeneracy**. `woodw` at 37k nnz loses 10x; `fit2p` at 50k nnz wins 5x. Since the count is unknowable before solving, `LpMethod` is a caller-facing choice, not a hidden heuristic.
  - *The one clean scaling signal:* per-iteration cost is nearly flat across a 20x range of nonzeros — **57 us at 7,777 nnz, 111 us at 144,848** — the signature of latency-bound execution. It is why the small set loses (the device is mostly idle) and it is the strongest available evidence that headroom exists at larger scale. That headroom is an inference from a cost curve, not a measurement; **no model in this repository is large enough to test it**, which leaves H1's actual target — refinery-scale models — unmeasured.
  - **A correction, and the methodology failure behind it.** An earlier version of this entry reported `degen3` at **3.94 s vs 153.83 s (39x)** and `d2q06c` at 8x, and declared H1 supported on that basis. Both numbers were fabricated by contaminated benchmarking — concurrent builds and test suites each spawning 16 OpenMP threads on a 16-core machine, inflating the CPU column asymmetrically while PDLP's GPU work went untouched. Clean: `degen3` is a **tie** (1.44 s vs 1.55 s) and `d2q06c` is **2.8x slower** (12.64 s vs 4.58 s). The mechanism claimed alongside them — that PDLP wins on degeneracy — was also wrong. The same contamination had already been caught once on `stocfor3` (a Devex time of 259.67 s, from which this document inferred a pricing defect that did not exist; the real Devex/Dantzig ratio is 2.16x). Catching it there and not asking what else had been measured under the same conditions is the failure worth recording. See `docs/architecture/PDLP.md` §5.
  - The surviving lesson about H1's wording still holds, in weaker form: H1 asks whether GPU sparse linear algebra can accelerate *repeated LP operations*, and the assumption travelling silently alongside it — that the acceleration would appear inside the simplex — is false. But "H1 supported" was too strong. The defensible claim is that **one algorithm change removes the synchronization barrier completely, and that this buys a wall-clock win only on the larger models, only sometimes, and reliably only where the simplex fails outright.**

**H2 — Numerical scaling plus residual-based iterative refinement can substantially improve robustness on poorly conditioned models.**
- *Measurable prediction:* KS-1's pipeline reduces the incidence of numerical-failure fallback triggers (§ architecture Phase 2.5) on synthetically ill-conditioned test matrices.
- *Benchmark:* condition-number-controlled synthetic matrix suite (Ruiz-before/after κ estimates) plus any refinery-structured instances obtained.
- *Control:* unscaled solve with the same solver core.
- *Metric:* κ(A) reduction factor; solve success/failure rate; residual norms.
- *Success criterion:* scaling+refinement measurably reduces failure rate on the synthetic ill-conditioned suite without degrading well-conditioned-instance performance.
- *Failure criterion:* no measurable robustness improvement, or scaling introduces new feasibility-tolerance artifacts (§1.4.1 limitation) that outweigh the benefit.
- **EXPERIMENTAL RESULT (partial, real-instance evidence, not yet the synthetic κ-controlled suite above):** Ruiz equilibration (`src/lp/Scaling.cpp`) was integrated directly into `Simplex` (scale once at construction, solve entirely in scaled space, unscale x/y/reduced-costs before the NUMERICS.md §6 residual checks). On the Netlib feasible set at row cap 1300 (`benchmarks/validate_netlib`), the unscaled engine reported `maros` (846×1443) as `ITER_LIMIT` after 37.3s / 68,670 iterations with objective 0 (no solution recovered); with scaling enabled, `maros` solves to `OPTIMAL` in 0.68s, matching the published value to a 2.17e-12 relative error. This is a measured, reproducible result on one real instance (not a synthetic suite, and not yet isolated to confirm scaling specifically — vs. e.g. a coincidental interaction — was the deciding factor), so H2 is **supported but not yet fully validated**: the full synthetic condition-number-controlled benchmark this hypothesis specifies has still not been run. Also observed: `finnis`'s already-passing 5.79e-07 relative error was unchanged by scaling (suggesting a different root cause there, not yet diagnosed), and `d6cube`'s solve time increased from 5.7s to 46.9s (still correct, but a real regression worth profiling before claiming scaling is a strict improvement across the board).

**H3 — Symmetry-aware branching can reduce branch-and-bound tree size for refinery scheduling structures containing interchangeable units.**
- *Measurable prediction:* KS-3 (and, if warranted, KS-4) reduces B&B node count on test instances with deliberately constructed interchangeable-unit symmetry, relative to no symmetry handling.
- *Benchmark:* synthetic refinery-scheduling-structured MILP instances with controlled, known symmetry group size (since no verified real MRPL instance/statistic exists yet, per §1.4b).
- *Control:* identical instance, identical solver core, symmetry handling disabled.
- *Metric:* B&B node count, time-to-proven-optimality.
- *Success criterion:* statistically consistent node-count reduction across the synthetic test suite.
- *Failure criterion:* no reduction, or reduction only appears under exact (not approximate/near-identical) symmetry — which would specifically flag the §1.4.3 approximate-symmetry limitation as decisive for real refinery data.

**H4 — A hybrid architecture can outperform CPU-only approaches when sufficient computational reuse exists between host decisions and GPU kernels.**
- *Measurable prediction:* end-to-end solve time for a workload with genuine repeated-LP structure (e.g., a decomposition method, KS-9) is lower with GPU-offloaded subproblem batching than with an all-CPU implementation of the same algorithm.
- *Benchmark:* a constructed decomposition-based refinery-scheduling-style test problem with many recurring subproblems.
- *Control:* identical decomposition algorithm, CPU-only subproblem solves.
- *Metric:* end-to-end wall-clock time; GPU utilization; PCIe transfer overhead as a fraction of total time.
- *Success criterion:* net wall-clock improvement after accounting for transfer overhead.
- *Failure criterion:* transfer/launch overhead dominates any compute benefit — expected to be a real risk given this project's PCIe/memory-architecture constraints (architecture Phase 2.3–2.4).

**H5 — Selective acceleration is superior to attempting to GPU-ify the entire MILP solver.**
- *Measurable prediction:* a solver that GPU-accelerates only KS-6/KS-8/KS-9-class operations while keeping B&B control flow, pivoting, and branching on CPU (per prompt.md's architectural mandate) achieves better wall-clock performance than any attempt to move simplex pivoting or B&B control flow itself to GPU.
- *Benchmark:* comparison is largely qualitative/architectural given that GPU-B&B-control-flow is explicitly excluded by this project's own design principles (not a system that will be built to compare against) — this hypothesis is primarily validated by the literature evidence already gathered (§1.3b: GPU simplex literature) rather than a new head-to-head experiment.
- *Control:* the GPU-simplex literature results themselves (Ploskas & Samaras and related work) serve as the external control.
- *Metric:* qualitative — consistency between this project's own kernel-level benchmarks (H1) and the literature's reported GPU-simplex underperformance.
- *Success criterion:* this project's own measurements (H1, H4) remain consistent with the literature's structural explanation (irregular/sequential operations don't GPU-accelerate well; regular/parallel ones do).
- *Failure criterion:* this project discovers a GPU-simplex or GPU-B&B-control approach that contradicts the literature — if so, this would itself be a notable, separately-verified finding requiring its own dedicated validation before being trusted, not an assumption to build on.
- **EXPERIMENTAL RESULT — H5 is SUPPORTED, and now on this project's own evidence rather than by citation.** This hypothesis anticipated being settled qualitatively, because GPU simplex was not going to be built to compare against. It was built anyway (`src/cuda/GpuPricer.hpp`), for the reason prompt.md §3.8 gives: a performance claim that has not been measured is not a claim. The measurement agrees with the literature and, more usefully, supplies the mechanism the literature only asserts — GPU simplex loses to **per-iteration synchronization latency**, quantified at a 26.7 us floor per iteration on this machine, not to warp divergence or memory irregularity in the abstract. The same measurement then predicted where the GPU *would* win (an algorithm with no per-iteration host decision), which is how the PDLP path came to be built and where H1's positive result came from. Selective acceleration is confirmed as the right strategy; what changed is that the selection is now made on a measured criterion rather than a structural intuition.

---

## 6. Preliminary Conclusions

1. **No existing-solver wrapper is needed or was used** to produce this report — all analysis is either original synthesis or attributed to public literature/documentation, consistent with prompt.md's non-negotiable constraint.
2. **The strongest, best-evidenced GPU opportunity is SpMV-centric** (cuSPARSE primitive, KS-6), not GPU simplex or GPU B&B — this is a LITERATURE EVIDENCE-supported conclusion (§1.3b), reinforced by this project's own architectural mandate that B&B control flow stays CPU-resident.
3. **Every refinery-specific numerical-pathology claim inherited from prompt.md's initial framing (extreme ill-conditioning, severe degeneracy, large symmetry, weak relaxations) remains an ASSUMPTION**, not a verified fact — MRPL's scale (15 MMTPA) is confirmed, but no public condition-number, degeneracy, or symmetry statistic for MRPL or comparable refinery LP/MILP models was located. This project's Phase 3+ benchmarking must generate this evidence directly rather than assume it.
4. **The target hardware (RTX 3050 Laptop, 4.29GB VRAM) is materially smaller than the datacenter GPUs used in every cited GPU-LP benchmark.** Literature speedup figures are evidence that the underlying techniques are real and actively researched — they are not a prediction of what this project will measure on this machine. H1's failure criterion explicitly anticipates the possibility that no beneficial GPU crossover exists at refinery-representative problem sizes on this hardware.
5. **The highest-leverage MILP-specific technique identified (MIR/flow-cover cuts targeting big-M scheduling structure, KS-5, and orbital-branching-style symmetry handling, KS-3/KS-4) are both RESEARCH HYPOTHESES**, not established facts for this domain — the supporting literature establishes general strength on structurally analogous problems (fixed-charge flow, unit commitment), not refinery-specific validation.
6. **Recommended v1 scope, per this research:** an indigenous CPU simplex/branch-and-cut core with Harris-ratio-test/Devex pricing (KS-2), Ruiz scaling + iterative refinement (KS-1), static symmetry-breaking as the first symmetry mitigation (KS-3, before investing in full orbital branching), and GPU acceleration scoped narrowly to cuSPARSE SpMV (KS-6) with measured, reported benchmarks — deferring PDLP-style first-order engines (KS-7), GPU decomposition batching (KS-9), and cuDSS-dependent factorization (until GA) to later research phases, per each section's explicit IMPLEMENTATION DECISION above.
7. **Next step per prompt.md's mandated workflow:** Phase 2 (mathematical and system architecture), building directly on the CPU/GPU division of labor synthesized in §2 above — not implementation.

---

## References

**Solver architecture (HiGHS, SCIP):**
- Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10(1), 119–142.
- ERGO-Code/HiGHS. GitHub repository and documentation. https://github.com/ERGO-Code/HiGHS
- Galabova, I. (2022). *Presolve, crash and software engineering for HiGHS*. PhD thesis, University of Edinburgh.
- Schork, L. (2020). Design and implementation of a modular interior-point solver for linear optimization. arXiv:2006.08814.
- Achterberg, T. (2007). *Constraint Integer Programming*. PhD thesis, TU Berlin.
- Bestuzheva, K., et al. (2021/2023). The SCIP Optimization Suite 8.0. arXiv:2112.08872; *ACM Transactions on Mathematical Software*.
- Achterberg, T., Bixby, R. E., Gu, Z., Rothberg, E., & Weninger, D. (2020). Presolve reductions in mixed integer programming. *INFORMS Journal on Computing*, 32(2), 473–506.
- Achterberg, T., Koch, T., & Martin, A. (2005). Branching rules revisited. *Operations Research Letters*, 33, 42–54.
- Berthold, T. (2006). *Primal Heuristics for Mixed Integer Programs*. Diploma thesis, TU Berlin.
- Bartels, R. H., & Golub, G. H. (1969). The simplex method of linear programming using LU decomposition. *CACM*, 12(5), 266–268.
- Forrest, J. J. H., & Tomlin, J. A. (1972). Updated triangular factors of the basis to maintain sparsity in the product form simplex method. *Mathematical Programming*, 2, 263–278.
- Wunderling, R. (1996). *Paralleler und Objektorientierter Simplex-Algorithmus*. PhD thesis, TU Berlin.
- Bixby, R. E. (2002). Solving real-world linear programs: A decade and more of progress. *Operations Research*, 50(1), 3–15.
- Maros, I. (2003). *Computational Techniques of the Simplex Method*. Kluwer/Springer.
- Pfetsch, M., & Rehn, T. (2019). A computational comparison of symmetry handling methods for mixed integer programs. *Mathematical Programming Computation*, 11, 37–93.
- Ostrowski, J., Linderoth, J., Rossi, F., & Smriglio, S. (2011). Orbital branching. *Mathematical Programming*, 126(1), 147–178.
- SCIP Optimization Suite documentation, symmetry-handling propagator. https://www.scipopt.org/doc/html/SYMMETRY.php

**Commercial SOTA, PDLP, Learn2Branch:**
- Bixby, R. E. (2012). A brief history of linear and mixed-integer programming computation. *Documenta Mathematica*, EMS Press, 107–121.
- Achterberg, T., & Wunderling, R. (2013). Mixed integer programming: Analyzing 12 years of progress. In *Facets of Combinatorial Optimization*, Springer, 449–481.
- Koch, T., et al. (2022). Progress in mathematical programming solvers from 2001 to 2020. arXiv:2206.09787.
- Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual (concurrent optimization, barrier/crossover, cut/heuristic parameters, numerical-issue parameters). docs.gurobi.com.
- Wunderling, R. (2022). Advanced Gurobi Algorithms. Gurobi Days Paris, October 2022.
- IBM. IBM ILOG CPLEX Optimization Studio documentation (barrier optimizer, algorithm selection, crossover).
- Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). Practical large-scale linear programming using primal-dual hybrid gradient. *NeurIPS 2021*. arXiv:2106.04756.
- Lu, H., & Yang, J. (2025). cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid gradient for linear programming in Julia. *Operations Research*, 73(6), 3440–3452 (preprint arXiv:2311.12180).
- Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with graph convolutional neural networks. *NeurIPS 2019*. arXiv:1906.01629.

**GPU sparse linear algebra:**
- NVIDIA. cuSPARSE documentation (generic API, SpMV algorithms). https://docs.nvidia.com/cuda/cusparse/
- NVIDIA. cuSOLVER documentation. https://docs.nvidia.com/cuda/cusolver/index.html
- NVIDIA. cuDSS (Preview) documentation. https://docs.nvidia.com/cuda/archive/13.0.0/cudss/index.html
- NVIDIA Developer Blog. NVIDIA cuDSS advances solver technologies for engineering and scientific computing.
- NVIDIA. LP/QP Features — cuOpt user guide. https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html
- Ploskas, N., & Samaras, N. (2015). Efficient GPU-based implementations of simplex type algorithms. *Applied Mathematics and Computation*.
- Lu, H., Yang, J., et al. cuPDLP-C: A strengthened implementation of cuPDLP for linear programming by C language. arXiv:2312.14832.
- cuPDLPx: A further enhanced GPU-based first-order solver for linear programming. arXiv:2507.14051 (2025).
- An overview of GPU-based first-order methods for linear programming and extensions. arXiv:2506.02174 (2025).
- Google Research Blog. Scaling up linear programming with PDLP.
- Haidar, A., Tomov, S., Dongarra, J., & Higham, N. J. (2018). Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers. *Proc. SC18*.
- Combining sparse approximate factorizations with mixed-precision iterative refinement. *ACM Transactions on Mathematical Software* (2023).
- Pacaud, F., et al. (2024). Accelerating optimal power flow with GPUs: SIMD abstraction of nonlinear programs and condensed-space interior-point methods.
- A GPU-accelerated interior point method for radiation therapy optimization. arXiv:2405.03584.
- Gunnerud, V., Foss, B., & Torgnes, E. (2010). Parallel Dantzig–Wolfe decomposition for real-time optimization — applied to a complex oil field. *Journal of Process Control*, 20(9), 1019–1026.
- Rahmaniani, R., Crainic, T. G., Gendreau, M., & Rei, W. (2017). The Benders decomposition algorithm: A literature review. *European Journal of Operational Research*, 259, 801–817.

**Numerical pathology (ill-conditioning, degeneracy, symmetry, cuts):**
- Ruiz, D. (2001). *A scaling algorithm to equilibrate both rows and columns norms in matrices*. RAL-TR-2001-034, Rutherford Appleton Laboratory.
- Hager, W. W. (1984). Condition estimates. *SIAM Journal on Scientific and Statistical Computing*, 5.
- Markowitz, H. M. (1957). The elimination form of the inverse and its application to linear programming. *Management Science*, 3(3), 255–269.
- Harris, P. M. J. (1973). Pivot selection methods of the Devex LP code. *Mathematical Programming*, 5, 1–28.
- Bland, R. G. (1977). New finite pivoting rules for the simplex method. *Mathematics of Operations Research*, 2(2), 103–107.
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989). A practical anti-cycling procedure for linearly constrained optimization. *Mathematical Programming*, 45, 437–474.
- Goldfarb, D., & Reid, J. K. (1977). A practicable steepest-edge simplex algorithm. *Mathematical Programming*, 12, 361–371.
- Forrest, J. J., & Goldfarb, D. (1992). Steepest-edge simplex algorithms for linear programming. *Mathematical Programming*, 57, 341–374.
- Kaibel, V., Peinhardt, M., & Pfetsch, M. E. (2007). Orbitopal fixing. IPCO 2007, LNCS 4513, 74–88.
- Gomory, R. E. (1958). Outline of an algorithm for integer solutions to linear programs. *Bulletin of the AMS*, 64, 275–278.
- Nemhauser, G. L., & Wolsey, L. A. (1990). A recursive procedure to generate all cuts for 0–1 mixed integer programs. *Mathematical Programming*, 46, 379–390.
- Marchand, H., & Wolsey, L. A. (2001). Aggregation and mixed integer rounding to solve MIPs. *Operations Research*, 49(3), 363–371.
- Crowder, H., Johnson, E. L., & Padberg, M. (1983). Solving large-scale zero-one linear programming problems. *Operations Research*, 31(5), 803–834.
- Gu, Z., Nemhauser, G. L., & Savelsbergh, M. W. P. (1998). Lifted cover inequalities for 0-1 integer programs: Computation. *INFORMS Journal on Computing*, 10, 427–437.
- Padberg, M. (1973). On the facial structure of set packing polyhedra. *Mathematical Programming*, 5, 199–215.
- Padberg, M., Van Roy, T. J., & Wolsey, L. A. (1985). Valid linear inequalities for fixed charge problems. *Operations Research*, 33, 842–861.
- Van Roy, T. J., & Wolsey, L. A. (1986). Valid inequalities for mixed 0-1 programs. *Discrete Applied Mathematics*, 14, 199–213.
- Lee, H., Pinto, J. M., Grossmann, I. E., & Park, S. (1996). Mixed-integer linear programming model for refinery short-term scheduling of crude oil unloading with inventory management. *Ind. Eng. Chem. Res.*, 35, 1630–1641.
- Wicaksono, D. S., & Karimi, I. A. (2008). Piecewise MILP under- and overestimators for global optimization of bilinear programs. *AIChE Journal*, 54(4), 991–1008.
- MRPL corporate profile, mrpl.co.in; Business Standard (2023), "MRPL in coastal Karnataka becomes India's largest single location refinery."
