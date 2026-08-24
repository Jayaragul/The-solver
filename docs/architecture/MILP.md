# MILP Engine Architecture (Branch-and-Bound, Cuts, Symmetry)

**Status:** PHASE 2 architecture. Per prompt.md §2.8, branch-and-bound control flow is 100% CPU-resident — this is a hard constraint, not a default that could later move to GPU. Per §2.9, symmetry-handling machinery (specifically orbital branching) is evaluated against actual refinery scheduling structure rather than implemented because it was named in the original brief.

---

## 1. Branch-and-Bound Core

### 1.1 Node representation

A node is represented **relative to its parent** — a set of bound-change deltas (tightened $l$/$u$ on one or more variables) plus a reference to the parent's basis (for `LP.md`'s warm-start path) — not a full copy of the LP. This is the standard technique that makes large B&B trees memory-tractable (`MEMORY.md` §3.1 tier 2: node-scratch is reset per node, but the *bound-change record* that defines the node's identity is small and kept in the solve-lifetime tier as part of the tree structure).

```cpp
struct BnBNode {
    NodeId id, parent_id;
    std::vector<BoundChange> deltas;  // relative to parent, not absolute model state
    double parent_bound;              // inherited LP relaxation bound, for pruning before re-solving
    int depth;
};
```

### 1.2 Node queue and selection

**v1 policy: best-bound selection with limited plunging (dive-then-backtrack on the current best-bound branch).** This is an explicit, simpler alternative to SCIP's full pseudocost-estimate node ranking (SOTA.md §1.1) — **IMPLEMENTATION DECISION**, justified by two Phase 1 findings: (a) pseudocost-based estimates are themselves built on potentially unreliable signal on degenerate refinery LPs (SOTA.md §1.1's hypothesis about pseudocost reliability under degeneracy), so a simpler ranking avoids inheriting that unvalidated assumption into the core search order; (b) best-bound is the simplest policy with a direct, provable relationship to the reported optimality gap. Upgrading to estimate-based ranking is a candidate future milestone, contingent on v1 benchmarking showing best-bound's tree sizes are a practical bottleneck.

### 1.3 Branching

**v1 policy: reliability branching** (strong branching for variables whose pseudocost history is not yet reliable, pseudocost branching otherwise) — SOTA.md §1.2 documents this as the de facto default across modern commercial and open-source solvers, making it the best-evidenced starting point rather than a novel choice.

- Strong branching evaluates a small set of fractional candidates by actually solving (a bounded number of iterations of) both child LPs before committing — expensive per node, cheap in node count.
- Pseudocost branching uses historical objective-degradation-per-unit-fractionality statistics, cheap per node, unreliable until enough history accumulates.
- Reliability branching runs strong branching only until a variable's pseudocost history is deemed reliable (a small fixed count of prior observations), then switches to pseudocost — the standard middle ground.

**Symmetry interaction (see §3):** on instances with interchangeable units/periods, multiple branching candidates may be structurally equivalent; v1 does not attempt symmetry-aware branching-candidate deduplication (a PROPOSED MODIFICATION noted in SOTA.md §1.1 but not validated) — this is deferred pending evidence from §3's simpler static symmetry-breaking that dynamic candidate deduplication is even needed.

### 1.4 Node presolve

Each node re-applies a restricted subset of the presolve rules (`SYSTEM.md` §2.3) — specifically bound propagation given the node's accumulated `deltas` — to tighten bounds locally before the LP re-solve. Node presolve reuses the *same* reduction-rule implementations as global presolve (no duplicated logic), applied to a node-local view rather than the full model.

## 2. Cuts (v1 scope: architectural stub)

Per SOTA.md §5 (KS-5), MIR and flow-cover cuts targeting the big-M fixed-charge structure typical of refinery scheduling formulations are the highest-*hypothesized*-leverage cut family for this workload — but this is an explicit RESEARCH HYPOTHESIS, not yet validated on any instance. Per prompt.md's development rule ("do not move to the next subsystem until the current one is internally consistent"), **v1 ships branch-and-bound without active cut generation** — the `CutManager` module (`SYSTEM.md` §2.9) exists as an architectural seam (a defined interface a future milestone plugs into) but performs no separation in the first working milestone. Adding MIR/flow-cover separation is the first planned extension *after* the uncut B&B core passes Level 6 benchmarking — this ordering is itself a test of whether cuts are worth their separation overhead on this specific workload, not an assumption that they will be.

```cpp
class CutManager {
public:
    // v1: returns empty — no cuts generated. Interface exists for the
    // planned MIR/flow-cover milestone (SOTA.md KS-5), not implemented yet.
    std::vector<Cut> separate(const LPResult& fractional_solution) { return {}; }
};
```

## 3. Symmetry

Per prompt.md §2.9's explicit instruction — **do not implement orbital branching simply because it was requested; evaluate whether it is appropriate for each identified refinery scheduling structure** — the evaluation, drawing on SOTA.md §1.4.3:

- **Exact automorphism-based methods (orbital branching, orbitopal fixing) are well-supported in the literature for *exact* symmetry** (Ostrowski et al.'s unit-commitment application is structurally analogous to refinery scheduling with parallel identical units).
- **But real refinery units are rarely exactly identical** — slightly different capacities, ages, or fouling factors break the exact automorphism these methods detect, per SOTA.md §1.4.3's explicitly stated limitation. Building a full graph-automorphism pipeline (nauty/bliss/saucy-equivalent, from scratch, per the no-existing-solver-library constraint) is a large implementation effort (KS-4's "implementation difficulty: high" rating) whose payoff is contingent on symmetry in the *actual* model being exact enough to matter — unverified.

**IMPLEMENTATION DECISION for v1: static symmetry-breaking constraints only** (SOTA.md KS-3), applied when the modeling layer explicitly declares a set of variables as interchangeable (e.g., "these $k$ parallel crude distillation trains are structurally identical in this instance") — not via automatic exact-automorphism detection. A declared interchangeable group $\{x_1, \dots, x_k\}$ gets a cheap a priori lexicographic ordering constraint ($x_1 \ge x_2 \ge \dots \ge x_k$ in the relevant sense for the group's role) added at model-construction time, with no B&B-time automorphism computation at all.

**Why this ordering of decisions is itself the correct engineering choice, not just a cost-cutting shortcut:** static symmetry-breaking is strictly cheaper to build, and if it fails to meaningfully reduce node counts on representative test instances, that failure is informative — it would suggest either (a) the symmetry in real refinery models is too approximate for *any* symmetry-handling approach to exploit cleanly (in which case orbital branching's automorphism-detection step would find little or nothing anyway), or (b) the declared interchangeable groups don't match where the actual combinatorial redundancy lives. Either finding sharpens whether orbital branching (§KS-4) is worth its implementation cost — this is exactly the "hypotheses that must be experimentally validated" discipline prompt.md §1 requires, applied to a Phase 2 architectural choice rather than deferred to a vague "future work" note.

```cpp
struct SymmetryGroup { std::vector<VarId> members; SymmetryRole role; };

// v1: caller-declared groups only. No automorphism detection.
std::vector<Constraint> break_symmetry(const std::vector<SymmetryGroup>& declared_groups);
```

## 4. Primal Heuristics and Incumbent Management

v1 includes a single trivial rounding heuristic (round the fractional LP solution to the nearest bound-respecting integer point, accept if feasible) — RENS/feasibility-pump/diving-class heuristics (SOTA.md §1.1) are deferred, consistent with the cuts decision in §2: validate the core search first, add heuristics once there is a baseline to measure their marginal benefit against. Incumbent management is a single global best-solution record (`SYSTEM.md` §2.11), single-writer, no concurrency primitives — v1's B&B is single-threaded; parallel tree search is out of scope until the sequential core is benchmarked.

## 5. GPU Involvement Inside a Node (restated boundary)

The only GPU-resident work anywhere in this engine is the SpMV call inside a node's LP relaxation solve (`LP.md`, `CPU_GPU.md` §2.1) and the residual verification that follows it. Node creation, selection, branching, cut management (even once §2's stub becomes real), incumbent updates, and node presolve are all CPU-resident, always. This is restated here, in the MILP document itself, because it is the constraint most likely to be violated by well-intentioned future "optimization" — any change that moves tree-control logic to GPU is an architecture violation, not a performance tuning decision, and must be rejected regardless of a claimed speedup.
