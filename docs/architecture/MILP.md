# MILP Engine Architecture (Branch-and-Bound, Cuts, Symmetry)

**Status:** PHASE 3 implementation. The first working MILP engine is a
CPU-resident, deterministic branch-and-bound solver over certified simplex
relaxations. Cuts, warm starts, and advanced branching are explicit follow-up
milestones; they are not silently represented as implemented features.

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

**Implemented v1 policy: most-fractional branching.** At each fractional LP
solution, the integer variable with the largest `min(fractional_part,
1-fractional_part)` score is selected, with the original column index as the
deterministic tie-break. This is deliberately a simple, auditable baseline;
it is not mislabeled as reliability branching.

Reliability branching remains the next branching milestone. Strong branching
and pseudocost updates must be added together with their own accounting and
regression benchmark before the policy is changed, because an unmeasured
candidate-evaluation budget can make the tree smaller while making total solve
time worse.

- Strong branching evaluates a small set of fractional candidates by actually solving (a bounded number of iterations of) both child LPs before committing — expensive per node, cheap in node count.
- Pseudocost branching uses historical objective-degradation-per-unit-fractionality statistics, cheap per node, unreliable until enough history accumulates.
- Reliability branching runs strong branching only until a variable's pseudocost history is deemed reliable (a small fixed count of prior observations), then switches to pseudocost — the standard middle ground.

**Symmetry interaction (see §3):** on instances with interchangeable units/periods, multiple branching candidates may be structurally equivalent; v1 does not attempt symmetry-aware branching-candidate deduplication (a PROPOSED MODIFICATION noted in SOTA.md §1.1 but not validated) — this is deferred pending evidence from §3's simpler static symmetry-breaking that dynamic candidate deduplication is even needed.

### 1.4 Node presolve

Each node materializes its accumulated bound deltas into a reusable LP
workspace and re-applies the existing presolve implementation before the LP
re-solve. The sparse matrix is copied once for the solve and then reused;
node creation does not copy the matrix. This keeps node state proportional to
the bound-change chain rather than to the full model.

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

The implementation includes a safe rounding heuristic: integer variables are
rounded and the resulting point is accepted only after a fresh original-model
feasibility and integrality check. A failed heuristic never changes the
search state. RENS, feasibility pump, and diving heuristics remain deferred.
Incumbent management is a single global best-solution record, single-writer,
with no concurrency primitives; the B&B control loop is single-threaded.

### 4.1 Correctness and termination contract

- Every node bound comes from the certified CPU simplex path. HYBRID and
  FIRST_ORDER preferences are overridden for relaxations because an
  approximate point is not a safe proof bound.
- A node is pruned only by an infeasible relaxation or by a lower bound that
  cannot improve the incumbent within the configured objective tolerance.
- An incumbent is stored only after checking original row bounds, variable
  bounds, and exact integer/binary membership after rounding.
- `OPTIMAL` means the open-node queue was exhausted. `NODE_LIMIT` and
  `TIME_LIMIT` are never relabeled as optimal merely because an incumbent
  exists. An unbounded LP relaxation with integer variables is reported as
  `UNBOUNDED_RELAXATION`, not as a proof that the MILP itself is unbounded.
- MPS `INTORG`/`INTEND`, `LI`, `UI`, `BV`, and `OBJSENSE MAX` metadata are
  preserved by the parser and converted into the MILP model contract.

## 5. GPU Involvement Inside a Node (restated boundary)

The only GPU-resident work anywhere in this engine is the SpMV call inside a node's LP relaxation solve (`LP.md`, `CPU_GPU.md` §2.1) and the residual verification that follows it. Node creation, selection, branching, cut management (even once §2's stub becomes real), incumbent updates, and node presolve are all CPU-resident, always. This is restated here, in the MILP document itself, because it is the constraint most likely to be violated by well-intentioned future "optimization" — any change that moves tree-control logic to GPU is an architecture violation, not a performance tuning decision, and must be rejected regardless of a claimed speedup.
