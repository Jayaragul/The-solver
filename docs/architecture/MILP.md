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

**Implemented policy: best-bound selection.** Each queued node carries a
valid inherited lower bound before its own relaxation is solved, and ties are
resolved deterministically by depth and creation order. The inherited bound
is conservative for gap reporting; it never permits an unsafe prune.

### 1.3 Branching

**Implemented policy: reliability branching.** The solver starts with the
most-fractional candidates, performs a bounded number of strong-branching LP
probes for unreliable variables, and records objective degradation per unit
branch distance as separate up/down pseudocosts. Once both directions reach
the configured reliability threshold, pseudocost scores rank candidates. The
column index is the deterministic final tie-break. `MOST_FRACTIONAL` and
`PSEUDOCOST` remain available for controlled comparisons.

Strong-branching probes are counted separately from processed B&B nodes and
are never used as proofs: an unsuccessful probe is ignored for ranking, while
the actual child relaxation still determines pruning and termination.

**Symmetry interaction (see §3):** on instances with interchangeable units/periods, multiple branching candidates may be structurally equivalent; v1 does not attempt symmetry-aware branching-candidate deduplication (a PROPOSED MODIFICATION noted in SOTA.md §1.1 but not validated) — this is deferred pending evidence from §3's simpler static symmetry-breaking that dynamic candidate deduplication is even needed.

### 1.4 Node presolve

Each node materializes its accumulated bound deltas into a reusable LP
workspace and re-applies the existing presolve implementation before the LP
re-solve. The sparse matrix is copied once for the solve and then reused;
node creation does not copy the matrix. This keeps node state proportional to
the bound-change chain rather than to the full model.

**Exception, off by default:** `MilpSolverOptions::warm_start_node_relaxations`
(`docs/architecture/LP.md` §8) skips this presolve step for non-root nodes
entirely, constructing `Simplex` directly and seating the parent's exported
basis instead. A warm basis is only valid over the *same* augmented column
space it was computed for, which node-level presolve is not guaranteed to
preserve under a tightened bound — so this path trades node presolve away
rather than trying to reconcile the two. `MEASURED` to be a net loss on the
current MIPLIB benchmark for exactly that reason (§4.1's certified-bound
invariant is unaffected either way — the traded-away presolve reductions,
not correctness, are what the measurement blames).

## 2. Cuts (implemented root mixed-row cover subset)

The first cut implementation is deliberately narrow and auditable: root-only
cover inequalities for rows with a finite upper activity bound and finite
term-wise lower contributions. Binary terms—including MPS `INTEGER`
variables explicitly bounded in $[0,1]$—are selected for a cover; all other
terms are accounted for by their exact bound-derived minimum contribution.
For a cover $C$ with $\sum_{j\in C} a_j > b'$, where $b'$ is the row upper
bound after subtracting those minimum contributions, it adds the valid
inequality $\sum_{j\in C} x_j \le |C|-1$. Rows with unbounded lower
contributions are rejected by the separator rather than approximated. Cuts
are globally valid and remain active in descendants; they are generated once
at the root and counted separately.

General MIR and flow-cover strengthening for mixed rows remains a future
extension. It requires a full row-bound transformation and independent
validity tests before it can be enabled, so the current implementation does
not claim that broader cut family.

```cpp
class CutManager {
public:
    // Current implementation: root-only nonnegative mixed-row cover
    // separation; binary-domain terms enter each cover inequality.
    // General MIR/flow-cover strengthening is intentionally not implied here.
    std::vector<Cut> separate(const LPResult& fractional_solution);
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
