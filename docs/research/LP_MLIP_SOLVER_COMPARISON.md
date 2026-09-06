# Comparison with `LP-and-MLIP-Solver`

Reference inspected: [Jayasuryamahadevan/LP-and-MLIP-Solver](https://github.com/Jayasuryamahadevan/LP-and-MLIP-Solver),
commit `f039e2b` (cloned on 2026-09-06). The comparison is source-level and
does not treat README claims as benchmark evidence.

## What the reference does better

1. **Tableau GMI cuts.** The reference has a guarded root Gomory mixed-integer
   separator. This repository now includes the same approach behind
   `enable_root_gmi_cuts`, with a direct regression on a fractional integer row.
   A narrower pure-integer coefficient-floor experiment was also tested and
   rejected by the MIPLIB gate; it remains opt-in for research only.
2. **Structure-specific exact search.** `ExactBinarySplit` recognizes a narrow
   binary-plus-unit-slack equality family and uses complete meet-in-the-middle
   enumeration. That is a legitimate answer for `markshare2`-class models,
   where the LP dual bound can remain zero. It is intentionally not a general
   MILP algorithm and is not yet enabled here because its memory/exponential
   gates need to be adapted to this laptop's available RAM.
3. **Parallel tree search and RENS.** The reference contains experimental
   multi-worker B&B and a root RENS heuristic. These are useful follow-up
   candidates, but they require independent race, memory, and benchmark gates.

## What this repository currently does better or more conservatively

- The production default is measured **pseudocost** branching (9/19 exact in
  the frozen 10-second sweep versus 6/19 reliability), rather than relying on
  the reference's reliability default.
- Warm-started node relaxations and integer lattice/equality reductions are
  enabled only after local correctness and timing evidence; every result still
  passes the original-space feasibility/integrality gate.
- CUDA is native C++/CUDA, with explicit GPU/CPU differential and reproducible
  tests. The reference's broad feature set is not automatically evidence of a
  wall-clock win on this machine.

## Decision

The best ideas were adopted selectively: guarded GMI and root RENS are
implemented as opt-in experiments, while exact binary split and parallel B&B
remain separate measured candidates. Both current gates are negative at 10
seconds, so neither is enabled by default; see
[`MIPLIB_GMI_ABLATION_10S.md`](../results/MIPLIB_GMI_ABLATION_10S.md) and
[`MIPLIB_RENS_ABLATION_10S.md`](../results/MIPLIB_RENS_ABLATION_10S.md), plus
[`MIPLIB_COEFFICIENT_ROUNDING_ABLATION_10S.md`](../results/MIPLIB_COEFFICIENT_ROUNDING_ABLATION_10S.md).
This preserves the repository's rule that a new lever must improve a declared
benchmark KPI before becoming production behavior.
