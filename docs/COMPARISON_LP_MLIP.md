# SANKHYA versus LP-and-MLIP-Solver

This comparison was made against the `main` revision of
[Jayasuryamahadevan/LP-and-MLIP-Solver](https://github.com/Jayasuryamahadevan/LP-and-MLIP-Solver)
on 2026-08-31. The comparison repository calls its engine **SIHPS**. Its
source was inspected locally; no solver library was copied or linked.

## What is different

| Area | SANKHYA (`c/` + `cuda/`) | SIHPS comparison repository | Assessment |
|---|---|---|---|
| Core language | C11 ABI plus CUDA C kernels | C++17 plus CUDA C++ | SANKHYA is easier to embed from C; SIHPS is easier to extend with value types and RAII. |
| LP | Bounded revised simplex, sparse LU, scaling, certificate repair, CUDA PDHG | Primal and dual revised simplex, Devex, sparse LU, warm starts, GPU PDLP/pricing | SIHPS has the broader LP control layer; both retain CPU control for pivot decisions. |
| Presolve | Conservative interval and singleton reductions | Adds integer-bound rounding, scoped doubleton substitution, and GCD tightening | Integer-bound rounding and a conservative equality-row GCD infeasibility gate are now enabled in our MILP nodes. Doubleton substitution remains staged because reversible postsolve correctness is harder. |
| MILP | Branch-and-bound, best-bound queue, pseudocost/reliability branching, cover cuts, rounding/diving/local improvement | Adds the same foundations plus optional GMI/RENS/GCD paths, exact binary meet-in-the-middle for one structural class, and a parallel tree | SIHPS has stronger breadth. Its exact binary path is valuable only when its structural gate proves applicable; it is not a general MILP replacement. |
| QP / MIQP | Native CPU QP (closed-form, KKT, PDHG), CUDA sparse/diagonal QP, guarded small MIQP | LP/MILP-focused; its documented v1 status defers QP | SANKHYA is clearly broader for the requested QP scope. |
| GPU | CUDA SpMV, LP/QP PDHG, projected KKT checks | CUDA SpMV, pricing, and device-resident PDLP | Comparable architecture; SANKHYA’s QP path and KKT admission checks are an advantage for this goal. |
| Verification | Public model validation plus original-space primal/dual/KKT and integrality checks; non-certified results are downgraded | Extensive original-space checks and reproducibility metadata | Both follow the right invariant. SANKHYA’s C API exposes validation before solving. |
| Benchmark telemetry | MIPLIB runner records wall time, CPU time, memory, and CUDA availability; Windows CPU time now uses `GetProcessTimes` | Reproducibility metadata and solver counters | Resource comparisons are now meaningful on the native MSVC/CUDA host. |
| Portability | CUDA is optional for the C core; native CPU build remains available | C++ core is configured around CUDA in the comparison build | SANKHYA is more usable on CPU-only hosts. |

## Adoption decision

The comparison repository has a better *research architecture* for the next
MILP phase: integer-aware reductions, exact structural subsolvers, and a
parallel node layer. We are adopting the sound, local reductions first: every
integer node bound is rounded inward before its LP relaxation, and exact
modularly impossible equality rows are rejected before solving. Neither needs
a postsolve mapping; both options can be disabled for controlled ablations.

We are not copying the comparison repository's full presolve or parallel tree
blindly. Doubleton substitution needs a complete reversible postsolve record
and dedicated differential tests; exact binary enumeration
needs a structural proof gate and a memory budget; parallel B&B needs bounded
live-basis ownership and deterministic verification under races. Those are the
next staged integrations, each enabled only after a measured correctness and
performance comparison.

## Bottom line

SIHPS is the better reference for scalable LP/MILP architecture. SANKHYA is
currently the better fit for the stated sovereign LP+MILP+QP deliverable because
it combines that native sparse LP/MILP direction with an independently guarded
QP/CUDA implementation and a C-compatible API. The two approaches are
complementary, not competing black boxes: we take the safe algorithmic ideas,
retain our verification gate, and do not claim a benchmark improvement until a
same-machine run demonstrates it.
