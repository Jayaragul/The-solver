# Sovereignty boundary

SANKHYA owns the optimization algorithms, sparse data structures, factorization
logic, presolve, tree search, heuristics, and verification formats in its solve
path. The production boundary is C-compatible: the native core is implemented in
C, with C++/CUDA permitted for specialized high-performance internals. No
Python component is part of this repository's solver or benchmark path.

Established solvers may run only as isolated benchmark processes. Their packages
must never be imported or linked by SANKHYA.

Benchmark instances, tolerances, verifiers, train/held-out splits, and timing code
are protected from automated tuning. Results must identify compiler/runtime,
hardware, solver versions, dataset hashes, limits, and failed instances.
