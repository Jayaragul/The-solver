#pragma once

#include <cstdint>

// Central OpenMP policy for this engine (prompt.md \S3.2: "OpenMP
// compatibility where justified").
//
// DETERMINISM COMES FIRST
// -----------------------
// docs/architecture/NUMERICS.md \S1 requires run-to-run bit-reproducible
// results. That rules out the obvious uses of OpenMP and permits a narrow
// one:
//
//   ALLOWED     -- loops where each OUTPUT element is produced by exactly
//                  one iteration, in the same summation order the serial
//                  loop would use. A row of an SpMV, a column's reduced
//                  cost, an entry of the pivot row. Thread count then
//                  changes only WHO does the arithmetic, never the
//                  arithmetic, so the result is bit-identical to serial
//                  and to any other thread count.
//
//   FORBIDDEN   -- `reduction(+:)` over floating point, and any scatter-add
//                  (a CSC-ordered SpMV, for instance). Both make the
//                  summation order depend on scheduling, so the last bits
//                  of the answer become a function of how many cores were
//                  idle. That is precisely the class of irreproducibility
//                  the CSR_ALG2 choice in GpuSpMV.hpp already rejects on
//                  the GPU; accepting it on the CPU would be incoherent.
//
// Every parallel loop in this codebase is of the first kind, and uses
// schedule(static) so even the assignment of indices to threads is fixed.
//
// SIZE GATING IS NOT AN OPTIMIZATION, IT IS THE WHOLE POINT
// ---------------------------------------------------------
// Entering an OpenMP parallel region costs a fork and a barrier -- on the
// order of a few microseconds. The simplex prices once per iteration, and
// on a small model that entire pricing pass takes less time than the
// barrier would. Parallelizing unconditionally therefore makes small
// models SLOWER, which is the same trap the GPU pricing path falls into
// for the same reason (docs/architecture/CPU_GPU.md \S4). Every pragma
// below is gated on a work estimate, and the threshold is a measured
// number rather than a guess -- benchmarks/bench_parallel.cpp.
//
// When the compiler has no OpenMP support the pragmas vanish and every
// loop runs serially, producing identical results. That is a property of
// the design above, not a coincidence: nothing here is only correct in
// parallel.

namespace sihps {

// Minimum number of nonzeros a matrix-shaped pass must touch before it is
// worth handing to a thread team. Below this, the fork/barrier costs more
// than the work saved.
//
// MEASURED on the 16-thread development machine
// (benchmarks/bench_parallel.cpp; full table in
// docs/architecture/CPU_GPU.md \S5). The sweep shows three regimes rather
// than a clean crossover: below ~2k nonzeros threading is a clear loss
// (0.06x at 488 nnz), between ~4k and ~10k the result is inconsistent run
// to run (1.32x, then 0.65x), and from ~30k it is a reliable 4-5x. The
// threshold sits above the inconsistent band, NOT at the first point a
// speedup appears -- an unreliable win on a pass the simplex executes
// thousands of times per solve is not worth taking.
constexpr std::int32_t kParallelNnzThreshold = 20000;

} // namespace sihps

// Emits an OpenMP pragma when the compiler supports it and nothing at all
// when it does not, so a serial build stays warning-free rather than
// tripping -Wunknown-pragmas. Usage:
//
//   SIHPS_OMP(omp parallel for schedule(static) if(nnz >= kParallelNnzThreshold))
//   for (...) { ... }
//
// The argument must contain no top-level comma (none of the clauses used
// here do).
#if defined(_OPENMP)
#define SIHPS_OMP(directive) _Pragma(#directive)
#else
#define SIHPS_OMP(directive)
#endif
