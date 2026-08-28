# Memory Architecture

**Status:** PHASE 2 architecture. This document defines the complete host/device memory model per prompt.md §2.3, and is the direct basis for Phase 3's `MemoryArena` implementation (§3.1 of prompt.md).

---

## 1. VRAM Budget (measured, not assumed)

`nvidia-smi` on the target device reports 4094 MiB total VRAM, with 808 MiB–1.7 GiB already resident from the OS desktop compositor and other GPU clients depending on which environment (native Windows vs. WSL2) is running at solve time. **IMPLEMENTATION DECISION:** the solver queries free VRAM at `SolveContext` initialization (`cudaMemGetInfo`) and sizes its device arena (§3) against a conservative fraction of *currently free* memory, not the nameplate 4094 MiB — treating the difference as a hard external constraint, not a bug to work around. If the presolved, scaled problem's persistent device footprint (§3.2) would exceed this budget, the solver falls back to CPU-only execution for that solve rather than attempting a GPU allocation that risks an out-of-memory failure mid-solve (a failure mode explicitly worse than not using the GPU at all, given prompt.md §2.5's reliability mandate).

## 2. Memory Categories

| Category | Used for | Justification |
|---|---|---|
| Pageable host memory | `RawModel`, `PresolvedModel`, one-time/low-reuse data | No PCIe-crossing hot path touches this data; pinning would be pure overhead |
| Pinned host memory | Staging buffers for CSR upload, per-call vector transfers | Enables true async DMA (`CPU_GPU.md` §3.3); applied only to data that crosses PCIe repeatedly |
| Unified memory | **Not used** | Rejected — see `CPU_GPU.md` §3.5 (determinism + explicit VRAM budgeting on a small, shared GPU) |
| Zero-copy | **Not used** for persistent solve data | Rejected — see `CPU_GPU.md` §3.4 (harmful for repeated-access hot-path data) |
| GPU VRAM | Device-resident CSR matrix, dense vector buffers, cuSPARSE workspace | Persistent for solve lifetime — §3.2 |
| Host RAM | B&B tree (node bound-change records), cut pool, incumbent, presolve mapping | CPU-resident by architectural mandate (`CPU_GPU.md` §1) |
| Persistent allocations | Sized once at `SolveContext` init, live for the whole solve | Matches prompt.md §3.1's zero-allocation-during-solve requirement |
| Temporary workspaces | cuSPARSE `SpMV` external buffer, per-node scratch | Sized once per shape-class, reused — never freed/reallocated per call |
| Arena allocation | Host-side `MemoryArena` (Phase 3, `src/memory/MemoryArena.hpp`) | Monotonic, reset-capable, zero-fragmentation bump allocator |
| Memory pools | Not needed in v1 | A single arena with two lifetime tiers (§3.3) suffices at current scope; a pool (multiple size classes, independent reuse) is not justified until a concrete allocation-pattern need is measured |

## 3. Ownership, Lifetime, and Residency

### 3.1 Two-tier host arena model

Two lifetime tiers, both backed by `MemoryArena` (Phase 3):

1. **Solve-lifetime arena.** Sized once, immediately after presolve+scaling fix the final problem dimensions (`SYSTEM.md` §2.3–2.4). Holds: the CSR/CSC host buffers, scale factors, presolve mapping, and any solve-global scratch. Lives for the entire `SolveContext::solve()` call; reset only when the context is destroyed or reused for a new, unrelated model.
2. **Node-scratch arena.** Reset (not freed — a bump-pointer reset, $O(1)$) after each B&B node finishes processing. Holds: node-local bound-change deltas, the node's LP working vectors, and any temporary buffers a single node's relaxation solve needs. This bounds memory growth across a B&B tree with tens of thousands of nodes — without a per-node reset, node-scratch would otherwise accumulate for the life of the solve.

This two-tier split is the direct answer to prompt.md §3.1's "no fragmentation during solve" and "reset capability" requirements: tier 1 never resets mid-solve (nothing depends on its contents changing), tier 2 resets on a predictable, high-frequency schedule (once per node) with no fragmentation because it is a monotonic bump allocator, not a general-purpose heap.

### 3.2 Device-resident persistent allocations

Allocated once, during `SolveContext` initialization, sized from the *presolved and scaled* problem (never the raw input — presolve/scaling may shrink the problem materially, and sizing device buffers from the raw model would waste VRAM on a device where every MB counts):

- CSR matrix (values: `nnz × 8B` FP64, column indices: `nnz × 4B`, row offsets: `(m+1) × 4B`) and, if the LP engine's node-local sub-problems need it, the CSC transpose.
- Dense vector buffers for $x$, $y$, and residual scratch, each $O(n)$ or $O(m)$ — small relative to the matrix for any realistically sparse model.
- cuSPARSE `SpMV` external workspace, sized via one `cusparseSpMV_bufferSize` query at initialization, never re-queried per call.

**Ownership:** each of the above is wrapped in an RAII `CudaBuffer<T>` (Phase 3, `src/cuda/`), whose destructor calls `cudaFree` deterministically — there is no scenario in this architecture where a device allocation's lifetime is implicit or where cleanup is deferred to program exit.

### 3.3 Explicit boundaries: ownership, lifetime, synchronization, transfer, residency

For every buffer that crosses the host/device boundary, this architecture requires four properties to be stated explicitly (not left implicit), matching prompt.md §2.3:

- **Ownership:** exactly one object (a `CudaBuffer<T>` or the host `MemoryArena`) owns each allocation; no buffer is ever owned jointly or "borrowed" without a non-owning view type making that explicit.
- **Lifetime:** solve-lifetime buffers (§3.2) vs. node-scratch buffers (§3.1 tier 2) vs. per-call transient staging (a pinned buffer reused across calls, but whose *contents* are transient) are three distinct lifetime classes, and no code path is permitted to conflate them (e.g., writing node-scratch data into a solve-lifetime buffer).
- **Synchronization:** a device buffer written by an async kernel/copy is not read by the CPU until the corresponding `CudaEvent` has been waited on — this is enforced at the `CudaBuffer`/stream API boundary (Phase 3), not left to caller discipline alone.
- **Transfer boundaries:** the *only* sanctioned crossing points are those enumerated in `CPU_GPU.md` §3.2 (matrix upload once; vector H2D/D2H per SpMV call; scalar D2H per residual check). Any new cross-boundary transfer introduced during Phase 3 implementation must be justified against this table, not added ad hoc.
- **Data residency:** at any point in the solve, it is always answerable, for any given piece of solver state, whether the authoritative copy is host- or device-resident — there is never a "currently being synchronized, ask again later" ambiguous state exposed to calling code (that ambiguity is confined to the wait-on-event boundary, which is internal to the buffer/stream abstraction).
