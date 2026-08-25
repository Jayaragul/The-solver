# CUDA queued-kernel benchmark

Date: 2026-08-25

This focused benchmark measures the CUDA PDHG device-pointer path before and
after removing host-side synchronization after every SpMV, transpose-SpMV, and
AXPY launch.  All kernels remain ordered on CUDA's default stream; the existing
device-to-host checks still provide completion and error visibility.

## Protocol

```text
build\cmake-cuda\cuda\sankhya_pdhg.exe bench\data\netlib\mps\afiro.mps
```

Each build was run three times on the same RTX 3050 Laptop GPU.  The reported
runtime is the median of the three solver-reported `solve_seconds` values.

| Variant | Runs (seconds) | Median (seconds) | Objective | Row violation |
|---|---:|---:|---:|---:|
| Per-kernel synchronization | 5.275648, 2.753199, 2.720693 | 2.753199 | -464.75313959875842 | 6.5887e-7 |
| Queued device primitives | 2.286050, 2.208453, 2.290358 | 2.286050 | -464.75313959875842 | 6.5887e-7 |

The queued path is 16.98% faster on this workload, with identical reported
objective, iteration count (46,600), and feasibility residual.  This is a
single-workload GPU kernel-launch result, not a claim that GPU acceleration
dominates revised simplex on every LP.
