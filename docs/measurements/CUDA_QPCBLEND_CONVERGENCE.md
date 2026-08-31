# CUDA QPCBLEND convergence extension

Date: 2026-09-01

The corrected Release CUDA path was extended from 100,000 to 1,000,000
iterations on `QPCBLEND.QPS` to measure whether the residual limit is merely a
short budget effect.

| Iterations | Objective | Primal inf | KKT residual | Runtime (s) | Independent check |
|---:|---:|---:|---:|---:|---|
| 100,000 | -0.00821668807709 | 6.528e-05 | 6.528e-05 | 9.049160 | passed |
| 1,000,000 | -0.00788942603681 | 1.278e-05 | 2.275e-05 | 97.097687 | passed |

The residual improves by roughly 2.9×, but remains above the `1e-5`
certificate tolerance after 1M iterations. The GPU path is therefore stable
and convergent on this case, but its current first-order method is not yet a
competitive high-accuracy QP solver. The result is retained as an explicit
limit, not an optimality claim; improving sparse convex-QP convergence is a
future algorithm milestone.
