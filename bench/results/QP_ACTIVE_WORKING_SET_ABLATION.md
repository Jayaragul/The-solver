# Bounded QP working-set fallback ablation

Date: 2026-09-10

The native CPU QP path now includes a verifier-gated dense working-set
fallback for convex models with at most 256 variables plus rows. It is not a
replacement for sparse PDHG: the fallback is attempted only after the
first-order iterate is already nearly feasible, and its candidate is retained
only if the independent original-space KKT verifier accepts it.

## Protocol

```text
build\cmake-cuda-release\sk_qp.exe <instance>.QPS --time-limit 5 --iter-limit 100000 --quiet
```

The focused regression is `sankhya_qp_working_set_smoke`, a 13-variable
positive-semidefinite QP intentionally above the exhaustive active-pattern
cap. It passes with the exact objective and KKT residuals expected by the
hand-derived solution.

The CPU PDHG termination gate now also runs the independent QP KKT verifier
before accepting an apparently stable iterate. This prevents a false early
stop on QAFIRO, where iterate stability previously coexisted with a large
dual residual. The new `sankhya_qafiro_cpu_qp_smoke` CTest regression covers
that case.

## External constrained checks

| Instance | Rows | Cols | Status | Objective | Primal inf | Dual inf | Complementarity |
|---|---:|---:|---|---:|---:|---:|---:|
| `QAFIRO` | 27 | 32 | `optimal` | -1.59078179543 | 4.742e-10 | 2.368e-09 | 1.367e-09 |
| `DUAL1` | 1 | 85 | `iteration_limit` | 0.0350129721602 | 9.796e-08 | 3.705e-02 | 2.327e-05 |

These two runs do not show a benchmark coverage or convergence improvement;
the large dual residuals remain explicit limits. The fallback therefore stays
bounded and certificate-gated, with no production performance claim until a
larger QP ablation demonstrates a measurable gain.
