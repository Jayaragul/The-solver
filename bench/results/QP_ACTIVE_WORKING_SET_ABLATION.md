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

## External constrained checks

| Instance | Rows | Cols | Status | Objective | Primal inf | Dual inf | Complementarity |
|---|---:|---:|---|---:|---:|---:|---:|
| `QAFIRO` | 27 | 32 | `iteration_limit` | -1.59078237517 | 8.995e-08 | 8.650e+00 | 6.356e-07 |
| `DUAL1` | 1 | 85 | `iteration_limit` | 0.0350129721602 | 9.796e-08 | 3.705e-02 | 2.327e-05 |

These two runs do not show a benchmark coverage or convergence improvement;
the large dual residuals remain explicit limits. The fallback therefore stays
bounded and certificate-gated, with no production performance claim until a
larger QP ablation demonstrates a measurable gain.
