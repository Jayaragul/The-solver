# Netlib degeneracy follow-up

Date: 2026-08-25

This focused run evaluates the revised-simplex anti-degeneracy policy added
after the initial Netlib sweep. After more than 200 zero-step pivots, pricing
temporarily switches from Devex to Dantzig; Bland's rule is retained as a
later finite fallback. All models use the same native executable and a fixed
five-second wall limit.

## Results

| Instance | Status | Objective at stop | Primal infeasibility | Dual infeasibility | Iterations | Seconds |
|---|---|---:|---:|---:|---:|---:|
| cycle | optimal | -5.22639302489 | 2.471e-11 | 3.085e-07 | 3224 | 1.075548 |
| d6cube | time limit | 343.164337309 | 3.638e-12 | 2.252e+02 | 8941 | 5.004574 |
| dfl001 | time limit | 94356633.6649 | 1.010e+02 | 6.231e+07 | 3618 | 5.009222 |
| modszk1 | time limit | 699998.447478 | 0.000e+00 | 1.041e+01 | 41974 | 5.000793 |

The `cycle` result now agrees with the independently measured HiGHS objective
`-5.226393024894102` and closes its native primal/dual certificate. The other
three rows remain time limits and are not presented as optimality claims.
This is a targeted regression record, not a replacement for the full Netlib
comparison or a performance ranking.
