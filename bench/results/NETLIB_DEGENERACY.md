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
| cycle | optimal | -5.22639302489 | 8.198e-10 | 1.929e-14 | 3428 | 0.999477 |
| d6cube | time limit | 323.5 | 3.638e-12 | 9.367e+02 | 9886 | 5.004387 |
| dfl001 | time limit | 55442369.5401 | 1.010e+02 | 5.015e+07 | 3085 | 5.008687 |
| modszk1 | optimal | 320.619729064 | 1.337e-10 | 5.684e-14 | 995 | 0.119798 |

The `cycle` and `modszk1` results agree with independently measured HiGHS
objectives and close their native primal/dual certificates. The other two rows
remain time limits and are not presented as optimality claims.
This is a targeted regression record, not a replacement for the full Netlib
comparison or a performance ranking.

## Input hashes

SHA-256 (lowercase):

```text
cycle.mps  bfc8edde30952b8dfa354679ea3a3231c2e7eb762c0ff8f0f845505741816d53
d6cube.mps 8cd94ee7783bcbce884acde1f4769f5c79739c9460fe73cb948924e724b02dfa
dfl001.mps 85d47c5f8a5fc53772eabac1805b7a1af72431cad9e14d26d673043980952730
modszk1.mps af109be42aa6bd62fc307b623184c69f1030db2f8e1682c636b387a49fd8da43
```
