# Netlib-31 — native revised-simplex fixed-limit sweep

Date: 2026-08-25

This is the first breadth check of the native C simplex path. It is deliberately
reported as a fixed-limit correctness sweep, not as a performance ranking.
All models are expanded from the public Netlib LP distribution with Netlib's
own `emps.c`; the third-party inputs remain outside version control under
`bench/data/netlib`.

## Protocol

```text
build\cmake-cuda\sk_lp.exe bench\data\netlib\mps\<name>.mps \
  --time-limit 5 --iter-limit 500000 --quiet
```

- One thread, native C revised simplex, four-pass power-of-two scaling.
- A result is called *optimal* here only when `sk_lp` returned that terminal
  status and its independent verifier recomputed primal, dual, and gap
  diagnostics from the original (unscaled) MPS model.
- A time limit is retained exactly as returned. It is not a proof of failure,
  infeasibility, or an optimality claim.
- The AFIRO record contains the isolated official HiGHS comparison required by
  the benchmark contract: [AFIRO](AFIRO.md).

## Summary

| Outcome | Count |
|---|---:|
| Optimal with independent diagnostics | 25 |
| Time limit at 5 s | 6 |
| Incorrectly claimed optimum | 0 |

For the 25 optimal rows, the largest reported primal violation was
`4.657e-10` (`agg`), the largest dual infeasibility was `2.648e-7`
(`etamacro`), and the largest relative LP gap was `1.102e-9` (`etamacro`).
Those values are reported rather than rounded away.

## Per-instance results

| Instance | Rows | Cols | Status | Objective at stop | Iterations | Solver s |
|---|---:|---:|---|---:|---:|---:|
| 25fv47 | 821 | 1571 | optimal | 5501.84588829 | 3283 | 1.435928 |
| 80bau3b | 2262 | 9799 | optimal | 987224.192409 | 7151 | 2.755293 |
| adlittle | 56 | 97 | optimal | 225494.963162 | 101 | 0.001571 |
| afiro | 27 | 32 | optimal | -464.753142857 | 16 | 0.000191 |
| agg | 488 | 163 | optimal | -35991767.2866 | 111 | 0.006530 |
| agg2 | 516 | 302 | optimal | -20239252.356 | 160 | 0.011308 |
| agg3 | 516 | 302 | optimal | 10312115.9351 | 163 | 0.010726 |
| bandm | 305 | 472 | optimal | -158.62801845 | 599 | 0.054802 |
| beaconfd | 173 | 262 | optimal | 33592.4858072 | 151 | 0.007709 |
| blend | 74 | 83 | optimal | -30.8121498458 | 88 | 0.006610 |
| bnl1 | 643 | 1175 | optimal | 1977.62956152 | 1417 | 0.265679 |
| bnl2 | 2324 | 3489 | optimal | 1811.23654036 | 4431 | 2.588791 |
| boeing1 | 351 | 384 | optimal | -335.213567507 | 593 | 0.035845 |
| boeing2 | 166 | 143 | optimal | -315.018728015 | 190 | 0.006117 |
| bore3d | 233 | 315 | optimal | 1373.08039421 | 238 | 0.012490 |
| brandy | 220 | 249 | optimal | 1518.50989649 | 595 | 0.022769 |
| capri | 271 | 353 | optimal | 2690.01291377 | 590 | 0.019848 |
| czprob | 929 | 3523 | optimal | 2185196.69886 | 2120 | 0.184908 |
| degen2 | 444 | 534 | optimal | -1435.178 | 1486 | 0.123529 |
| e226 | 223 | 282 | optimal | -11.6389290664 | 692 | 0.025355 |
| etamacro | 400 | 688 | optimal | -755.715232935 | 1193 | 0.050361 |
| fffff800 | 524 | 854 | optimal | 555679.564817 | 984 | 0.054217 |
| finnis | 497 | 614 | optimal | 172791.065596 | 662 | 0.026049 |
| fit1d | 24 | 1026 | optimal | -9146.37809242 | 2691 | 0.113134 |
| fit1p | 627 | 1677 | optimal | 9146.37809242 | 3175 | 0.393925 |
| cycle | 1903 | 2857 | time limit | -5.1642570399 | 16185 | 5.003810 |
| d2q06c | 2171 | 5167 | time limit | 309112.413835 | 8563 | 5.006465 |
| d6cube | 415 | 6184 | time limit | 402.551588329 | 11462 | 5.010693 |
| degen3 | 1503 | 1818 | time limit | -978.6548125 | 8515 | 5.006061 |
| dfl001 | 6071 | 12230 | time limit | 88954634.0786 | 4066 | 5.017722 |
| fit2d | 25 | 10500 | time limit | -26405.5289589 | 5649 | 5.018498 |

The 31 rows above are a single deterministic prefix of the locally expanded
corpus used during this run. They should be rerun with randomized order and
multiple repetitions before any timing comparison is made. The remaining
downloaded Netlib instances are intentionally not summarized yet.
