# Benchmark protocol

1. Netlib is the immutable LP correctness suite and is never a tuning target.
2. Larger Mittelmann LPs provide the LP performance workload.
3. Convex QPLIB instances validate QP; a frozen MIPLIB 2017 subset validates MILP.
4. HiGHS is the initial baseline and runs in a fresh, isolated process.
5. Native JSONL records retain cut, heuristic, propagation, refactorization,
   and anti-cycling counters so algorithm changes can be audited separately
   from wall-clock noise.
6. Both solvers receive the same original instance, limits, and thread count.
7. Performance runs use randomized order and report the median of at least three.
8. Every claimed LP optimum must pass an independent primal-dual certificate.
9. Record status, objective, residuals, iterations/nodes, time, and peak memory.
10. Any disagreement is listed per instance; timeouts and failures are never hidden.

## Native collection command

Use the C harness for a frozen file list:

```text
sk_bench.exe --time-limit 5 instance1.mps instance2.qps instance3.mps
```

It emits one JSON object per input with solver provenance, dimensions, status,
objective, valid dual bound when available, residuals, iterations/nodes, and
read/solve times. Non-finite diagnostics are emitted as JSON `null`; a limit
status is retained as a limit and is never rewritten as optimal.

