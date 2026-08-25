# Benchmark protocol

1. Netlib is the immutable LP correctness suite and is never a tuning target.
2. Larger Mittelmann LPs provide the LP performance workload.
3. Convex QPLIB instances validate QP; a frozen MIPLIB 2017 subset validates MILP.
4. HiGHS is the initial baseline and runs in a fresh, isolated process.
5. Both solvers receive the same original instance, limits, and thread count.
6. Performance runs use randomized order and report the median of at least three.
7. Every claimed LP optimum must pass an independent primal-dual certificate.
8. Record status, objective, residuals, iterations/nodes, time, and peak memory.
9. Any disagreement is listed per instance; timeouts and failures are never hidden.

