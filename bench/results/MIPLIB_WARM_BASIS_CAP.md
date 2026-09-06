# Warm-start basis retention cap

Warm-started node relaxations retain a parent simplex basis for queued child
nodes. On a model whose bound does not prune the tree, retaining every basis
can make memory grow with frontier width. `MilpSolverOptions::
max_pending_warm_start_bases` now bounds that retention; a child beyond the
cap takes the existing cold, certified LP path.

## Protocol

`markshare2`, native C++/CUDA debug build, pseudocost branching, warm starts,
automatic LP parallelism, GMI off. The cap is zero for unlimited retention.

| budget | cap | nodes | incumbent | peak RSS | warm solves | cap skips |
|---:|---:|---:|---:|---:|---:|---:|
| 10 s | unlimited | 20,463 | 231 | 132.816 MB | 20,459 | 0 |
| 10 s | 1,000 | 11,752 | 231 | 119.676 MB | 4,465 | 18,033 |
| 30 s | 50,000 | 48,061 | 231 | 156.371 MB | 48,057 | 0 |
| 60 s | 50,000 | 89,375 | 231 | 186.230 MB | 89,370 | 39,374 |

The tight 1,000-basis cap trades node throughput for lower memory, but keeps
the same valid incumbent and bound. The production 50,000-basis cap does not
activate in the 30-second run, then activates cleanly during the 60-second
run. An unlimited 60-second comparison was intentionally not run on this
16-GB laptop: the point of the cap is to avoid assuming that unbounded
frontier memory is safe.

Raw logs: `MIPLIB_WARM_BASIS_UNLIMITED_10S.txt`,
`MIPLIB_WARM_BASIS_CAP1000_10S.txt`,
`MIPLIB_WARM_BASIS_CAP50000_30S.txt`, and
`MIPLIB_WARM_BASIS_CAP50000_60S.txt`.
