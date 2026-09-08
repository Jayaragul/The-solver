# MIPLIB integer-inequality propagation ablation

Date: 2026-09-09

This controlled native Release experiment evaluates the new opt-in node
interval propagator for one-sided rows containing only finite-bounded integer
terms. The propagator rounds derived bounds outward and never changes the
global row, so it is a presolve-strengthening experiment rather than a cut
validity shortcut.

Both configurations use pseudocost branching, warm starts, serial certified
simplex relaxations, the default rounding heuristic, and a ten-second limit.
The only change is `integer_inequality_propagation=on`.

| instance | incumbent off | incumbent on | nodes off | nodes on | wall off (s) | wall on (s) |
|---|---:|---:|---:|---:|---:|---:|
| `gen-ip002` | -4770.0310425 | -4770.8663426 | 34,106 | 57,655 | 10.028 | 10.069 |
| `gen-ip054` | 6858.2629061 | 6858.2629061 | 41,224 | 51,150 | 10.038 | 10.050 |
| `markshare2` | 189 | 189 | 105,406 | 115,734 | 10.061 | 10.070 |
| `pk1` | 44 | 44 | 17,304 | 28,987 | 10.013 | 10.025 |

The propagator improved the time-limited `gen-ip002` incumbent, but increased
node work on every case and did not improve the other incumbents. It therefore
remains disabled in the production default. The implementation is retained as
an auditable research option with focused unit coverage; no optimality claim
depends on it.
