# `markshare2` serial/parallel MILP ablation

Date: 2026-09-01

Only the LP relaxation parallel policy changes. Both runs use reliability
branching, warm starts, the optimized native C++/CUDA build, and a 15-second
wall-clock limit.

| LP mode | Status | Incumbent | Best bound | Relative gap | Nodes | LP relaxations | Wall s | Process CPU s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serial | TIME_LIMIT | 231 | 0 | 0.99568966 | 183,177 | 183,443 | 15.125 | 12.719 |
| parallel | TIME_LIMIT | 231 | 0 | 0.99568966 | 48,078 | 48,344 | 15.099 | 58.844 |

Parallel LP work does not improve this combinatorial instance: it processes
about 3.8× fewer nodes while consuming about 4.6× more process CPU time from
thread overhead. The incumbent and bound are unchanged. The result supports
keeping the existing adaptive/serial policy for small MILP nodes and reserving
parallel arithmetic for sufficiently large sparse relaxations.
