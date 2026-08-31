# `markshare2` branching ablation

Date: 2026-09-01

All runs use the same optimized native C++/CUDA build, warm starts enabled,
serial LP relaxations, and a 15-second limit. Only the branching rule changes.

| Rule | Status | Incumbent | Best bound | Relative gap | Nodes | LP relaxations | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Most fractional | TIME_LIMIT | 231 | 0 | 0.99568966 | 193,296 | 193,328 | 15.208 |
| Pseudocost | TIME_LIMIT | 231 | 0 | 0.99568966 | 224,846 | 224,878 | 15.154 |
| Reliability | TIME_LIMIT | 231 | 0 | 0.99568966 | 219,345 | 219,611 | 15.154 |

The three policies reach the same incumbent and bound. Pseudocost processes
slightly more nodes, but the difference is not a reliable quality improvement;
no default was changed. The result reinforces that `markshare2` needs a
stronger primal repair or cutting strategy rather than another ordinary
branching-policy toggle.
