# MIPLIB `pk1` combined heuristic ablation

Date: 2026-09-11

This is a focused native Release run on the previously weak `pk1` instance.
It keeps pseudocost branching, warm node bases, and serial LP relaxations,
while enabling coefficient-floor cuts, the bounded feasibility pump, RENS, and
the ordinary rounding heuristic together.

| configuration | incumbent | reference | best bound | wall s | nodes |
|---|---:|---:|---:|---:|---:|
| production (all optional heuristics off) | 44 | 11 | 4.86045 | 10.027 | 21,673 |
| combined optional configuration | **21** | 11 | 5.27319 | 10.019 | 27,624 |

The combined configuration improves the incumbent from 44 to 21, but it still
does not close the proof gap within the 10-second budget. This is a single
focused ablation, not a production-default promotion; repeated measurement and
the full 19-instance gate are required before changing defaults.

Command:

```text
build\\cmake-cuda-release\\benchmarks\\bench_miplib.exe \\
  data\\miplib2017_small data\\miplib2017_small\\miplib2017-v36.solu pk1 \\
  10 pseudocost on serial 1 off 50000 on on on on 0
```
