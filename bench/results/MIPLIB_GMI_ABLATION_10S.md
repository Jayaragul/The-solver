# Root GMI-cut ablation

The referenced LP-and-MLIP-Solver repository uses tableau Gomory mixed-integer
(GMI) cuts as an optional root-strengthening method. This repository now has
the same family behind `MilpSolverOptions::enable_root_gmi_cuts`, with guards
for tiny tableau fractionality, coefficient cleanup, dynamic range, and
cut violation.

## Gate

Frozen 10-second native MIPLIB runs, pseudocost branching, warm starts,
automatic LP parallelism, one repetition:

| instance | GMI off incumbent | GMI on incumbent | reference |
|---|---:|---:|---:|
| `gen-ip002` | −4769.7405888 | −4754.4235292 | −4783.733392 |
| `gen-ip054` | 6858.2629061 | 6898.6748355 | 6840.9656418 |
| `markshare2` | 469* | 570 | 1 |

`*` The `markshare2` off run is time-budget sensitive; the committed
production sweep records 231 on an earlier repeat. The direction of the GMI
comparison is unchanged: it does not improve the incumbent or close the gap.

The raw GMI-on outputs are `MIPLIB_GMI_GENIP002_10S.txt`,
`MIPLIB_GMI_GENIP054_10S.txt`, and `MIPLIB_GMI_MARKSHARE2_10S.txt`.

## Decision

GMI cuts are **implemented but not adopted by default**. They remain useful
for controlled experiments and future cut-pool work; the benchmark rule
requires a demonstrated end-to-end gain before changing production defaults.
