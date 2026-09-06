# MIPLIB coefficient-floor cut ablation

Date: 2026-09-07

This is a controlled native Release comparison on the two general-integer
MIPLIB cases that currently dominate the MILP gap. Both runs use pseudocost
branching, serial LP execution, warm starts, a 50,000 pending-basis cap, no
GMI, and no RENS. The only change is the opt-in pure-integer
coefficient-floor separator (`coefficient_rounding=on`).

| instance | cuts | incumbent | best bound | nodes | LP relaxations | wall s |
|---|---:|---:|---:|---:|---:|---:|
| gen-ip002, off | 0 | -4770.8663426 | -4814.85763130 | 39,254 | 39,254 | 10.040 |
| gen-ip002, on | 0 | -4770.8663426 | -4814.89736479 | 38,882 | 38,882 | 10.045 |
| gen-ip054, off | 0 | 6858.26290606 | 6787.14229898 | 44,189 | 44,429 | 10.065 |
| gen-ip054, on | 0 | 6858.26290606 | 6787.31756125 | 46,176 | 46,416 | 10.052 |

The separator is implemented and covered by a validity regression, but the
gate is negative: it did not improve either incumbent; it slightly reduced
node throughput on `gen-ip054` and was effectively noise on `gen-ip002`.
It remains disabled by default. General MIR and
flow-cover cuts remain separate future work; this experiment is deliberately
narrower and only applies where a pure-integer coefficient-floor proof is
available.

Commands:

```text
build/cmake-cuda-release/benchmarks/bench_miplib.exe data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu gen-ip002 10 pseudocost on serial 2 off 50000 off [on]
build/cmake-cuda-release/benchmarks/bench_miplib.exe data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu gen-ip054 10 pseudocost on serial 2 off 50000 off [on]
```
