# MIPLIB feasibility-pump ablation

Date: 2026-09-07

This controlled native Release experiment enables the bounded feasibility pump
and disables the ordinary rounding heuristic (`rounding=off`) so the pump's
effect is isolated. The table uses `fp_weight=0`, i.e. pure distance
minimization. Other settings are pseudocost branching, serial LP execution,
warm starts, a 50,000 pending-basis cap, no GMI, no RENS, and no
coefficient-floor cuts.

| instance | pump LPs | incumbent | reference | best bound | wall s |
|---|---:|---:|---:|---:|---:|
| gen-ip002 | 1 | -4460.4002357 | -4783.733392 | -4814.89206894 | 10.045 |
| gen-ip054 | 1 | none | 6840.96564179 | 6787.87846616 | 10.055 |
| markshare2 | 1 | 469 | 1 | 0.00000000 | 10.075 |
| pk1 | 1 | 21 | 11 | 5.26600509 | 10.023 |

The bounded pump is implemented behind `use_feasibility_pump` and every
candidate still passes the original-model feasibility and integrality gate.
This gate is negative: it does not improve the production rounding heuristic
and remains disabled by default. A stronger future version needs objective
perturbation and a repair/anti-cycling strategy before another adoption test.

An objective tie-break is available through
`feasibility_pump_objective_weight`. On `gen-ip002`, weights `0.1` and `1.0`
still produced `-4460.4002357`; on `markshare2`, weights `1.0` and `10.0`
produced `231`. These measurements do not clear the adoption gate, so the
weight remains zero by default.

Command (final optional arguments are coefficient rounding, pump, and ordinary
rounding respectively):

```text
build/cmake-cuda-release/benchmarks/bench_miplib.exe data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu <instance> 10 pseudocost on serial 1 off 50000 off off on off 0
```
