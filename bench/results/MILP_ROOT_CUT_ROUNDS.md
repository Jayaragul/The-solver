# Root cut-round ablation — rejected

Date: 2026-09-06

This is a controlled ablation of repeated root separation. The experiment
allowed up to four rounds of the existing validity-guarded cover and integer
rounding separators, then re-solved the root relaxation after each round. No
new cut family or numerical tolerance was introduced. The 31 MILP-focused
tests passed before the benchmark run.

## `markshare2`, 10-second native release run

| configuration | cuts | nodes | incumbent | best bound | gap |
|---|---:|---:|---:|---:|---:|
| single root round (baseline) | 2 | 120,012 | 231 | 0 | 0.99568966 |
| up to four root rounds | 7 | 76,289 | 549 | 0 | 0.99818182 |

Command shape for both runs:

```text
bench_miplib data/miplib2017_small data/miplib2017_small/miplib2017-v36.solu markshare2 10 reliability on serial
```

The extra rounds reduced node count but made the incumbent worse. Under the
project rule that an optimization must improve a declared end-to-end KPI
without reducing solution quality, this change is **rejected** and the source
tree remains on the single-round implementation.

## RENS-style incumbent heuristic — rejected

A second controlled experiment fixed integer variables near their LP values
and solved up to three restricted LPs. On the same 10-second release protocol,
`markshare2` remained at incumbent **231** (baseline 231) while `gen-ip002`
remained at **-4762.7873726** (baseline -4762.7873726) and processed more
nodes (34,042 versus 28,243). It therefore supplied no incumbent or
certification improvement and was removed from the default solver.

## Nonnegative lattice-rounding cuts — rejected

For fractional-coefficient, nonnegative integer rows, a prototype rounded
coefficients directionally after scaling by `10^6`. On `gen-ip002` with the
same 10-second release protocol it generated **zero** violated cuts, retained
the incumbent **-4762.7873726**, and processed 41,974 nodes versus the
28,243-node baseline. The prototype was removed; no numerical or benchmark
claim is made for it.
