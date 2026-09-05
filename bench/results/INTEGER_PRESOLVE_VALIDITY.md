# Integer presolve validity regressions

2026-09-05, native Windows MSVC build; baseline `f07fe51` plus this fix.
This record covers correctness and two Release integration checks. It does
not establish a speedup or validate the full MIPLIB collection.

## Reproduced failures

With integer `y=1e9`, the equality `x - 5e-10*y = 0.5` admits `x=1`.
Both equality propagation and the GCD gate treated the small coefficient as
zero, which could declare the model infeasible. Likewise, RHS lattice rounding
of `2*x - 5e-10*y <= 1.5` could remove `x=1`.

The ranged row `2*x+s=1`, `-1<=s<=1`, admits `x=1`. Checking it as the
exact equality `2*x=1` also produced a false infeasibility result. The new
small-coefficient and ranged-equality tests failed on the baseline.

## Fix and validation

Integer coefficients must now be exact and within `2^53-1`. Equality gates
require a zero fixed slack. GCD tests operate directly on the RHS without
subtracting lower-bound activities, so free integer variables are supported.
Propagation checks the total absolute activity plus RHS against the exact
integer range before using its sums. RHS lattice rounding skips nonzero
finite slack endpoints and avoids lower-bound shifts.

The focused Debug run passed all **29 MILP tests**. Six new tests cover:

- Separate propagation, GCD and RHS-rounding paths with a small coefficient.
- Ranged equalities and both signs of nonzero inequality slack endpoints.
- Coefficients at `2^53`, positive/negative `2^63`, and `1e300` on fixed-zero
  models, exercising the conversion guards for equality and inequality rows.
- The impossible equation `2*x+4*y=1` with free integer variables, pruned by
  GCD before any LP relaxation.
- 36 signed-coefficient bounded integer models, including ranged and
  fractional-RHS cases, each solved with reductions on and off and compared
  with exhaustive enumeration of all 36 points in `[-2,3]^2`.

```text
cmake --build --preset cuda-debug --target sihps_tests
build\cmake-cuda\tests\sihps_tests.exe milp_
```

The repository now registers 144 C++/CUDA tests. Only the focused 29-test
selection was rerun for this change; the unrelated suite is not a fresh claim.

## Release integration results

Each model ran separately after the build, with a 15-second limit,
reliability branching, warm starts on and serial mode. Both returned `EXACT`
against `miplib2017-v36.solu`.

| Instance | Status | Objective / best bound | Absolute reference error | Gap | Nodes | LPs | Wall seconds | CPU seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 22433 | OPTIMAL | 21477 | 7.27595761418e-12 | 0 | 47 | 259 | 3.473 | 3.469 |
| blend2 | OPTIMAL | 7.598985 | 0 | 0 | 7859 | 5679 | 6.432 | 6.391 |

`blend2` reported seven warm-start fallbacks; `22433` reported zero. The
Windows runner still prints zero RSS, which is missing measurement, not zero
memory use. GPU availability is reported but is not a claim of GPU execution
for these serial simplex MILP solves. These single-run times cannot attribute
a speed difference to the presolve change.

```text
cmake --build --preset cuda-release --target bench_miplib
build\cmake-cuda-release\benchmarks\bench_miplib.exe data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu 22433 15 reliability on serial
build\cmake-cuda-release\benchmarks\bench_miplib.exe data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu blend2 15 reliability on serial
```

SHA-256 inputs:

```text
22433.mps: 395621D0C47BEA253F5023448A11E1A89D19A7FBA1802DB6C7EF626CEEDE1F95
blend2.mps: 48C009606D5E949BAA8E5E060C8B0CAFD3EB96F9D615D98FFF0D626D5B6E503F
miplib2017-v36.solu: 3EB5D830BF6DD39002705C9D99F9CD59BB7C3B10BF6295C371563EBB0DE5AA4A
```
