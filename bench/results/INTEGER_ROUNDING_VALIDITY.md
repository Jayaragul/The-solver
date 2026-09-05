# Integer-row rounding validity regression

Date: 2026-09-05. Baseline: `e3e1136`. Native MSVC Debug C++/CUDA build.
This is an adversarial correctness record, not a performance benchmark.

The baseline separator accepted coefficients within `1e-9` of an integer.
That can remove a true optimum. Consider minimizing `-x` for integer
`0 <= x <= 2`, integer `y = 1e9`, and `x - 5e-10*y <= 0.75`.
The true optimum is `x=1`, objective `-1`. Dropping the small coefficient
creates the invalid cut `x<=0`. The new objective regression failed on the
baseline and passes after requiring exactly integral coefficients.

The separator also now computes the effective side using slack bounds.
For `x+y+s=0.75`, `s>=-0.5`, the actual side is `x+y<=1.25`, so its valid
integer rounding is `x+y<=1`. Rounding the stored RHS alone yields `x+y<=0`
and removes valid solutions.

Validation: all 23 tests selected by `milp_` passed. New tests check both
inequality orientations of the small-coefficient and slack-bound examples,
and 24 signed-coefficient, bounded-integer problems against exhaustive search
over every point in `[-2,3]^2`. At least one cut must be generated in that
enumeration suite. This establishes these regression cases, not universal
numerical correctness or a MIPLIB speedup.

Reproduce in the Visual Studio x64 developer environment:

```text
cmake --build --preset cuda-debug --target sihps_tests
build\cmake-cuda\tests\sihps_tests.exe milp_
```

The test runner accepts an optional name substring; no argument still runs
all tests. An unmatched filter exits with code 2, preventing an empty green run.

A Release integration check on MIPLIB `22433`, using a 10-second limit,
reliability branching, warm starts and serial mode, returned `OPTIMAL`:
objective 21477, absolute reference error `7.27595761418e-12`, bound 21477,
gap 0, 47 processed nodes, 259 LP relaxations, and 7.536 wall seconds.
This single run checks integration; it is not a before/after speed comparison.

```text
cmake --build --preset cuda-release --target bench_miplib
build\cmake-cuda-release\benchmarks\bench_miplib.exe data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu 22433 10 reliability on serial
```
