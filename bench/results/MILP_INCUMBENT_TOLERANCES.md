# Absolute MILP incumbent tolerances

Baseline: `07aa0d3`; MSVC Debug tests and Release benchmark on the Windows
development host. This change affects `src/milp`, not the separate C engine.

## Reproduced wrong optima

1. Minimize integer `x` with `0<=x<=1`, `x>=0.25`, and an independent
   continuous variable fixed by `y=1e8`. The true objective is 1. The old
   feasibility check divided every row violation by the largest model RHS,
   admitting rounded `x=0`. The new test failed before the fix.
2. Minimize integer `x` with `1e7<=x<=1e7+2` and `x>=1e7+0.25`. The true
   objective is `1e7+1`. The old integrality check scaled its tolerance with
   `x`, classifying the fractional LP point as integral, while the relative
   feasibility check admitted an infeasible rounded point. This regression
   also failed before the fix.

The tests disable root cuts to expose the incumbent gate independently.
Both now pass, along with all 31 tests selected by `milp_`. The repository
registers 146 C++/CUDA tests; this run does not claim a fresh full-suite result.

## Changed semantics

`MilpSolverOptions::feasibility_tolerance` is now an absolute row/bound
violation limit in original model units. `integrality_tolerance` is an
absolute distance to the nearest integer, consistent with branch candidate
selection. Non-finite activities and objective values are rejected. Default
values remain `1e-6` and `1e-7` respectively. These stricter checks can reject
candidates that the former relative gate admitted. Feasibility polishing is
still needed on difficult, badly scaled models; a caller's tolerances must
be reported when comparing runs. Objective-gap and LP tolerances were not
changed.

## Full bundled MIPLIB sweep

All 19 bundled instances ran in sorted order, with a 10-second limit each,
reliability branching, warm starts on, and serial mode, after the Release
build completed. Every result is preserved in
[raw stdout](MIPLIB_ABSOLUTE_GATE_10S.txt). The executable and all 20 input
files (19 models plus the solution-reference file) are fingerprinted with
SHA-256 in the [manifest](MIPLIB_ABSOLUTE_GATE_10S_manifest.json).

- Six exact reference matches: `22433`, `enlight4` (infeasible), `flugpl`,
  `gr4x6`, `gt2`, and `p0201`.
- Thirteen `TIME_LIMIT` results, retained without treating incumbents as optima.
- Zero explicit `NUMERICAL_FAILURE` statuses; this is not universal proof
  of numerical correctness.

CPU time is substantially below wall time for many rows. These single-run
times therefore do not support a performance improvement claim. This is not
directly comparable with the earlier 60-second sweep: both the budget and
the incumbent gate differ.

The old text runner uses zero placeholders when no incumbent exists. It
also labels `neos859080` as `INCUMBENT_ONLY` because its published reference
is infeasible, even though this run timed out without an incumbent. That
label is not a successful result. Zero RSS is unavailable Windows measurement;
GPU availability does not imply GPU MILP execution. These output limitations
are retained explicitly rather than interpreted as solution evidence.

Run from the Visual Studio x64 developer environment:

```text
cmake --build --preset cuda-debug --target sihps_tests
build\cmake-cuda\tests\sihps_tests.exe milp_
cmake --build --preset cuda-release --target bench_miplib
build\cmake-cuda-release\benchmarks\bench_miplib.exe data\miplib2017_small data\miplib2017_small\miplib2017-v36.solu "" 10 reliability on serial
```

The sweep exits 1 because 13 instances do not match a certified reference
outcome within the limit; all 19 attempts still completed and were recorded.
