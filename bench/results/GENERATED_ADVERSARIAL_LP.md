# Generated adversarial LP regression

This deterministic regression complements Netlib with generated sparse models
whose coefficient magnitudes span six orders of magnitude.

## Protocol

- 12 trials, each with 8 rows and 12 bounded variables.
- Sparse coefficients use random signs and scales from `1e-6` through `1e6`.
- A seeded feasible point constructs every upper row bound, so each model is
  known feasible and bounded by construction.
- Seed: `0x51A5` (`std::mt19937`), fixed for reproducibility.
- Presolve and Ruiz equilibration are enabled.
- Acceptance requires optimal status, finite objective, and original-space
  primal residual at most `1e-6`.

## Result

The focused regression passes **12/12**. The complete C++/CUDA suite passes
**148/148**, with 28 additional native C smoke tests. The test is implemented
in `tests/lp/test_generated_adversarial.cpp` and included in `sihps_tests`.

This is not yet MPS parser fuzzing, sparse-structure mutation fuzzing, or a
Compute Sanitizer run; those remain explicit follow-up work.
