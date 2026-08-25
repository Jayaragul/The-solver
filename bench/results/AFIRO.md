# AFIRO — Netlib LP benchmark record

Date: 2026-08-25

## Source and preparation

- Source: Netlib `lp/data/afiro` (compressed MPS).
- Decompression: Netlib's own `lp/data/emps.c`, compiled locally with MSVC.
- Expanded input used for the run: `bench/data/netlib/afiro.mps` (not bundled;
  third-party inputs are excluded from the repository).
- Compressed Netlib file SHA-256: `964D9DF9579E2D16A025693B64057910BA99F1EC6BAFE40EF0098319E40B28BA`.
- Expanded MPS SHA-256: `04992B87E57E57C1C417C96B846833BFA12E055EEAA623F043C45FC773B5BE41`.
- Published Netlib optimum: `-464.75314286`.
- Baseline package: official HiGHS 1.15.1 x86_64 Windows static release (also
  not bundled; it was executed as an isolated process).

## Default automatic-step run

```text
build\native\sankhya_pdhg.exe bench\data\netlib\afiro.mps \
  --iterations 200000 --tolerance 1e-5
```

| Metric | Value |
|---|---:|
| Solver | SANKHYA CUDA PDHG prototype |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB |
| CUDA build target | `sm_86` |
| Automatic operator-norm estimate | `6.7070384958488107` |
| Selected `tau` / `sigma` | `0.074548550796221788` / `0.074548550796221788` |
| Iterations | 40,100 |
| Median GPU solve time (3 runs) | `2.8981232 s` |
| Reported objective | `-464.75316860907924` |
| Published objective | `-464.75314286` |
| Relative objective error | `5.54e-8` |
| Independent primal violation | `9.567336036298002e-6` |
| Independent feasibility check | pass at `1e-5` |
| Integrality check | pass (continuous model) |
| Terminal status | approximate convergence |
| Reproducibility check | CMake-built CUDA CLI produced the identical result | pass |

The three successful solver-clock samples were `2.8981232 s`, `2.8633931 s`,
and `2.9232234 s`. This is a first-order approximate LP result, not an LP
optimality certificate.

## Isolated baseline comparison

The baseline is the official HiGHS 1.15.1 x86_64 Windows static MIT release,
executed as a separate process on the identical expanded MPS file. It is never
linked or imported by SANKHYA.

| Metric | SANKHYA CUDA PDHG | HiGHS 1.15.1 simplex |
|---|---:|---:|
| Objective | `-464.75316860907924` | `-464.75314286` |
| Relative error vs. Netlib optimum | `5.54e-8` | within printed precision |
| Primal verification | pass at `1e-5` | HiGHS reports optimal |
| Iterations | 40,100 | 6 |
| Timing measure | median solver timer, 3 runs | median fresh-process wall time, 3 runs |
| Time | `2.8981232 s` | `0.2830541 s` |

HiGHS reports its internal runtime as `0.00` seconds at its display precision;
the reported `0.2830541 s` is process wall time, including executable startup
and MPS parsing. The two timing columns therefore are not a strict kernel-only
comparison, but they correctly show that this unoptimized PDHG prototype is far
slower on AFIRO. That gap is expected: HiGHS uses presolve and dual simplex,
while SANKHYA performs 40,100 first-order iterations without presolve.

## Prior manual tuning record

Before the automatic-step policy, a manually chosen `tau=sigma=0.05` completed
in 75,000 iterations with objective `-464.75312053654375`, primal violation
`8.4352122229347515e-6`, and a three-run median GPU solve time of `6.116048 s`.
It is retained here for auditability only; it is not the default benchmark path.

HiGHS release zip SHA-256: `B6CB06D5488CD5F3FD241873EE01378AED4A58195D31488E118B5E960D2808D7`.
HiGHS executable SHA-256: `79D5B0825D79BD58A51EBB996579B9A9108857286C4AA2C9C112EFACC71809AA`.

## Unsuccessful conservative run

The following run is retained to avoid parameter cherry-picking:

```text
--iterations 200000 --tau 0.001 --sigma 0.001 --tolerance 1e-5
```

It reached the iteration limit with primal violation `0.34201992630116251` and
is counted as a failure. This shows why an automatic norm-based step-size policy
is required before larger benchmark sweeps.
