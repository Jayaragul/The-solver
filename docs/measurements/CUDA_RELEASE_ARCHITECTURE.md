# Release CUDA architecture regression and fix

Date: 2026-09-01

The optimized `build/cmake-cuda-release` cache was found targeting CUDA
architecture `75`. On the target RTX 3050 Laptop GPU this produced
`the provided PTX was compiled with an unsupported toolchain`; the Release
CUDA smoke and both GPU CLIs returned a numeric error before iteration 1.

The repository's supported target is Ampere/Ada (`sm_86;sm_89`). Reconfiguring
the Release build with:

```text
cmake -S . -B build/cmake-cuda-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89" \
  -DSANKHYA_BUILD_TESTS=ON -DSANKHYA_ENABLE_CUDA=ON
```

and rebuilding restored all three checks:

| Check | Result |
|---|---|
| Release CUDA sparse-operator smoke | passed |
| Release `sankhya_pdhg` on `unit_lp.mps` | status 0, 200 iterations, objective `1.0000000000004277`, zero verified violation |
| Release `sankhya_qp_cuda` on `tiny_qp.qps` | status 0, 200 iterations, objective `-4`, KKT residual `2.220e-16` |

This was a build-cache/toolchain mismatch, not a solver-kernel defect. The
architecture setting must be kept explicit when reusing an existing CMake
build directory.
