# CUDA kernels

The initial GPU surface is deliberately small and verifiable:

- `sankhya_cuda_spmv_f64`: host-memory convenience CSR matrix–vector multiply.
- `sankhya_cuda_axpy_f64`: host-memory convenience fused `y = alpha*x + y` update.
- `sankhya_cuda_csr_create` plus `sankhya_cuda_spmv_device_f64`: persistent device-resident CSR for iterative algorithms.
- `sankhya_cuda_axpy_device_f64`: device-pointer vector update for iterative algorithms.
- `sankhya_qp_cuda`: native CUDA CLI for continuous QPS models with either an
  implicit nonnegative diagonal Hessian or a resident sparse Hessian; the
  result is checked by the sovereign C verifier and a component-wise GPU
  stationarity/complementarity gate.

Every iterative result exposes row feasibility, projected KKT, dual
stationarity/sign, and normalized complementarity residuals. The convergence
status is accepted only when all components pass the requested tolerance. The
implementation also recognizes the native C `+/-1e30` infinity sentinel, so
free variables and rows cannot create false complementarity failures.
`sankhya_cuda_qp_pdhg_preconditioned_with_dual` additionally returns the
device dual multipliers so the host `sk_verify` routine can independently
recompute the dual and complementarity certificate.

The host-memory API copies arrays for each call and is a correctness baseline. The
persistent API keeps the sparse matrix resident so repeated Krylov, PDHG, or
interior-point iterations can avoid matrix-transfer overhead. Kernel timings still
require a matched CPU baseline and a reproducible benchmark harness.

Build with CMake and a CUDA-capable host toolchain. The current machine has CUDA
13.3 and an RTX 3050 Laptop GPU; the benchmark manifest records the exact device.
If `nvcc` cannot find a supported host compiler, install/configure MSVC or use the
CUDA toolkit's documented host-compiler override. Do not compare GPU timings until
the CPU baseline uses the same data layout and precision.
