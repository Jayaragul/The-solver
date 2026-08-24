#pragma once

#include "CudaBuffer.hpp"
#include "CudaStream.hpp"
#include "CusparseHandle.hpp"
#include "PinnedBuffer.hpp"
#include "../sparse/CSRMatrix.hpp"

#include <cusparse.h>

#include <cstdint>

namespace sihps {

// Device-resident CSR matrix plus the cuSPARSE descriptors/workspace
// needed to run repeated y = A*x via cusparseSpMV -- the single validated
// v1 GPU placement (docs/architecture/CPU_GPU.md \S2.1). Per prompt.md
// \S3.5, this wraps cuSPARSE's generic SpMV API; it does not implement a
// custom SpMV kernel.
//
// All device/host buffers this object owns are sized once, at
// construction, from the matrix's fixed dimensions -- no allocation
// happens inside either multiply() overload
// (docs/architecture/MEMORY.md \S3.2).
class GpuCsrSpMV {
public:
    // CSR_ALG2, not the default/ALG1: docs/architecture/NUMERICS.md \S1
    // requires a deterministic FP64 SpMV, and cuSPARSE documents ALG1 as
    // not bit-reproducible run to run because its row-splitting makes the
    // summation order depend on scheduling (docs/research/SOTA.md \S1.3b).
    //
    // This is a real cost, not a free choice, which is why it is named and
    // measured rather than buried: ALG2's load-balancing does per-call work
    // that ALG1 does not. `cusparseSpMV_preprocess` is issued once at
    // construction to hoist as much of that as the API allows out of the
    // repeated call.
    static constexpr cusparseSpMVAlg_t kDefaultAlgorithm = CUSPARSE_SPMV_CSR_ALG2;

    // `handle` and `stream` are borrowed references (SolveContext-level
    // resources, docs/architecture/SYSTEM.md \S3), not owned by this
    // object.
    //
    // `algorithm` selects the cuSPARSE SpMV variant. It is a parameter
    // rather than a constant because the choice is a measured trade, not a
    // preference -- see kDefaultAlgorithm below and
    // benchmarks/bench_spmv_algorithm.cpp.
    GpuCsrSpMV(const CSRMatrix& host_matrix, CusparseHandle& handle, CudaStream& stream,
                cusparseSpMVAlg_t algorithm = kDefaultAlgorithm);
    ~GpuCsrSpMV();

    GpuCsrSpMV(const GpuCsrSpMV&) = delete;
    GpuCsrSpMV& operator=(const GpuCsrSpMV&) = delete;

    // Full round trip: host x -> device -> cusparseSpMV -> device -> host
    // y, synchronized before returning. This is the correctness-testing
    // entry point (CPU vs GPU equivalence, prompt.md \S3.7 Level 4) -- NOT
    // the shape a real repeated-SpMV hot loop should use, which keeps x/y
    // device-resident across calls (see multiply_device_resident()).
    void multiply(const double* host_x, double* host_y);

    // Assumes device-resident x (see device_x()) is already populated;
    // writes into device-resident y (see device_y()); touches no PCIe
    // transfer and does not synchronize -- the caller times/synchronizes
    // via an explicit CudaEvent. This is the benchmark-relevant path
    // (docs/architecture/CPU_GPU.md \S3.2): only x/y ever cross PCIe per
    // call in the full round trip above, and this path isolates the
    // kernel itself from that transfer cost.
    void multiply_device_resident();

    // Stages host x into device x asynchronously and returns immediately.
    // `host_x` MUST be pinned (a PinnedBuffer) and must stay valid until
    // the stream reaches this copy -- pageable memory would silently make
    // the transfer synchronous, which is the exact failure prompt.md \S3.4
    // warns about ("do not accidentally introduce synchronous transfers
    // into the hot path"). Pairs with multiply_device_resident() to give a
    // caller the full SpMV with no D2H at all (GpuPricer).
    void upload_x_async(const double* pinned_host_x);

    // Rebinds the cuSPARSE dense-vector descriptors to caller-owned device
    // memory, so a repeated-SpMV loop can keep its own x/y buffers and
    // never copy into this object at all.
    //
    // This is what lets a first-order method run an entire restart window
    // without touching the host: bind A to (xbar, Axbar) and A^T to
    // (y, Aty) once, then every SpMV in the window is a single queued call
    // with no rebinding and no transfer (GpuPdlp.hpp). Rebinding itself is
    // a host-side descriptor update -- cheap, but not free, so it belongs
    // outside the loop.
    //
    // The pointers must stay valid and correctly sized (cols() for x,
    // rows() for y) until the next call. Passing nullptr for either leaves
    // that side bound to this object internal buffer.
    void set_vectors(double* d_x, double* d_y);

    double* device_x() noexcept { return d_x_.data(); }
    double* device_y() noexcept { return d_y_.data(); }
    std::int32_t rows() const noexcept { return rows_; }
    std::int32_t cols() const noexcept { return cols_; }
    std::int32_t nnz() const noexcept { return nnz_; }

private:
    std::int32_t rows_, cols_, nnz_;
    CusparseHandle& handle_;
    CudaStream& stream_;

    CudaBuffer<std::int32_t> d_row_ptr_, d_col_idx_;
    CudaBuffer<double> d_values_;
    CudaBuffer<double> d_x_, d_y_;
    CudaBuffer<unsigned char> d_workspace_;

    PinnedBuffer<double> h_x_staging_, h_y_staging_;

    cusparseSpMatDescr_t mat_descr_;
    cusparseDnVecDescr_t vec_x_descr_, vec_y_descr_;

    cusparseSpMVAlg_t algorithm_;
};

} // namespace sihps
