#include "sankhya_cuda.h"

#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    /* min x, subject to x >= 1 and x >= 0; optimum is x = 1. */
    const int offsets[] = {0, 1};
    const int columns[] = {0};
    const double values[] = {1.0};
    const double cost[] = {1.0};
    const double row_lower[] = {1.0};
    const double row_upper[] = {std::numeric_limits<double>::infinity()};
    const double col_lower[] = {0.0};
    const double col_upper[] = {std::numeric_limits<double>::infinity()};
    SankhyaCudaCSR matrix{};
    SankhyaCudaLPSettings settings{5000, 10, 0.25, 2.0, 1.0, 1e-5};
    SankhyaCudaLPResult result{};
    double solution[] = {0.0};
    if (sankhya_cuda_csr_create(&matrix, 1, 1, 1, offsets, columns, values) != 0) return 1;
    const double invalid_cost[] = {std::numeric_limits<double>::quiet_NaN()};
    if (sankhya_cuda_lp_pdhg(&matrix, invalid_cost, row_lower, row_upper, col_lower, col_upper,
            settings, solution, &result) == 0) {
        sankhya_cuda_csr_destroy(&matrix);
        return 2;
    }
    /* Finite input data can still overflow during a sparse product. That is a
       numeric failure, never a convergence result. */
    const double overflow_values[] = {1.0e308};
    const double overflow_lower[] = {1.0e308};
    SankhyaCudaCSR overflow_matrix{};
    if (sankhya_cuda_csr_create(&overflow_matrix, 1, 1, 1, offsets, columns, overflow_values) != 0) {
        sankhya_cuda_csr_destroy(&matrix);
        return 3;
    }
    if (sankhya_cuda_lp_pdhg(&overflow_matrix, cost, overflow_lower, row_upper, col_lower, col_upper,
            settings, solution, &result) != -2 || result.status != -2) {
        sankhya_cuda_csr_destroy(&overflow_matrix);
        sankhya_cuda_csr_destroy(&matrix);
        return 4;
    }
    sankhya_cuda_csr_destroy(&overflow_matrix);
    const int status = sankhya_cuda_lp_pdhg(
        &matrix, cost, row_lower, row_upper, col_lower, col_upper,
        settings, solution, &result);
    sankhya_cuda_csr_destroy(&matrix);
    if (status != 0) {
        std::fprintf(stderr, "PDHG call failed\n");
        return 5;
    }
    if (result.status != 0 || result.maximum_row_violation > 1e-5 || result.maximum_kkt_residual > 1e-5 ||
        std::fabs(solution[0] - 1.0) > 1e-3) {
        std::fprintf(stderr, "PDHG result: status=%d x=%.17g infeas=%.17g kkt=%.17g\n",
            result.status, solution[0], result.maximum_row_violation, result.maximum_kkt_residual);
        return 6;
    }
    return 0;
}
