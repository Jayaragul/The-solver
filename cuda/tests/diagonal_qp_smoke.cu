#include "sankhya_cuda.h"

#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    /* min x^2 - 4x subject to x >= 1; optimum is x = 2, objective -4. */
    const int offsets[] = {0, 1};
    const int columns[] = {0};
    const double values[] = {1.0};
    const double diagonal[] = {2.0};
    const double cost[] = {-4.0};
    const double row_lower[] = {1.0};
    const double row_upper[] = {std::numeric_limits<double>::infinity()};
    const double col_lower[] = {0.0};
    const double col_upper[] = {std::numeric_limits<double>::infinity()};
    SankhyaCudaCSR matrix{};
    SankhyaCudaLPSettings settings{10000, 10, 0.15, 0.5, 1.0, 1e-7};
    SankhyaCudaLPResult result{};
    double solution[] = {0.0};
    if (sankhya_cuda_csr_create(&matrix, 1, 1, 1, offsets, columns, values) != 0) return 1;
    const int status = sankhya_cuda_diagonal_qp_pdhg(&matrix, diagonal, cost,
        row_lower, row_upper, col_lower, col_upper, settings, solution, &result);
    sankhya_cuda_csr_destroy(&matrix);
    if (status != 0 || result.status != 0 || result.maximum_row_violation > 1e-7 ||
        std::fabs(solution[0] - 2.0) > 1e-5 || std::fabs(result.objective + 4.0) > 1e-5) {
        std::fprintf(stderr, "Diagonal QP result: status=%d x=%.17g objective=%.17g infeas=%.17g\n",
            result.status, solution[0], result.objective, result.maximum_row_violation);
        return 2;
    }
    return 0;
}
