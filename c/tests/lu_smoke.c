#include "sankhya_lu.h"

#include <math.h>

int main(void) {
    const double matrix[] = {4.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0};
    const double rhs[] = {1.0, 2.0, 3.0};
    double solution[3];
    SankhyaLU factor = {0};
    if (sankhya_lu_factorize(&factor, 3, matrix, 1e-14) != SANKHYA_OK) return 1;
    if (sankhya_lu_solve(&factor, rhs, solution) != SANKHYA_OK) return 2;
    if (fabs(4.0 * solution[0] + solution[1] - 1.0) > 1e-12 ||
        fabs(2.0 * solution[0] + 3.0 * solution[1] + solution[2] - 2.0) > 1e-12 ||
        fabs(solution[1] + 2.0 * solution[2] - 3.0) > 1e-12) return 3;
    sankhya_lu_destroy(&factor);
    return 0;
}

