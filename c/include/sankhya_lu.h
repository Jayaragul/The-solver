#ifndef SANKHYA_LU_H
#define SANKHYA_LU_H

#include <stddef.h>

#include "sankhya_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SankhyaLU {
    size_t dimension;
    size_t* row_permutation;
    double* factors;
    double pivot_tolerance;
} SankhyaLU;

SankhyaStatus sankhya_lu_factorize(
    SankhyaLU* factor,
    size_t dimension,
    const double* matrix_row_major,
    double pivot_tolerance);

void sankhya_lu_destroy(SankhyaLU* factor);
SankhyaStatus sankhya_lu_solve(const SankhyaLU* factor, const double* rhs, double* solution);
SankhyaStatus sankhya_lu_solve_transpose(const SankhyaLU* factor, const double* rhs, double* solution);

#ifdef __cplusplus
}
#endif

#endif

