#include "sankhya_lu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static SankhyaStatus lu_fail(const char* message) {
    sankhya_set_error(message);
    return SANKHYA_INVALID_ARGUMENT;
}

void sankhya_lu_destroy(SankhyaLU* factor) {
    if (factor == NULL) return;
    free(factor->row_permutation);
    free(factor->factors);
    factor->dimension = 0;
    factor->row_permutation = NULL;
    factor->factors = NULL;
    factor->pivot_tolerance = 0.0;
}

SankhyaStatus sankhya_lu_factorize(
    SankhyaLU* factor, size_t dimension, const double* matrix, double pivot_tolerance) {
    size_t k;
    double scale = 0.0;
    SankhyaLU candidate;
    if (factor == NULL || matrix == NULL || dimension == 0 || pivot_tolerance <= 0.0) {
        return lu_fail("invalid LU factorization input");
    }
    candidate.dimension = dimension;
    candidate.pivot_tolerance = pivot_tolerance;
    candidate.row_permutation = (size_t*)malloc(dimension * sizeof(size_t));
    candidate.factors = (double*)malloc(dimension * dimension * sizeof(double));
    if (candidate.row_permutation == NULL || candidate.factors == NULL) {
        sankhya_lu_destroy(&candidate);
        return SANKHYA_OUT_OF_MEMORY;
    }
    memcpy(candidate.factors, matrix, dimension * dimension * sizeof(double));
    for (k = 0; k < dimension; ++k) {
        size_t column;
        candidate.row_permutation[k] = k;
        for (column = 0; column < dimension; ++column) {
            double value = fabs(candidate.factors[k * dimension + column]);
            if (value > scale) scale = value;
        }
    }
    if (!isfinite(scale) || scale == 0.0) {
        sankhya_lu_destroy(&candidate);
        return SANKHYA_INVALID_ARGUMENT;
    }
    for (k = 0; k < dimension; ++k) {
        size_t pivot = k;
        size_t row;
        for (row = k + 1; row < dimension; ++row) {
            if (fabs(candidate.factors[row * dimension + k]) >
                fabs(candidate.factors[pivot * dimension + k])) pivot = row;
        }
        if (fabs(candidate.factors[pivot * dimension + k]) <= pivot_tolerance * scale) {
            sankhya_lu_destroy(&candidate);
            return SANKHYA_INVALID_ARGUMENT;
        }
        if (pivot != k) {
            size_t column;
            double temporary;
            size_t permutation = candidate.row_permutation[k];
            candidate.row_permutation[k] = candidate.row_permutation[pivot];
            candidate.row_permutation[pivot] = permutation;
            for (column = 0; column < dimension; ++column) {
                temporary = candidate.factors[k * dimension + column];
                candidate.factors[k * dimension + column] = candidate.factors[pivot * dimension + column];
                candidate.factors[pivot * dimension + column] = temporary;
            }
        }
        for (row = k + 1; row < dimension; ++row) {
            size_t column;
            candidate.factors[row * dimension + k] /= candidate.factors[k * dimension + k];
            for (column = k + 1; column < dimension; ++column) {
                candidate.factors[row * dimension + column] -=
                    candidate.factors[row * dimension + k] * candidate.factors[k * dimension + column];
            }
        }
    }
    sankhya_lu_destroy(factor);
    *factor = candidate;
    return SANKHYA_OK;
}

SankhyaStatus sankhya_lu_solve(const SankhyaLU* factor, const double* rhs, double* solution) {
    size_t i;
    if (factor == NULL || rhs == NULL || solution == NULL || factor->dimension == 0 ||
        factor->factors == NULL || factor->row_permutation == NULL) return SANKHYA_INVALID_ARGUMENT;
    for (i = 0; i < factor->dimension; ++i) solution[i] = rhs[factor->row_permutation[i]];
    for (i = 0; i < factor->dimension; ++i) {
        size_t row;
        for (row = i + 1; row < factor->dimension; ++row)
            solution[row] -= factor->factors[row * factor->dimension + i] * solution[i];
    }
    for (i = factor->dimension; i-- > 0;) {
        size_t column;
        if (factor->factors[i * factor->dimension + i] == 0.0) return SANKHYA_INVALID_ARGUMENT;
        solution[i] /= factor->factors[i * factor->dimension + i];
        for (column = 0; column < i; ++column)
            solution[column] -= factor->factors[column * factor->dimension + i] * solution[i];
    }
    return SANKHYA_OK;
}

SankhyaStatus sankhya_lu_solve_transpose(const SankhyaLU* factor, const double* rhs, double* solution) {
    size_t i;
    if (factor == NULL || rhs == NULL || solution == NULL || factor->dimension == 0 ||
        factor->factors == NULL || factor->row_permutation == NULL) return SANKHYA_INVALID_ARGUMENT;
    memcpy(solution, rhs, factor->dimension * sizeof(double));
    for (i = 0; i < factor->dimension; ++i) {
        size_t row;
        if (factor->factors[i * factor->dimension + i] == 0.0) return SANKHYA_INVALID_ARGUMENT;
        solution[i] /= factor->factors[i * factor->dimension + i];
        for (row = i + 1; row < factor->dimension; ++row)
            solution[row] -= factor->factors[i * factor->dimension + row] * solution[i];
    }
    for (i = factor->dimension; i-- > 0;) {
        size_t column;
        for (column = 0; column < i; ++column)
            solution[column] -= factor->factors[i * factor->dimension + column] * solution[i];
    }
    /* We have solved U'L'z = rhs with z = P*x, so recover x = P'z, that is
       x[row_permutation[i]] = z[i].  A permutation cannot be unwound by
       in-place adjacent swaps, so scatter through a temporary. */
    {
        double* permuted = (double*)malloc(factor->dimension * sizeof(double));
        if (permuted == NULL) return SANKHYA_OUT_OF_MEMORY;
        memcpy(permuted, solution, factor->dimension * sizeof(double));
        for (i = 0; i < factor->dimension; ++i)
            solution[factor->row_permutation[i]] = permuted[i];
        free(permuted);
    }
    return SANKHYA_OK;
}

