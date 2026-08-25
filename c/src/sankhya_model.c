#include "sankhya_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char g_last_error[256];

static SankhyaStatus fail(SankhyaStatus status, const char* message) {
    size_t length = strlen(message);
    if (length >= sizeof(g_last_error)) length = sizeof(g_last_error) - 1;
    memcpy(g_last_error, message, length);
    g_last_error[length] = '\0';
    return status;
}

const char* sankhya_last_error(void) { return g_last_error; }

void sankhya_set_error(const char* message) {
    if (message == NULL) { g_last_error[0] = 0; return; }
    (void)fail(SANKHYA_OK, message);
}

void sankhya_csc_destroy(SankhyaCSC* matrix) {
    if (matrix == NULL) return;
    free(matrix->column_offsets);
    free(matrix->row_indices);
    free(matrix->values);
    matrix->rows = 0;
    matrix->columns = 0;
    matrix->nonzeros = 0;
    matrix->column_offsets = NULL;
    matrix->row_indices = NULL;
    matrix->values = NULL;
}

SankhyaStatus sankhya_csc_validate(const SankhyaCSC* matrix) {
    size_t column;
    if (matrix == NULL) return fail(SANKHYA_INVALID_ARGUMENT, "null CSC matrix");
    if (matrix->column_offsets == NULL ||
        (matrix->nonzeros > 0 && (matrix->row_indices == NULL || matrix->values == NULL))) {
        return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "null CSC storage");
    }
    if (matrix->column_offsets[0] != 0 ||
        matrix->column_offsets[matrix->columns] != matrix->nonzeros) {
        return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "invalid CSC offsets");
    }
    for (column = 0; column < matrix->columns; ++column) {
        size_t start = matrix->column_offsets[column];
        size_t end = matrix->column_offsets[column + 1];
        size_t position;
        if (end < start) return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "decreasing CSC offsets");
        for (position = start; position < end; ++position) {
            int row = matrix->row_indices[position];
            if (row < 0 || (size_t)row >= matrix->rows) {
                return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "CSC row index out of range");
            }
            if (position > start && matrix->row_indices[position - 1] >= row) {
                return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "CSC rows are not strictly sorted");
            }
            if (!isfinite(matrix->values[position])) {
                return fail(SANKHYA_INVALID_SPARSE_STRUCTURE, "nonfinite CSC coefficient");
            }
        }
    }
    g_last_error[0] = '\0';
    return SANKHYA_OK;
}

SankhyaStatus sankhya_csc_create(
    SankhyaCSC* matrix, size_t rows, size_t columns, size_t nonzeros,
    const size_t* column_offsets, const int* row_indices, const double* values) {
    SankhyaCSC candidate;
    if (matrix == NULL || column_offsets == NULL ||
        (nonzeros > 0 && (row_indices == NULL || values == NULL))) {
        return fail(SANKHYA_INVALID_ARGUMENT, "null CSC input");
    }
    candidate.rows = rows;
    candidate.columns = columns;
    candidate.nonzeros = nonzeros;
    candidate.column_offsets = (size_t*)malloc((columns + 1) * sizeof(size_t));
    candidate.row_indices = nonzeros ? (int*)malloc(nonzeros * sizeof(int)) : NULL;
    candidate.values = nonzeros ? (double*)malloc(nonzeros * sizeof(double)) : NULL;
    if (candidate.column_offsets == NULL || (nonzeros > 0 &&
        (candidate.row_indices == NULL || candidate.values == NULL))) {
        sankhya_csc_destroy(&candidate);
        return fail(SANKHYA_OUT_OF_MEMORY, "CSC allocation failed");
    }
    memcpy(candidate.column_offsets, column_offsets, (columns + 1) * sizeof(size_t));
    if (nonzeros > 0) {
        memcpy(candidate.row_indices, row_indices, nonzeros * sizeof(int));
        memcpy(candidate.values, values, nonzeros * sizeof(double));
    }
    if (sankhya_csc_validate(&candidate) != SANKHYA_OK) {
        sankhya_csc_destroy(&candidate);
        return SANKHYA_INVALID_SPARSE_STRUCTURE;
    }
    sankhya_csc_destroy(matrix);
    *matrix = candidate;
    return SANKHYA_OK;
}

SankhyaStatus sankhya_csc_matvec(const SankhyaCSC* matrix, const double* x, double* y) {
    size_t row, column;
    SankhyaStatus status = sankhya_csc_validate(matrix);
    if (status != SANKHYA_OK || x == NULL || y == NULL) {
        return status != SANKHYA_OK ? status : fail(SANKHYA_INVALID_ARGUMENT, "null matvec vector");
    }
    for (row = 0; row < matrix->rows; ++row) y[row] = 0.0;
    for (column = 0; column < matrix->columns; ++column) {
        size_t position;
        for (position = matrix->column_offsets[column]; position < matrix->column_offsets[column + 1]; ++position) {
            y[matrix->row_indices[position]] += matrix->values[position] * x[column];
        }
    }
    return SANKHYA_OK;
}

SankhyaStatus sankhya_csc_transpose_matvec(const SankhyaCSC* matrix, const double* y, double* x) {
    size_t column;
    SankhyaStatus status = sankhya_csc_validate(matrix);
    if (status != SANKHYA_OK || y == NULL || x == NULL) {
        return status != SANKHYA_OK ? status : fail(SANKHYA_INVALID_ARGUMENT, "null transpose vector");
    }
    for (column = 0; column < matrix->columns; ++column) {
        size_t position;
        x[column] = 0.0;
        for (position = matrix->column_offsets[column]; position < matrix->column_offsets[column + 1]; ++position) {
            x[column] += matrix->values[position] * y[matrix->row_indices[position]];
        }
    }
    return SANKHYA_OK;
}

