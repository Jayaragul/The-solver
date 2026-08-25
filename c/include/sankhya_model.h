#ifndef SANKHYA_MODEL_H
#define SANKHYA_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SankhyaStatus {
    SANKHYA_OK = 0,
    SANKHYA_INVALID_ARGUMENT = 1,
    SANKHYA_OUT_OF_MEMORY = 2,
    SANKHYA_INVALID_SPARSE_STRUCTURE = 3
} SankhyaStatus;

typedef struct SankhyaCSC {
    size_t rows;
    size_t columns;
    size_t nonzeros;
    size_t* column_offsets;
    int* row_indices;
    double* values;
} SankhyaCSC;

/* Allocate and deep-copy a validated CSC matrix. */
SankhyaStatus sankhya_csc_create(
    SankhyaCSC* matrix,
    size_t rows,
    size_t columns,
    size_t nonzeros,
    const size_t* column_offsets,
    const int* row_indices,
    const double* values);

void sankhya_csc_destroy(SankhyaCSC* matrix);
SankhyaStatus sankhya_csc_validate(const SankhyaCSC* matrix);
SankhyaStatus sankhya_csc_matvec(const SankhyaCSC* matrix, const double* x, double* y);
SankhyaStatus sankhya_csc_transpose_matvec(const SankhyaCSC* matrix, const double* y, double* x);
const char* sankhya_last_error(void);
void sankhya_set_error(const char* message);

#ifdef __cplusplus
}
#endif

#endif

