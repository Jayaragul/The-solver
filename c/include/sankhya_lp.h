#ifndef SANKHYA_LP_H
#define SANKHYA_LP_H

#include <stddef.h>

#include "sankhya_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SankhyaVariableType {
    SANKHYA_CONTINUOUS = 0,
    SANKHYA_INTEGER = 1,
    SANKHYA_BINARY = 2
} SankhyaVariableType;

typedef struct SankhyaLPModel {
    SankhyaCSC A;
    double* objective;
    double* row_lower;
    double* row_upper;
    double* column_lower;
    double* column_upper;
    unsigned char* variable_type;
    char** row_names;
    char** column_names;
    double objective_offset;
} SankhyaLPModel;

void sankhya_lp_model_init(SankhyaLPModel* model);
void sankhya_lp_model_destroy(SankhyaLPModel* model);
SankhyaStatus sankhya_lp_read_mps(const char* path, SankhyaLPModel* model);
double sankhya_lp_objective(const SankhyaLPModel* model, const double* x);
double sankhya_lp_max_primal_violation(const SankhyaLPModel* model, const double* x);

#ifdef __cplusplus
}
#endif

#endif

