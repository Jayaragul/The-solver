#include "sankhya_verify.h"

#include <math.h>
#include <stdlib.h>

int main(void) {
    const size_t offsets[] = {0, 1, 2};
    const int rows[] = {0, 0};
    const double values[] = {1.0, 1.0};
    SankhyaLPModel model;
    SankhyaVerificationResult result;
    double* x;
    sankhya_lp_model_init(&model);
    if (sankhya_csc_create(&model.A, 1, 2, 2, offsets, rows, values) != SANKHYA_OK) return 1;
    model.objective = (double*)malloc(2 * sizeof(double));
    model.row_lower = (double*)malloc(sizeof(double));
    model.row_upper = (double*)malloc(sizeof(double));
    model.column_lower = (double*)calloc(2, sizeof(double));
    model.column_upper = (double*)malloc(2 * sizeof(double));
    model.variable_type = (unsigned char*)calloc(2, sizeof(unsigned char));
    if (!model.objective || !model.row_lower || !model.row_upper || !model.column_lower || !model.column_upper || !model.variable_type) return 2;
    model.objective[0] = 1.0; model.objective[1] = 2.0;
    model.row_lower[0] = 1.0; model.row_upper[0] = 1.0;
    model.column_upper[0] = model.column_upper[1] = 1.0;
    x = (double*)malloc(2 * sizeof(double)); if (!x) return 3;
    x[0] = 1.0; x[1] = 0.0;
    if (sankhya_verify_objective(&model, x, 1.0, 1e-9, &result) != SANKHYA_OK) return 4;
    if (!result.feasible || !result.integral || result.objective_error > 1e-12) return 5;
    x[1] = 0.25;
    if (sankhya_verify_primal(&model, x, 1e-9, &result) != SANKHYA_OK) return 6;
    if (result.feasible || result.maximum_primal_violation < 0.24) return 7;
    free(x); sankhya_lp_model_destroy(&model);
    return 0;
}
