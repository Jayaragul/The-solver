#include "sankhya_verify.h"

#include <math.h>
#include <stdlib.h>

static void clear_result(SankhyaVerificationResult* result) {
    result->feasible = 0;
    result->integral = 1;
    result->objective = NAN;
    result->maximum_primal_violation = INFINITY;
    result->maximum_integrality_violation = INFINITY;
    result->objective_error = INFINITY;
}

SankhyaStatus sankhya_verify_primal(
    const SankhyaLPModel* model, const double* solution, double tolerance,
    SankhyaVerificationResult* result) {
    size_t i;
    double* activity;
    double maximum_violation = 0.0;
    double maximum_integrality = 0.0;
    if (result == NULL) return SANKHYA_INVALID_ARGUMENT;
    clear_result(result);
    if (model == NULL || solution == NULL || tolerance < 0.0 || !isfinite(tolerance)) return SANKHYA_INVALID_ARGUMENT;
    activity = (double*)calloc(model->A.rows, sizeof(double));
    if (activity == NULL && model->A.rows > 0) return SANKHYA_OUT_OF_MEMORY;
    if (sankhya_csc_matvec(&model->A, solution, activity) != SANKHYA_OK) { free(activity); return SANKHYA_INVALID_SPARSE_STRUCTURE; }
    for (i = 0; i < model->A.rows; ++i) {
        if (isfinite(model->row_lower[i]) && model->row_lower[i] - activity[i] > maximum_violation) maximum_violation = model->row_lower[i] - activity[i];
        if (isfinite(model->row_upper[i]) && activity[i] - model->row_upper[i] > maximum_violation) maximum_violation = activity[i] - model->row_upper[i];
    }
    for (i = 0; i < model->A.columns; ++i) {
        if (!isfinite(solution[i])) maximum_violation = INFINITY;
        if (isfinite(model->column_lower[i]) && model->column_lower[i] - solution[i] > maximum_violation) maximum_violation = model->column_lower[i] - solution[i];
        if (isfinite(model->column_upper[i]) && solution[i] - model->column_upper[i] > maximum_violation) maximum_violation = solution[i] - model->column_upper[i];
        if (model->variable_type[i] != SANKHYA_CONTINUOUS) {
            double distance = fabs(solution[i] - round(solution[i]));
            if (distance > maximum_integrality) maximum_integrality = distance;
        }
    }
    free(activity);
    result->objective = sankhya_lp_objective(model, solution);
    result->maximum_primal_violation = maximum_violation > 0.0 ? maximum_violation : 0.0;
    result->maximum_integrality_violation = maximum_integrality;
    result->integral = maximum_integrality <= tolerance;
    result->feasible = result->maximum_primal_violation <= tolerance;
    return SANKHYA_OK;
}

SankhyaStatus sankhya_verify_objective(
    const SankhyaLPModel* model, const double* solution, double reference_objective,
    double tolerance, SankhyaVerificationResult* result) {
    SankhyaStatus status = sankhya_verify_primal(model, solution, tolerance, result);
    double scale;
    if (status != SANKHYA_OK) return status;
    scale = 1.0 + fabs(reference_objective);
    result->objective_error = fabs(result->objective - reference_objective) / scale;
    return SANKHYA_OK;
}

