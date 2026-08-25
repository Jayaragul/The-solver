#ifndef SANKHYA_VERIFY_H
#define SANKHYA_VERIFY_H

#include "sankhya_lp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SankhyaVerificationResult {
    int feasible;
    int integral;
    double objective;
    double maximum_primal_violation;
    double maximum_integrality_violation;
    double objective_error;
} SankhyaVerificationResult;

SankhyaStatus sankhya_verify_primal(
    const SankhyaLPModel* model,
    const double* solution,
    double tolerance,
    SankhyaVerificationResult* result);

SankhyaStatus sankhya_verify_objective(
    const SankhyaLPModel* model,
    const double* solution,
    double reference_objective,
    double tolerance,
    SankhyaVerificationResult* result);

#ifdef __cplusplus
}
#endif

#endif

