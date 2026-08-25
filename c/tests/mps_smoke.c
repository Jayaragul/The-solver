#include "sankhya_lp.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const char* path = "sankhya_mps_smoke.mps";
    const char* text = "NAME TEST\nOBJSENSE\n MIN\nROWS\n N OBJ\n L CAP\n E BAL\nCOLUMNS\n X OBJ 1 CAP 1 BAL 1\n Y OBJ 2 CAP 1 BAL -1\nRHS\n RHS1 OBJ 0 CAP 4 BAL 1\nRANGES\n RNG1 BAL -2\nBOUNDS\n BV BND X\n LO BND Y 0\nENDATA\n";
    FILE* file = fopen(path, "w");
    SankhyaLPModel model;
    double x[] = {1.0, 0.0};
    if (file == NULL) return 1;
    fputs(text, file); fclose(file);
    sankhya_lp_model_init(&model);
    if (sankhya_lp_read_mps(path, &model) != SANKHYA_OK) return 2;
    remove(path);
    if (model.A.rows != 2 || model.A.columns != 2 || model.A.nonzeros != 4) return 3;
    if (fabs(model.objective[0] - 1.0) > 1e-12 || fabs(model.objective[1] - 2.0) > 1e-12) return 4;
    if (model.variable_type[0] != SANKHYA_BINARY) return 5;
    if (fabs(model.row_lower[1] + 1.0) > 1e-12 || fabs(model.row_upper[1] - 1.0) > 1e-12) return 6;
    if (fabs(sankhya_lp_objective(&model, x) - 1.0) > 1e-12) return 7;
    if (sankhya_lp_max_primal_violation(&model, x) > 1e-12) return 8;
    sankhya_lp_model_destroy(&model);
    return 0;
}
