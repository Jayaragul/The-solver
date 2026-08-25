#include "sankhya.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    const char *path = "sankhya_qps_smoke.qps";
    FILE *file = fopen(path, "w");
    sk_model model;
    sk_solution solution;
    sk_options options;
    if (!file) return 1;
    fputs("NAME DIAGQP\nROWS\n N OBJ\n G R1\nCOLUMNS\n X1 OBJ -4 R1 1\n X2 OBJ 0\nRHS\n RHS1 R1 1\nBOUNDS\n LO BND X1 0\nQUADOBJ\n X1 X1 2\n X1 X2 1\n X2 X2 4\nENDATA\n", file);
    fclose(file);
    sk_model_init(&model);
    if (sk_read_mps(path, &model) != SK_OK) return 2;
    if (!model.Q || model.Q->nrow != 2 || model.Q->ncol != 2 || model.Q->nzmax != 4 ||
        model.Q->p[0] != 0 || model.Q->p[1] != 2 || model.Q->p[2] != 4 ||
        model.Q->i[0] != 0 || model.Q->x[0] != 2.0 ||
        model.Q->i[1] != 1 || model.Q->x[1] != 1.0 ||
        model.Q->i[2] != 0 || model.Q->x[2] != 1.0 ||
        model.Q->i[3] != 1 || model.Q->x[3] != 4.0) {
        sk_model_free(&model);
        return 3;
    }
    sk_solution_init(&solution);
    sk_options_default(&options);
    options.iteration_limit = 200000;
    if (sk_solve(&model, &options, &solution) != SK_OK || solution.result != SK_RESULT_OPTIMAL ||
        !solution.x || fabs(solution.x[0] - 2.0) > 1e-5 || fabs(solution.x[1]) > 1e-5 ||
        fabs(solution.objective + 4.0) > 1e-5 || solution.primal_infeasibility > 1e-7) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 4;
    }
    if (sk_verify(&model, &solution) != SK_OK || fabs(solution.objective + 4.0) > 1e-5 ||
        solution.primal_infeasibility > 1e-7 || solution.dual_infeasibility > 1e-5 ||
        solution.complementarity > 1e-5) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 5;
    }
    sk_solution_free(&solution);
    sk_model_free(&model);
    remove(path);
    return 0;
}
