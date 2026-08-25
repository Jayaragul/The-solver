#include "sankhya.h"

#include <math.h>

int main(void)
{
    sk_model model;
    sk_solution solution;
    sk_options options;
    sk_model_init(&model);
    sk_solution_init(&solution);
    sk_options_default(&options);
    if (sk_model_alloc(&model, 1, 1, 1) != SK_OK) return 1;
    model.A.p[0] = 0; model.A.p[1] = 1;
    model.A.i[0] = 0; model.A.x[0] = 1.0;
    model.rlow[0] = 0.0; model.rupp[0] = 1.0;
    model.clow[0] = 0.0; model.cupp[0] = 1.0;
    if (sk_model_validate(&model) != SK_OK) { sk_model_free(&model); return 2; }
    model.A.p[1] = 2;
    if (sk_model_validate(&model) != SK_ERR_STRUCTURE ||
        sk_solve(&model, &options, &solution) != SK_ERR_STRUCTURE) {
        sk_solution_free(&solution); sk_model_free(&model); return 3;
    }
    model.A.p[1] = 1;
    model.A.x[0] = NAN;
    if (sk_model_validate(&model) != SK_ERR_STRUCTURE) {
        sk_solution_free(&solution); sk_model_free(&model); return 4;
    }
    sk_solution_free(&solution);
    sk_model_free(&model);
    return 0;
}
