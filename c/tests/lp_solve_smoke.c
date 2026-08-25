#include "sankhya.h"

#include <math.h>

int main(void)
{
    sk_model model;
    sk_solution solution;
    sk_options options;
    sk_model_init(&model);
    if (sk_model_alloc(&model, 1, 1, 1) != SK_OK) return 1;
    model.A.p[0] = 0; model.A.p[1] = 1;
    model.A.i[0] = 0; model.A.x[0] = 1.0;
    model.c[0] = 1.0;
    model.rlow[0] = 1.0; model.rupp[0] = SK_INFINITY;
    model.clow[0] = 0.0; model.cupp[0] = SK_INFINITY;
    sk_solution_init(&solution);
    sk_options_default(&options);
    options.iteration_limit = 200000;
    if (sk_solve(&model, &options, &solution) != SK_OK || solution.result != SK_RESULT_OPTIMAL ||
        !solution.x || fabs(solution.x[0] - 1.0) > 1e-5 || fabs(solution.objective - 1.0) > 1e-5 ||
        solution.primal_infeasibility > 1e-7) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 2;
    }
    sk_solution_free(&solution);

    /* Invalid engine selectors must fail explicitly instead of silently
       changing the numerical method. */
    options.lp_engine = 99;
    if (sk_solve(&model, &options, &solution) != SK_ERR_ARG) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 3;
    }

    sk_model_free(&model);
    return 0;
}
