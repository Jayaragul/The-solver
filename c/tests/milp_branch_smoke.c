#include "sankhya.h"

#include <math.h>

int main(void)
{
    sk_model model;
    sk_solution solution;
    sk_options options;
    sk_model_init(&model);
    if (sk_model_alloc(&model, 1, 2, 2) != SK_OK) return 1;
    model.A.p[0] = 0; model.A.p[1] = 1; model.A.p[2] = 2;
    model.A.i[0] = 0; model.A.i[1] = 0;
    /* LP relaxation is x=2/3,y=1 (objective -17/6); the best binary point is
       x=1,y=0 (objective -2), so this requires an actual branch. */
    model.A.x[0] = 3.0; model.A.x[1] = 2.0;
    model.c[0] = -2.0; model.c[1] = -1.5;
    model.rlow[0] = -SK_INFINITY; model.rupp[0] = 4.0;
    model.clow[0] = model.clow[1] = 0.0;
    model.cupp[0] = model.cupp[1] = 1.0;
    model.vartype[0] = model.vartype[1] = SK_INTEGER;
    sk_solution_init(&solution);
    sk_options_default(&options);
    options.iteration_limit = 200000;
    options.node_limit = 32;
    if (sk_solve(&model, &options, &solution) != SK_OK || solution.result != SK_RESULT_OPTIMAL ||
        !solution.x || fabs(solution.x[0] - 1.0) > 1e-5 || fabs(solution.x[1]) > 1e-5 ||
        fabs(solution.objective + 2.0) > 1e-5 || solution.nodes < 3 || solution.mip_gap != 0.0) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 2;
    }
    sk_solution_free(&solution);
    sk_model_free(&model);
    return 0;
}
