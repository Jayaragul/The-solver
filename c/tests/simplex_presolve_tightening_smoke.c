#include "sankhya.h"
#include "sk_simplex.h"

#include <math.h>

int main(void)
{
    /* The singleton row is equivalent to x >= 2. Presolve must transfer that
       fact to the working column bound so simplex starts feasible. */
    sk_model model;
    sk_solution solution;
    sk_options options;
    sk_model_init(&model);
    sk_solution_init(&solution);
    sk_options_default(&options);
    if (sk_model_alloc(&model, 1, 1, 1) != SK_OK) return 1;
    model.A.p[0] = 0; model.A.p[1] = 1;
    model.A.i[0] = 0; model.A.x[0] = 1.0;
    model.c[0] = 1.0;
    model.rlow[0] = 2.0; model.rupp[0] = SK_INFINITY;
    model.clow[0] = 0.0; model.cupp[0] = SK_INFINITY;
    if (sk_simplex_solve(&model, &options, &solution, NULL) != SK_OK ||
        solution.result != SK_RESULT_OPTIMAL || solution.iterations != 0 ||
        fabs(solution.x[0] - 2.0) > 1e-9 || model.clow[0] != 0.0) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 2;
    }
    sk_solution_free(&solution);
    /* The same direct simplex entry point must reject contradictory implied
       bounds before it allocates a basis or enters Phase I. */
    model.cupp[0] = 1.0;
    sk_solution_init(&solution);
    if (sk_simplex_solve(&model, &options, &solution, NULL) != SK_OK ||
        solution.result != SK_RESULT_INFEASIBLE || solution.iterations != 0) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 3;
    }
    sk_solution_free(&solution);
    sk_model_free(&model);
    return 0;
}
