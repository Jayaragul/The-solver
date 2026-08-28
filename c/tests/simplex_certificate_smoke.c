#include "sankhya.h"
#include "sk_simplex.h"

#include <math.h>

int main(void)
{
    /* Two singleton rows share x.  Both imply the same active value, so the
       postsolve dual repair must consume the reduced-cost residual only once
       rather than applying a stale residual twice. */
    sk_model model;
    sk_solution solution;
    sk_options options;
    sk_model_init(&model);
    sk_solution_init(&solution);
    sk_options_default(&options);
    if (sk_model_alloc(&model, 2, 1, 2) != SK_OK) return 1;
    model.A.p[0] = 0; model.A.p[1] = 2;
    model.A.i[0] = 0; model.A.x[0] = 1.0;
    model.A.i[1] = 1; model.A.x[1] = 1.0;
    model.c[0] = 1.0;
    model.rlow[0] = 1.0; model.rupp[0] = SK_INFINITY;
    model.rlow[1] = -SK_INFINITY; model.rupp[1] = 1.0;
    model.clow[0] = 0.0; model.cupp[0] = SK_INFINITY;
    if (sk_simplex_solve(&model, &options, &solution, NULL) != SK_OK ||
        solution.result != SK_RESULT_OPTIMAL || fabs(solution.x[0] - 1.0) > 1e-8 ||
        solution.dual_infeasibility > 1e-8 || solution.complementarity > 1e-8) {
        sk_solution_free(&solution);
        sk_model_free(&model);
        return 2;
    }
    sk_solution_free(&solution);
    sk_model_free(&model);
    return 0;
}
