#include "sankhya.h"

#include <math.h>

/* A bounded interruption must retain the active node's relaxation bound.
   The root rounding heuristic finds -17, then the search finds -20 before
   the fifth node limit; the remaining open relaxation is strictly lower. */
int main(void)
{
    sk_model m;
    sk_solution s;
    sk_options o;

    sk_model_init(&m);
    if (sk_model_alloc(&m, 1, 4, 4) != SK_OK) return 1;
    m.A.p[0] = 0; m.A.p[1] = 1; m.A.p[2] = 2; m.A.p[3] = 3; m.A.p[4] = 4;
    m.A.i[0] = m.A.i[1] = m.A.i[2] = m.A.i[3] = 0;
    m.A.x[0] = 3.0; m.A.x[1] = 5.0; m.A.x[2] = 2.0; m.A.x[3] = 4.0;
    m.c[0] = -10.0; m.c[1] = -13.0; m.c[2] = -7.0; m.c[3] = -9.0;
    m.rlow[0] = -SK_INFINITY; m.rupp[0] = 7.0;
    m.clow[0] = m.clow[1] = m.clow[2] = m.clow[3] = 0.0;
    m.cupp[0] = m.cupp[1] = m.cupp[2] = m.cupp[3] = 1.0;
    m.vartype[0] = m.vartype[1] = m.vartype[2] = m.vartype[3] = SK_INTEGER;

    sk_solution_init(&s);
    sk_options_default(&o);
    o.node_limit = 5;
    if (sk_solve(&m, &o, &s) != SK_OK || s.result != SK_RESULT_ITERATION_LIMIT ||
        !s.x || !isfinite(s.objective) || fabs(s.objective + 20.0) > 1e-6 ||
        !isfinite(s.dual_bound) || !(s.dual_bound < s.objective - 1e-6)) {
        sk_solution_free(&s);
        sk_model_free(&m);
        return 2;
    }
    sk_solution_free(&s);
    sk_model_free(&m);
    return 0;
}
