#include "sankhya.h"
#include "sk_milp.h"

#include <math.h>

int main(void)
{
    sk_model m;
    sk_options o;
    sk_solution s;
    sk_milp_stats stats;
    int rc;

    sk_model_init(&m);
    if (sk_model_alloc(&m, 1, 2, 2) != SK_OK) return 1;
    /* x + y >= 1.2, x,y binary. The LP root is (1,0.2): nearest and
       floor rounding fail, while ceiling rounding yields (1,1). */
    m.A.p[0] = 0; m.A.p[1] = 1; m.A.p[2] = 2;
    m.A.i[0] = 0; m.A.i[1] = 0;
    m.A.x[0] = 1.0; m.A.x[1] = 1.0;
    m.rlow[0] = 1.2; m.rupp[0] = SK_INFINITY;
    m.clow[0] = m.clow[1] = 0.0;
    m.cupp[0] = m.cupp[1] = 1.0;
    m.c[0] = 0.0; m.c[1] = 1.0;
    m.vartype[0] = m.vartype[1] = SK_INTEGER;

    sk_options_default(&o);
    o.time_limit = 2.0;
    sk_solution_init(&s);
    rc = sk_milp_solve(&m, &o, &s, &stats);
    if (rc != SK_OK || s.result != SK_RESULT_OPTIMAL || !s.x ||
        fabs(s.objective - 1.0) > 1e-8 || stats.heuristic_hits < 1 ||
        sk_verify(&m, &s) != SK_OK || s.primal_infeasibility > 1e-8) {
        sk_solution_free(&s); sk_model_free(&m); return 1;
    }
    sk_solution_free(&s);
    sk_model_free(&m);
    return 0;
}
