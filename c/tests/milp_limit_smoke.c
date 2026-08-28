#include "sankhya.h"

#include <math.h>

/* An interrupted branch-and-bound must still report a *valid* dual bound.
 *
 * The instance is a 4-item binary knapsack whose optimum is -20, taken by
 * items 2 and 3 (weights 5 + 2 = 7, exactly the capacity).
 *
 * The assertion is the invariant, not a node count: however the search is
 * truncated, the dual bound it reports must never exceed the true optimum,
 * and any incumbent it reports must never beat it.  An earlier version of
 * this test pinned the exact node at which -20 appeared, which made it fail
 * the moment branching got *better* and closed the tree sooner - it was
 * measuring the search's shape rather than its correctness.
 */
static void build(sk_model *m)
{
    sk_model_init(m);
    if (sk_model_alloc(m, 1, 4, 4) != SK_OK) return;
    m->A.p[0] = 0; m->A.p[1] = 1; m->A.p[2] = 2; m->A.p[3] = 3; m->A.p[4] = 4;
    m->A.i[0] = m->A.i[1] = m->A.i[2] = m->A.i[3] = 0;
    m->A.x[0] = 3.0; m->A.x[1] = 5.0; m->A.x[2] = 2.0; m->A.x[3] = 4.0;
    m->c[0] = -10.0; m->c[1] = -13.0; m->c[2] = -7.0; m->c[3] = -9.0;
    m->rlow[0] = -SK_INFINITY; m->rupp[0] = 7.0;
    m->clow[0] = m->clow[1] = m->clow[2] = m->clow[3] = 0.0;
    m->cupp[0] = m->cupp[1] = m->cupp[2] = m->cupp[3] = 1.0;
    m->vartype[0] = m->vartype[1] = m->vartype[2] = m->vartype[3] = SK_INTEGER;
}

#define OPTIMUM (-20.0)

int main(void)
{
    sk_model m;
    sk_solution s;
    sk_options o;
    long long limits[] = { 1, 2, 5, 0 };   /* 0 = no limit */
    int k;

    for (k = 0; k < 4; k++) {
        build(&m);
        if (m.ncol != 4) return 1;
        sk_solution_init(&s);
        sk_options_default(&o);
        o.node_limit = limits[k];

        if (sk_solve(&m, &o, &s) != SK_OK) { sk_solution_free(&s); sk_model_free(&m); return 2; }

        /* A dual bound above the true optimum would mean the search discarded
           the region containing it -- the failure mode that lets a solver
           report "optimal" on a suboptimal point. */
        if (isfinite(s.dual_bound) && s.dual_bound > OPTIMUM + 1e-6) {
            sk_solution_free(&s); sk_model_free(&m); return 3;
        }
        /* No incumbent may beat the optimum: that would mean an infeasible
           point was accepted. */
        if (s.x && isfinite(s.objective) && s.objective < OPTIMUM - 1e-6) {
            sk_solution_free(&s); sk_model_free(&m); return 4;
        }
        /* Given no limit, the search must close the tree exactly. */
        if (limits[k] == 0) {
            if (s.result != SK_RESULT_OPTIMAL ||
                fabs(s.objective - OPTIMUM) > 1e-6 ||
                fabs(s.dual_bound - OPTIMUM) > 1e-6) {
                sk_solution_free(&s); sk_model_free(&m); return 5;
            }
        }
        sk_solution_free(&s);
        sk_model_free(&m);
    }
    return 0;
}
