#include "sankhya.h"

#include <stdlib.h>

/* Regression for the public iteration-count contract: a QP that cannot take
 * an exact small-problem shortcut must report exactly the configured number
 * of executed PDHG steps when the budget is exhausted. */
int main(void)
{
    enum { n = 513, qnnz = 515 };
    sk_model m;
    sk_solution s;
    sk_options o;
    int j, k = 0;

    sk_model_init(&m);
    if (sk_model_alloc(&m, 0, n, 0) != SK_OK) return 1;
    m.Q = (sk_csc *)calloc(1, sizeof(*m.Q));
    if (!m.Q) { sk_model_free(&m); return 1; }
    m.Q->nrow = m.Q->ncol = n;
    m.Q->nzmax = qnnz;
    m.Q->p = (int *)calloc((size_t)n + 1, sizeof(int));
    m.Q->i = (int *)calloc(qnnz, sizeof(int));
    m.Q->x = (double *)calloc(qnnz, sizeof(double));
    if (!m.Q->p || !m.Q->i || !m.Q->x) { sk_model_free(&m); return 1; }
    /* Two symmetric off-diagonal entries force the guarded general QP path;
       the remaining diagonal is positive, so the Hessian is convex. */
    m.Q->p[0] = 0;
    m.Q->i[k] = 0; m.Q->x[k++] = 2.0;
    m.Q->i[k] = 1; m.Q->x[k++] = 0.1;
    m.Q->p[1] = k;
    m.Q->i[k] = 0; m.Q->x[k++] = 0.1;
    m.Q->i[k] = 1; m.Q->x[k++] = 2.0;
    m.Q->p[2] = k;
    for (j = 2; j < n; ++j) {
        m.Q->i[k] = j; m.Q->x[k++] = 2.0;
        m.Q->p[j + 1] = k;
    }
    for (j = 0; j < n; ++j) {
        m.c[j] = j == 0 ? -1.0 : 0.0;
        m.clow[j] = 0.0;
        m.cupp[j] = 1.0;
    }
    sk_solution_init(&s);
    sk_options_default(&o);
    o.iteration_limit = 1;
    if (sk_solve(&m, &o, &s) != SK_OK || s.result != SK_RESULT_ITERATION_LIMIT ||
        s.iterations != 1) {
        sk_solution_free(&s); sk_model_free(&m); return 2;
    }
    sk_solution_free(&s);
    sk_model_free(&m);
    return 0;
}
