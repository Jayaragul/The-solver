#include "sankhya.h"

#include <math.h>
#include <stdlib.h>

/* A 13-variable convex QP is intentionally just above the exhaustive
 * active-pattern cap.  The bounded working-set fallback must recover the
 * exact solution at a variable bound without relying on enumeration. */
int main(void)
{
    enum { n = 13, qnnz = 15 };
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

    /* Symmetric PSD block [[2,1],[1,2]], then eleven independent diagonal
       entries.  The linear term makes x=(1/2,0,...,0) optimal. */
    m.Q->p[0] = 0;
    m.Q->i[k] = 0; m.Q->x[k++] = 2.0;
    m.Q->i[k] = 1; m.Q->x[k++] = 1.0;
    m.Q->p[1] = k;
    m.Q->i[k] = 0; m.Q->x[k++] = 1.0;
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
    if (sk_solve(&m, &o, &s) != SK_OK || s.result != SK_RESULT_OPTIMAL ||
        s.iterations == 0 || fabs(s.x[0] - 0.5) > 1e-7 ||
        fabs(s.x[1]) > 1e-7 || fabs(s.objective + 0.25) > 1e-7 ||
        sk_verify(&m, &s) != SK_OK || s.primal_infeasibility > 1e-7 ||
        s.dual_infeasibility > 1e-7 || s.complementarity > 1e-7) {
        sk_solution_free(&s); sk_model_free(&m); return 2;
    }
    sk_solution_free(&s);
    sk_model_free(&m);
    return 0;
}
