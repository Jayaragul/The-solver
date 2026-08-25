#include "sankhya.h"

#include <stdlib.h>

int main(void)
{
    sk_model m;
    sk_solution s;
    sk_options o;
    sk_model_init(&m);
    if (sk_model_alloc(&m, 0, 2, 0) != SK_OK) return 1;
    m.clow[0] = m.clow[1] = -SK_INFINITY;
    m.cupp[0] = m.cupp[1] = SK_INFINITY;
    m.Q = (sk_csc *)calloc(1, sizeof(*m.Q));
    if (!m.Q) { sk_model_free(&m); return 1; }
    m.Q->nrow = m.Q->ncol = 2; m.Q->nzmax = 3;
    m.Q->p = (int *)calloc(3, sizeof(int));
    m.Q->i = (int *)calloc(3, sizeof(int));
    m.Q->x = (double *)calloc(3, sizeof(double));
    if (!m.Q->p || !m.Q->i || !m.Q->x) { sk_model_free(&m); return 1; }
    /* Q = [[2, 0], [1, 2]] is deliberately not symmetric. */
    m.Q->p[0] = 0; m.Q->p[1] = 1; m.Q->p[2] = 3;
    m.Q->i[0] = 0; m.Q->i[1] = 0; m.Q->i[2] = 1;
    m.Q->x[0] = 2.0; m.Q->x[1] = 1.0; m.Q->x[2] = 2.0;
    sk_solution_init(&s); sk_options_default(&o);
    if (sk_solve(&m, &o, &s) != SK_ERR_UNSUPPORTED) {
        sk_solution_free(&s); sk_model_free(&m); return 2;
    }
    sk_solution_free(&s); sk_model_free(&m); return 0;
}
