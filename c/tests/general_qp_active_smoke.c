#include "sankhya.h"

#include <math.h>
#include <stdlib.h>

int main(void)
{
    sk_model m;
    sk_solution s;
    sk_options o;
    sk_model_init(&m);
    if (sk_model_alloc(&m, 1, 2, 2) != SK_OK) return 1;
    m.A.p[0] = 0; m.A.p[1] = 1; m.A.p[2] = 2;
    m.A.i[0] = m.A.i[1] = 0;
    m.A.x[0] = m.A.x[1] = 1.0;
    m.rlow[0] = -SK_INFINITY; m.rupp[0] = 1.0;
    m.clow[0] = m.clow[1] = 0.0;
    m.cupp[0] = m.cupp[1] = 10.0;
    m.c[0] = m.c[1] = -3.0;
    m.Q = (sk_csc *)calloc(1, sizeof(*m.Q));
    if (!m.Q) { sk_model_free(&m); return 1; }
    m.Q->nrow = m.Q->ncol = 2; m.Q->nzmax = 4;
    m.Q->p = (int *)calloc(3, sizeof(int));
    m.Q->i = (int *)calloc(4, sizeof(int));
    m.Q->x = (double *)calloc(4, sizeof(double));
    if (!m.Q->p || !m.Q->i || !m.Q->x) { sk_model_free(&m); return 1; }
    m.Q->p[0] = 0; m.Q->p[1] = 2; m.Q->p[2] = 4;
    m.Q->i[0] = 0; m.Q->i[1] = 1; m.Q->i[2] = 0; m.Q->i[3] = 1;
    m.Q->x[0] = 2.0; m.Q->x[1] = 1.0; m.Q->x[2] = 1.0; m.Q->x[3] = 2.0;
    sk_solution_init(&s); sk_options_default(&o);
    if (sk_solve(&m, &o, &s) != SK_OK || s.result != SK_RESULT_OPTIMAL ||
        s.iterations != 0 || fabs(s.x[0] - 0.5) > 1e-10 ||
        fabs(s.x[1] - 0.5) > 1e-10 || fabs(s.objective + 2.25) > 1e-10 ||
        sk_verify(&m, &s) != SK_OK || s.primal_infeasibility > 1e-10 ||
        s.dual_infeasibility > 1e-10 || s.complementarity > 1e-10) {
        sk_solution_free(&s); sk_model_free(&m); return 2;
    }
    sk_solution_free(&s); sk_model_free(&m); return 0;
}
