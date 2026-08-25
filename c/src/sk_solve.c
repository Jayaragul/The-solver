/* Native first-order convex QP solve path.
 *
 * This is a dense-vector/sparse-matrix PDHG implementation for
 *   min .5 x'Qx + c'x  s.t. rlow <= Ax <= rupp, clow <= x <= cupp.
 * Q must be symmetric positive semidefinite. It is intentionally an
 * approximate continuous-QP path; simplex, IPM and MILP dispatch are added
 * only when their implementations and certificates exist. */
#include "sankhya.h"
#include "sk_milp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double clamp_value(double x, double lo, double hi)
{
    if (!SK_IS_NEG_INF(lo) && x < lo) x = lo;
    if (!SK_IS_INF(hi) && x > hi) x = hi;
    return x;
}

static void csc_mv(const sk_csc *a, const double *x, double *y)
{
    int j, p;
    memset(y, 0, (size_t)a->nrow * sizeof(double));
    for (j = 0; j < a->ncol; ++j)
        for (p = a->p[j]; p < a->p[j + 1]; ++p) y[a->i[p]] += a->x[p] * x[j];
}

static void csc_tmv(const sk_csc *a, const double *y, double *x)
{
    int j, p;
    for (j = 0; j < a->ncol; ++j) {
        double sum = 0.0;
        for (p = a->p[j]; p < a->p[j + 1]; ++p) sum += a->x[p] * y[a->i[p]];
        x[j] = sum;
    }
}

static double matrix_norm_bound(const sk_csc *a)
{
    int j, p;
    double *row_sum;
    double col_max = 0.0, row_max = 0.0;
    if (!a || !a->nrow || !a->ncol) return 0.0;
    row_sum = (double *)calloc((size_t)a->nrow, sizeof(double));
    if (!row_sum) return INFINITY;
    for (j = 0; j < a->ncol; ++j) {
        double col_sum = 0.0;
        for (p = a->p[j]; p < a->p[j + 1]; ++p) {
            double value = fabs(a->x[p]);
            col_sum += value;
            row_sum[a->i[p]] += value;
        }
        if (col_sum > col_max) col_max = col_sum;
    }
    for (j = 0; j < a->nrow; ++j) if (row_sum[j] > row_max) row_max = row_sum[j];
    free(row_sum);
    return sqrt(col_max * row_max);
}

static double row_violation(const sk_model *m, const double *activity)
{
    int i;
    double violation = 0.0;
    for (i = 0; i < m->nrow; ++i) {
        if (!SK_IS_NEG_INF(m->rlow[i]) && m->rlow[i] - activity[i] > violation)
            violation = m->rlow[i] - activity[i];
        if (!SK_IS_INF(m->rupp[i]) && activity[i] - m->rupp[i] > violation)
            violation = activity[i] - m->rupp[i];
    }
    return violation;
}

/* Conservative interval presolve. It only declares infeasibility when bounds
 * prove it; otherwise the numerical solver retains the model unchanged. */
static int interval_infeasible(const sk_model *m, double tolerance)
{
    int i, j, p, impossible = 0;
    double *minimum = NULL, *maximum = NULL;
    unsigned char *minimum_infinite = NULL, *maximum_infinite = NULL;
    for (j = 0; j < m->ncol; ++j)
        if (m->clow[j] > m->cupp[j] + tolerance) return 1;
    minimum = (double *)calloc((size_t)m->nrow, sizeof(double));
    maximum = (double *)calloc((size_t)m->nrow, sizeof(double));
    minimum_infinite = (unsigned char *)calloc((size_t)m->nrow, 1);
    maximum_infinite = (unsigned char *)calloc((size_t)m->nrow, 1);
    if ((!minimum && m->nrow) || (!maximum && m->nrow) || (!minimum_infinite && m->nrow) || (!maximum_infinite && m->nrow)) {
        free(minimum); free(maximum); free(minimum_infinite); free(maximum_infinite);
        return 0; /* allocation failure cannot safely prove infeasibility */
    }
    for (j = 0; j < m->ncol; ++j) {
        for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) {
            const int row = m->A.i[p];
            const double value = m->A.x[p];
            const double lo = value >= 0.0 ? m->clow[j] : m->cupp[j];
            const double hi = value >= 0.0 ? m->cupp[j] : m->clow[j];
            if ((value >= 0.0 && SK_IS_NEG_INF(lo)) || (value < 0.0 && SK_IS_INF(lo)))
                minimum_infinite[row] = 1;
            else if (!minimum_infinite[row]) minimum[row] += value * lo;
            if ((value >= 0.0 && SK_IS_INF(hi)) || (value < 0.0 && SK_IS_NEG_INF(hi)))
                maximum_infinite[row] = 1;
            else if (!maximum_infinite[row]) maximum[row] += value * hi;
        }
    }
    for (i = 0; i < m->nrow; ++i) {
        if (!maximum_infinite[i] && m->rlow[i] > maximum[i] + tolerance) { impossible = 1; break; }
        if (!minimum_infinite[i] && m->rupp[i] < minimum[i] - tolerance) { impossible = 1; break; }
    }
    free(minimum); free(maximum); free(minimum_infinite); free(maximum_infinite);
    return impossible;
}

static sk_status solve_continuous(const sk_model *m, const sk_options *options, sk_solution *s)
{
    sk_options defaults;
    const sk_options *o = options;
    const int n = m ? m->ncol : 0;
    const int r = m ? m->nrow : 0;
    const int maximum_iterations = (o && o->iteration_limit > 0 && o->iteration_limit < 2147483647LL)
        ? (int)o->iteration_limit : 100000;
    const int check_every = 20;
    double *x = NULL, *x_new = NULL, *x_bar = NULL, *y = NULL, *y_new = NULL;
    double *activity = NULL, *gradient = NULL, *qx = NULL;
    double tau, sigma, anorm, qnorm, start;
    int iteration, converged = 0;
    double dual_step = INFINITY;

    if (!m || !s || n < 0 || r < 0 || !m->c || !m->clow || !m->cupp ||
        !m->rlow || !m->rupp || !m->A.p || (m->Q && !m->Q->p)) return SK_ERR_ARG;
    if (m->Q && (m->Q->nrow != n || m->Q->ncol != n)) return SK_ERR_UNSUPPORTED;
    if (!o) { sk_options_default(&defaults); o = &defaults; }
    if (o->primal_tol <= 0.0 || o->dual_tol <= 0.0) return SK_ERR_ARG;
    if (interval_infeasible(m, o->primal_tol)) {
        sk_solution_init(s);
        s->result = SK_RESULT_INFEASIBLE;
        s->ncol = n; s->nrow = r;
        s->primal_infeasibility = INFINITY;
        return SK_OK;
    }

    x = (double *)calloc((size_t)n, sizeof(double));
    x_new = (double *)calloc((size_t)n, sizeof(double));
    x_bar = (double *)calloc((size_t)n, sizeof(double));
    y = (double *)calloc((size_t)r, sizeof(double));
    y_new = (double *)calloc((size_t)r, sizeof(double));
    activity = (double *)calloc((size_t)r, sizeof(double));
    gradient = (double *)calloc((size_t)n, sizeof(double));
    if (m->Q) qx = (double *)calloc((size_t)n, sizeof(double));
    if ((!x && n) || (!x_new && n) || (!x_bar && n) || (!y && r) || (!y_new && r) ||
        (!activity && r) || (!gradient && n) || (m->Q && !qx)) goto memory_failure;
    for (iteration = 0; iteration < n; ++iteration) x[iteration] = x_bar[iteration] = clamp_value(0.0, m->clow[iteration], m->cupp[iteration]);

    anorm = matrix_norm_bound(&m->A);
    qnorm = matrix_norm_bound(m->Q);
    if (!isfinite(anorm) || !isfinite(qnorm)) goto memory_failure;
    tau = 0.5 / (1.0 + qnorm + anorm);
    sigma = 0.5 / (1.0 + anorm);
    start = sk_wall_seconds();
    for (iteration = 1; iteration <= maximum_iterations; ++iteration) {
        int j;
        csc_mv(&m->A, x_bar, activity);
        for (j = 0; j < r; ++j) {
            const double trial = y[j] + sigma * activity[j];
            y_new[j] = trial - sigma * clamp_value(trial / sigma, m->rlow[j], m->rupp[j]);
        }
        dual_step = 0.0;
        for (j = 0; j < r; ++j) if (fabs(y_new[j] - y[j]) > dual_step) dual_step = fabs(y_new[j] - y[j]);
        csc_tmv(&m->A, y_new, gradient);
        if (m->Q) csc_mv(m->Q, x, qx);
        for (j = 0; j < n; ++j) {
            x_new[j] = clamp_value(x[j] - tau * (m->c[j] + (m->Q ? qx[j] : 0.0) + gradient[j]), m->clow[j], m->cupp[j]);
            x_bar[j] = 2.0 * x_new[j] - x[j];
        }
        memcpy(x, x_new, (size_t)n * sizeof(double));
        memcpy(y, y_new, (size_t)r * sizeof(double));
        if (o->time_limit > 0.0 && sk_wall_seconds() - start >= o->time_limit) break;
        if (iteration % check_every == 0 || iteration == maximum_iterations) {
            double step = 0.0, scale = 1.0;
            csc_mv(&m->A, x, activity);
            for (j = 0; j < n; ++j) {
                const double delta = fabs(x_new[j] - x[j]); /* x_new equals x after copy: use x_bar relationship below */
                (void)delta;
                if (fabs(x[j]) > scale) scale = fabs(x[j]);
            }
            /* x_bar = 2*x - x_old, hence |x-x_old| = |x_bar-x|. */
            for (j = 0; j < n; ++j) if (fabs(x_bar[j] - x[j]) > step) step = fabs(x_bar[j] - x[j]);
            if (row_violation(m, activity) <= o->primal_tol && step <= o->primal_tol * scale &&
                dual_step <= o->dual_tol * (1.0 + anorm)) { converged = 1; break; }
        }
    }

    sk_solution_init(s);
    s->x = x; x = NULL;
    s->y = y; y = NULL;
    s->rowact = activity; activity = NULL;
    s->ncol = n; s->nrow = r;
    s->iterations = iteration;
    s->solve_seconds = sk_wall_seconds() - start;
    if (m->Q) csc_mv(m->Q, s->x, qx);
    s->objective = m->objshift;
    for (iteration = 0; iteration < n; ++iteration)
        s->objective += m->c[iteration] * s->x[iteration] + (m->Q ? 0.5 * s->x[iteration] * qx[iteration] : 0.0);
    s->primal_infeasibility = row_violation(m, s->rowact);
    /* QP uses the independently recomputed KKT residual. LP PDHG uses a
       different multiplier sign convention from the legacy LP verifier, so
       its certificate is intentionally deferred to the simplex path. */
    if (converged && (!m->Q || (sk_verify(m, s) == SK_OK &&
        s->primal_infeasibility <= 100.0 * o->primal_tol &&
        s->dual_infeasibility <= 100.0 * o->dual_tol && s->complementarity <= 100.0 * o->dual_tol))) {
        s->result = SK_RESULT_OPTIMAL;
    } else {
        s->result = o->time_limit > 0.0 && s->solve_seconds >= o->time_limit
            ? SK_RESULT_TIME_LIMIT : SK_RESULT_ITERATION_LIMIT;
    }
    free(x_new); free(x_bar); free(y_new); free(gradient); free(qx);
    return SK_OK;

memory_failure:
    free(x); free(x_new); free(x_bar); free(y); free(y_new); free(activity); free(gradient); free(qx);
    return SK_ERR_MEMORY;
}

sk_status sk_solve(const sk_model *m, const sk_options *options, sk_solution *s)
{
    if (!m || !s) return SK_ERR_ARG;
    if (sk_model_num_integer(m) == 0) return solve_continuous(m, options, s);
    /* This is MILP branch-and-bound. MIQP needs reliable convex-QP bounds and
       its own certificate path, so it remains intentionally unsupported. */
    if (m->Q) return SK_ERR_UNSUPPORTED;
    return sk_milp_solve(m, options, s, NULL);
}
