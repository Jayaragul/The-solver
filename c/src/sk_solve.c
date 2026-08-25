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
#if defined(SANKHYA_HAS_OPENMP)
#include <omp.h>
#endif

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
    int j;
#if defined(SANKHYA_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (j = 0; j < a->ncol; ++j) {
        double sum = 0.0;
        int p;
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

/* Exact closed form for the separable case
 *   min 1/2 sum(q_j x_j^2) + c_j x_j,  clow <= x <= cupp.
 * Keeping this outside PDHG removes avoidable iteration work from common
 * diagonal-QP kernels while retaining the independent verifier as the final
 * acceptance gate.  Return 1 when solved, -1 when unbounded, and 0 when the
 * model is not in the supported structure. */
static int qp_diagonal_unconstrained(const sk_model *m, const sk_options *o,
                                     sk_solution *s)
{
    int j, p, diagonal = 1;
    double *diag = NULL;
    if (!m->Q || m->A.p[m->ncol] != 0) return 0;
    diag = (double *)calloc((size_t)m->ncol, sizeof(double));
    if (!diag && m->ncol) return 0;
    for (j = 0; j < m->ncol; ++j) for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p) {
        if (m->Q->i[p] != j) { diagonal = 0; break; }
        diag[j] += m->Q->x[p];
    }
    if (!diagonal) { free(diag); return 0; }
    for (j = 0; j < m->ncol; ++j) if (diag[j] < -1e-12) {
        free(diag); return 0; /* non-convex: leave the general path in charge */
    }
    sk_solution_init(s);
    s->x = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    s->y = (double *)calloc((size_t)m->nrow + 1, sizeof(double));
    if ((!s->x && m->ncol) || (!s->y && m->nrow)) { sk_solution_free(s); free(diag); return 0; }
    s->ncol = m->ncol; s->nrow = m->nrow; s->iterations = 0;
    for (j = 0; j < m->ncol; ++j) {
        double v;
        if (diag[j] > 1e-14) v = -m->c[j] / diag[j];
        else if (m->c[j] > 1e-14) {
            if (SK_IS_NEG_INF(m->clow[j])) { sk_solution_free(s); free(diag); return -1; }
            v = m->clow[j];
        } else if (m->c[j] < -1e-14) {
            if (SK_IS_INF(m->cupp[j])) { sk_solution_free(s); free(diag); return -1; }
            v = m->cupp[j];
        } else v = clamp_value(0.0, m->clow[j], m->cupp[j]);
        s->x[j] = clamp_value(v, m->clow[j], m->cupp[j]);
    }
    if (sk_verify(m, s) != SK_OK || s->primal_infeasibility > 100.0 * o->primal_tol ||
        s->dual_infeasibility > 100.0 * o->dual_tol ||
        s->complementarity > 100.0 * o->dual_tol) {
        sk_solution_free(s); free(diag); return 0;
    }
    s->dual_bound = s->objective;
    s->mip_gap = 0.0;
    s->result = SK_RESULT_OPTIMAL;
    free(diag);
    return 1;
}

static int qp_dense_solve(double *a, double *b, int n);

static int qp_psd_check(const sk_model *m)
{
    int n = m->ncol, i, j, k, p;
    double *h = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    double *l = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    if ((!h && n) || (!l && n)) { free(h); free(l); return 0; }
    for (j = 0; j < n; ++j) for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p) {
        if (!isfinite(m->Q->x[p])) { free(h); free(l); return 0; }
        h[(size_t)m->Q->i[p] * (size_t)n + (size_t)j] += m->Q->x[p];
    }
    /* Work with the symmetric part; the public Q contract is symmetric, but
       averaging also makes the guard conservative for hand-built models. */
    for (i = 0; i < n; ++i) for (j = i; j < n; ++j) {
        double v = 0.5 * (h[(size_t)i * n + j] + h[(size_t)j * n + i]);
        h[(size_t)i * n + j] = h[(size_t)j * n + i] = v;
    }
    for (i = 0; i < n; ++i) {
        double d = h[(size_t)i * n + i];
        for (k = 0; k < i; ++k) d -= l[(size_t)i * n + k] * l[(size_t)i * n + k];
        if (d < -1e-10 * (1.0 + fabs(h[(size_t)i * n + i]))) { free(h); free(l); return 0; }
        if (d > 1e-12 * (1.0 + fabs(h[(size_t)i * n + i]))) {
            l[(size_t)i * n + i] = sqrt(d);
            for (j = i + 1; j < n; ++j) {
                double v = h[(size_t)j * n + i];
                for (k = 0; k < i; ++k) v -= l[(size_t)j * n + k] * l[(size_t)i * n + k];
                l[(size_t)j * n + i] = v / l[(size_t)i * n + i];
            }
        } else {
            for (j = i + 1; j < n; ++j) {
                double v = h[(size_t)j * n + i];
                for (k = 0; k < i; ++k) v -= l[(size_t)j * n + k] * l[(size_t)i * n + k];
                if (fabs(v) > 1e-9 * (1.0 + fabs(h[(size_t)j * n + i]))) {
                    free(h); free(l); return 0;
                }
            }
        }
    }
    free(h); free(l); return 1;
}

/* Exact KKT solve for small unconstrained/equality-constrained QPs without
 * variable bounds.
 * This is deliberately guarded: redundant constraints or an indefinite
 * Hessian simply fall back to the general verified PDHG path. */
static int qp_equality_kkt(const sk_model *m, const sk_options *o, sk_solution *s)
{
    int i, j, p, dim, equality = 1;
    double *kmat = NULL, *rhs = NULL;
    if (!m->Q || m->ncol + m->nrow > 512) return 0;
    for (j = 0; j < m->ncol; ++j)
        if (!SK_IS_NEG_INF(m->clow[j]) || !SK_IS_INF(m->cupp[j])) return 0;
    for (i = 0; i < m->nrow; ++i)
        if (SK_IS_NEG_INF(m->rlow[i]) || SK_IS_INF(m->rupp[i]) ||
            fabs(m->rlow[i] - m->rupp[i]) > 1e-9) { equality = 0; break; }
    if (!equality || (m->nrow == 0 && m->A.p[m->ncol] != 0) || !qp_psd_check(m)) return 0;
    dim = m->ncol + m->nrow;
    kmat = (double *)calloc((size_t)dim * (size_t)dim, sizeof(double));
    rhs = (double *)calloc((size_t)dim, sizeof(double));
    if (!kmat || !rhs) { free(kmat); free(rhs); return 0; }
    for (j = 0; j < m->ncol; ++j) {
        rhs[j] = -m->c[j];
        for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p)
            kmat[j * dim + m->Q->i[p]] += m->Q->x[p];
        for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) {
            i = m->A.i[p];
            kmat[j * dim + m->ncol + i] += m->A.x[p];
            kmat[(m->ncol + i) * dim + j] += m->A.x[p];
        }
    }
    for (i = 0; i < m->nrow; ++i) rhs[m->ncol + i] = m->rlow[i];
    if (!qp_dense_solve(kmat, rhs, dim)) { free(kmat); free(rhs); return 0; }
    sk_solution_init(s);
    s->x = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    s->y = (double *)calloc((size_t)m->nrow + 1, sizeof(double));
    if ((!s->x && m->ncol) || (!s->y && m->nrow)) { sk_solution_free(s); free(kmat); free(rhs); return 0; }
    memcpy(s->x, rhs, (size_t)m->ncol * sizeof(double));
    memcpy(s->y, rhs + m->ncol, (size_t)m->nrow * sizeof(double));
    s->ncol = m->ncol; s->nrow = m->nrow; s->iterations = 0;
    if (sk_verify(m, s) != SK_OK || s->primal_infeasibility > 100.0 * o->primal_tol ||
        s->dual_infeasibility > 100.0 * o->dual_tol ||
        s->complementarity > 100.0 * o->dual_tol) {
        sk_solution_free(s); free(kmat); free(rhs); return 0;
    }
    s->dual_bound = s->objective; s->mip_gap = 0.0; s->result = SK_RESULT_OPTIMAL;
    free(kmat); free(rhs);
    return 1;
}

/* Small-problem active-set polish for the first-order QP path.  PDHG is
 * deliberately retained for large sparse models, but on a small model a few
 * dense KKT corrections remove the long feasibility tail common to QPS
 * instances.  The correction is accepted only through the normal independent
 * KKT verifier below; it is never an unverified optimality shortcut. */
static int qp_dense_solve(double *a, double *b, int n)
{
    int k, i, j, pivot;
    for (k = 0; k < n; ++k) {
        double best = fabs(a[k * n + k]);
        pivot = k;
        for (i = k + 1; i < n; ++i) {
            double v = fabs(a[i * n + k]);
            if (v > best) { best = v; pivot = i; }
        }
        if (best < 1e-12) return 0;
        if (pivot != k) {
            for (j = k; j < n; ++j) {
                double t = a[k * n + j]; a[k * n + j] = a[pivot * n + j]; a[pivot * n + j] = t;
            }
            { double t = b[k]; b[k] = b[pivot]; b[pivot] = t; }
        }
        for (i = k + 1; i < n; ++i) {
            double f = a[i * n + k] / a[k * n + k];
            if (f == 0.0) continue;
            a[i * n + k] = 0.0;
            for (j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (i = n - 1; i >= 0; --i) {
        double v = b[i];
        for (j = i + 1; j < n; ++j) v -= a[i * n + j] * b[j];
        b[i] = v / a[i * n + i];
    }
    return 1;
}

static double qp_kkt_score(const sk_model *m, const double *x, const double *y)
{
    sk_solution probe;
    sk_solution_init(&probe);
    probe.x = (double *)x;
    probe.y = (double *)y;
    if (sk_verify(m, &probe) != SK_OK) return INFINITY;
    return probe.primal_infeasibility + probe.dual_infeasibility + probe.complementarity;
}

static void qp_active_polish(const sk_model *m, double *x, double *y)
{
    int n = m->ncol, r = m->nrow, max_active, iter, all_equalities = 1;
    int *free_var = NULL, *active_row = NULL;
    double *act = NULL, *qx = NULL, *grad = NULL, *kmat = NULL, *rhs = NULL;
    double *orig_x = NULL, *orig_y = NULL, before, after;

    if (!m->Q || n > 512 || r > 512) return;
    max_active = r + n;
    for (iter = 0; iter < r; ++iter)
        if (SK_IS_NEG_INF(m->rlow[iter]) || SK_IS_INF(m->rupp[iter]) ||
            fabs(m->rlow[iter] - m->rupp[iter]) > 1e-9) { all_equalities = 0; break; }
    free_var = (int *)malloc((size_t)n * sizeof(int));
    active_row = (int *)malloc((size_t)r * sizeof(int));
    act = (double *)calloc((size_t)r, sizeof(double));
    qx = (double *)calloc((size_t)n, sizeof(double));
    grad = (double *)calloc((size_t)n, sizeof(double));
    kmat = (double *)calloc((size_t)max_active * (size_t)max_active, sizeof(double));
    rhs = (double *)calloc((size_t)max_active, sizeof(double));
    orig_x = (double *)malloc((size_t)n * sizeof(double));
    orig_y = (double *)malloc((size_t)r * sizeof(double));
    if (!free_var || !active_row || (!act && r) || !qx || !grad || !kmat || !rhs || !orig_x || (!orig_y && r)) goto done;
    memcpy(orig_x, x, (size_t)n * sizeof(double));
    memcpy(orig_y, y, (size_t)r * sizeof(double));
    {
        double pinf = 0.0;
        int i, j, p;
        memset(act, 0, (size_t)r * sizeof(double));
        for (j = 0; j < n; ++j) for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) act[m->A.i[p]] += m->A.x[p] * x[j];
        for (i = 0; i < r; ++i) {
            if (!SK_IS_NEG_INF(m->rlow[i]) && m->rlow[i] - act[i] > pinf) pinf = m->rlow[i] - act[i];
            if (!SK_IS_INF(m->rupp[i]) && act[i] - m->rupp[i] > pinf) pinf = act[i] - m->rupp[i];
        }
        for (j = 0; j < n; ++j) {
            if (!SK_IS_NEG_INF(m->clow[j]) && m->clow[j] - x[j] > pinf) pinf = m->clow[j] - x[j];
            if (!SK_IS_INF(m->cupp[j]) && x[j] - m->cupp[j] > pinf) pinf = x[j] - m->cupp[j];
        }
        if (pinf > 1e-5 && !all_equalities) goto done;
    }
    before = qp_kkt_score(m, x, y);

    for (iter = 0; iter < 8; ++iter) {
        int i, j, p, nf = 0, na = 0, dim;
        double max_step = 0.0;
        memset(act, 0, (size_t)r * sizeof(double));
        memset(qx, 0, (size_t)n * sizeof(double));
        for (j = 0; j < n; ++j) {
            for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) act[m->A.i[p]] += m->A.x[p] * x[j];
            for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p) qx[m->Q->i[p]] += m->Q->x[p] * x[j];
        }
        for (j = 0; j < n; ++j) {
            int atlo = !SK_IS_NEG_INF(m->clow[j]) && fabs(x[j] - m->clow[j]) <= 1e-7;
            int athi = !SK_IS_INF(m->cupp[j]) && fabs(x[j] - m->cupp[j]) <= 1e-7;
            if (all_equalities || (!atlo && !athi)) free_var[nf++] = j;
        }
        for (i = 0; i < r; ++i) {
            int equality = !SK_IS_NEG_INF(m->rlow[i]) && !SK_IS_INF(m->rupp[i]) &&
                           fabs(m->rlow[i] - m->rupp[i]) <= 1e-9;
            int lower = !SK_IS_NEG_INF(m->rlow[i]) &&
                        (equality || fabs(act[i] - m->rlow[i]) <= 1e-4 * (1.0 + fabs(m->rlow[i])) ||
                         act[i] < m->rlow[i]);
            int upper = !SK_IS_INF(m->rupp[i]) &&
                        (equality || fabs(act[i] - m->rupp[i]) <= 1e-4 * (1.0 + fabs(m->rupp[i])) ||
                         act[i] > m->rupp[i]);
            if (lower || upper) active_row[na++] = i;
        }
        dim = nf + na;
        if (dim == 0 || dim > max_active) break;
        memset(kmat, 0, (size_t)dim * (size_t)dim * sizeof(double));
        memset(rhs, 0, (size_t)dim * sizeof(double));
        for (i = 0; i < nf; ++i) {
            int v = free_var[i];
            grad[v] = m->c[v] + qx[v];
            for (p = m->A.p[v]; p < m->A.p[v + 1]; ++p) grad[v] += m->A.x[p] * y[m->A.i[p]];
            rhs[i] = -grad[v];
            for (p = m->Q->p[v]; p < m->Q->p[v + 1]; ++p) {
                int row = m->Q->i[p], q;
                for (q = 0; q < nf; ++q) if (free_var[q] == row) kmat[i * dim + q] += m->Q->x[p];
            }
        }
        for (j = 0; j < na; ++j) {
            int row = active_row[j];
            rhs[nf + j] = (!SK_IS_NEG_INF(m->rlow[row]) &&
                           (SK_IS_INF(m->rupp[row]) || fabs(act[row] - m->rlow[row]) <= fabs(act[row] - m->rupp[row])))
                          ? m->rlow[row] - act[row] : m->rupp[row] - act[row];
            for (i = 0; i < nf; ++i) {
                int v = free_var[i], p0;
                for (p0 = m->A.p[v]; p0 < m->A.p[v + 1]; ++p0) if (m->A.i[p0] == row) {
                    kmat[i * dim + nf + j] += m->A.x[p0];
                    kmat[(nf + j) * dim + i] += m->A.x[p0];
                }
            }
        }
        if (!qp_dense_solve(kmat, rhs, dim)) break;
        memset(y, 0, (size_t)r * sizeof(double));
        for (i = 0; i < nf; ++i) {
            x[free_var[i]] += rhs[i];
            if (!SK_IS_NEG_INF(m->clow[free_var[i]]) && x[free_var[i]] < m->clow[free_var[i]]) x[free_var[i]] = m->clow[free_var[i]];
            if (!SK_IS_INF(m->cupp[free_var[i]]) && x[free_var[i]] > m->cupp[free_var[i]]) x[free_var[i]] = m->cupp[free_var[i]];
            if (fabs(rhs[i]) > max_step) max_step = fabs(rhs[i]);
        }
        for (j = 0; j < na; ++j) y[active_row[j]] = orig_y[active_row[j]] + rhs[nf + j];
        if (max_step <= 1e-9) break;
    }
    after = qp_kkt_score(m, x, y);
    if (!(after < before * (1.0 - 1e-6) && after <= 1e-5)) {
        memcpy(x, orig_x, (size_t)n * sizeof(double));
        memcpy(y, orig_y, (size_t)r * sizeof(double));
    }

done:
    free(free_var); free(active_row); free(act); free(qx); free(grad); free(kmat); free(rhs);
    free(orig_x); free(orig_y);
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
    double *activity = NULL, *gradient = NULL, *qx = NULL, *qdiag = NULL;
    double *tau_vec = NULL, *sigma_vec = NULL;
    double anorm, qnorm, start;
    int iteration, converged = 0;
    double dual_step = INFINITY;

    if (!m || !s || n < 0 || r < 0 || !m->c || !m->clow || !m->cupp ||
        !m->rlow || !m->rupp || !m->A.p || (m->Q && !m->Q->p)) return SK_ERR_ARG;
    if (m->Q && (m->Q->nrow != n || m->Q->ncol != n)) return SK_ERR_UNSUPPORTED;
    if (!o) { sk_options_default(&defaults); o = &defaults; }
    if (o->primal_tol <= 0.0 || o->dual_tol <= 0.0) return SK_ERR_ARG;
#if defined(SANKHYA_HAS_OPENMP)
    if (o->threads > 0) omp_set_num_threads(o->threads);
#endif
    if (interval_infeasible(m, o->primal_tol)) {
        sk_solution_init(s);
        s->result = SK_RESULT_INFEASIBLE;
        s->ncol = n; s->nrow = r;
        s->primal_infeasibility = INFINITY;
        return SK_OK;
    }
    /* KKT residuals certify a global minimum only for convex QPs.  For small
       Hessians we can prove PSD directly; reject indefinite input rather than
       returning a merely stationary PDHG point as `optimal`. */
    if (m->Q && n <= 512 && !qp_psd_check(m)) return SK_ERR_UNSUPPORTED;
    if (m->Q) {
        int fast = qp_diagonal_unconstrained(m, o, s);
        if (fast == 1) return SK_OK;
        if (fast < 0) {
            sk_solution_init(s);
            s->result = SK_RESULT_UNBOUNDED;
            s->ncol = n; s->nrow = r;
            return SK_OK;
        }
        if (qp_equality_kkt(m, o, s)) return SK_OK;
    }

    x = (double *)calloc((size_t)n, sizeof(double));
    x_new = (double *)calloc((size_t)n, sizeof(double));
    x_bar = (double *)calloc((size_t)n, sizeof(double));
    y = (double *)calloc((size_t)r, sizeof(double));
    y_new = (double *)calloc((size_t)r, sizeof(double));
    activity = (double *)calloc((size_t)r, sizeof(double));
    gradient = (double *)calloc((size_t)n, sizeof(double));
    if (m->Q) qx = (double *)calloc((size_t)n, sizeof(double));
    if (m->Q) qdiag = (double *)calloc((size_t)n, sizeof(double));
    tau_vec = (double *)calloc((size_t)n, sizeof(double));
    sigma_vec = (double *)calloc((size_t)r, sizeof(double));
    if ((!x && n) || (!x_new && n) || (!x_bar && n) || (!y && r) || (!y_new && r) ||
        (!activity && r) || (!gradient && n) || (m->Q && (!qx || !qdiag)) ||
        (!tau_vec && n) || (!sigma_vec && r)) goto memory_failure;
    {
        int diagonal = m->Q != NULL, j, p;
        if (m->Q) for (j = 0; j < n; ++j) for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p) {
            if (m->Q->i[p] != j) { diagonal = 0; break; }
            qdiag[j] += m->Q->x[p];
        }
        if (!diagonal) { free(qdiag); qdiag = NULL; }
    }
    for (iteration = 0; iteration < n; ++iteration) x[iteration] = x_bar[iteration] = clamp_value(0.0, m->clow[iteration], m->cupp[iteration]);

    anorm = matrix_norm_bound(&m->A);
    qnorm = matrix_norm_bound(m->Q);
    if (!isfinite(anorm) || !isfinite(qnorm)) goto memory_failure;
    /* Diagonal preconditioning: each row and variable gets a step based on
       its local absolute operator mass.  The 0.5 safety factor keeps the
       primal-dual product conservative while avoiding a single large row or
       Hessian column throttling the entire sparse QP. */
    {
        double *row_mass = (double *)calloc((size_t)r, sizeof(double));
        double *col_mass = (double *)calloc((size_t)n, sizeof(double));
        int j, p;
        if ((!row_mass && r) || (!col_mass && n)) { free(row_mass); free(col_mass); goto memory_failure; }
        for (j = 0; j < n; ++j) {
            for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) {
                row_mass[m->A.i[p]] += fabs(m->A.x[p]);
                col_mass[j] += fabs(m->A.x[p]);
            }
            if (m->Q) for (p = m->Q->p[j]; p < m->Q->p[j + 1]; ++p)
                col_mass[j] += fabs(m->Q->x[p]);
        }
        for (j = 0; j < n; ++j) tau_vec[j] = 0.9 / (1.0 + col_mass[j]);
        for (j = 0; j < r; ++j) sigma_vec[j] = 0.9 / (1.0 + row_mass[j]);
        free(row_mass); free(col_mass);
    }
    start = sk_wall_seconds();
    for (iteration = 1; iteration <= maximum_iterations; ++iteration) {
        int j;
        csc_mv(&m->A, x_bar, activity);
        for (j = 0; j < r; ++j) {
            const double trial = y[j] + sigma_vec[j] * activity[j];
            y_new[j] = trial - sigma_vec[j] * clamp_value(trial / sigma_vec[j], m->rlow[j], m->rupp[j]);
        }
        dual_step = 0.0;
        for (j = 0; j < r; ++j) if (fabs(y_new[j] - y[j]) > dual_step) dual_step = fabs(y_new[j] - y[j]);
        csc_tmv(&m->A, y_new, gradient);
        if (m->Q) {
            if (qdiag) for (j = 0; j < n; ++j) qx[j] = qdiag[j] * x[j];
            else csc_mv(m->Q, x, qx);
        }
        for (j = 0; j < n; ++j) {
            if (qdiag) {
                /* Implicit diagonal-Q prox: this is more stable than an
                   explicit Qx step when a QP Hessian is badly scaled. */
                x_new[j] = (x[j] - tau_vec[j] * (m->c[j] + gradient[j])) /
                           (1.0 + tau_vec[j] * qdiag[j]);
            } else {
                x_new[j] = x[j] - tau_vec[j] *
                           (m->c[j] + (m->Q ? qx[j] : 0.0) + gradient[j]);
            }
            x_new[j] = clamp_value(x_new[j], m->clow[j], m->cupp[j]);
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

    if (m->Q) qp_active_polish(m, x, y);

    sk_solution_init(s);
    s->x = x; x = NULL;
    s->y = y; y = NULL;
    s->rowact = activity; activity = NULL;
    s->ncol = n; s->nrow = r;
    s->iterations = iteration;
    s->solve_seconds = sk_wall_seconds() - start;
    if (m->Q) {
        if (qdiag) for (iteration = 0; iteration < n; ++iteration) qx[iteration] = qdiag[iteration] * s->x[iteration];
        else csc_mv(m->Q, s->x, qx);
    }
    s->objective = m->objshift;
    for (iteration = 0; iteration < n; ++iteration)
        s->objective += m->c[iteration] * s->x[iteration] + (m->Q ? 0.5 * s->x[iteration] * qx[iteration] : 0.0);
    s->primal_infeasibility = row_violation(m, s->rowact);
    /* QP uses the independently recomputed KKT residual. LP PDHG uses a
       different multiplier sign convention from the legacy LP verifier, so
       its certificate is intentionally deferred to the simplex path. */
    if (!isfinite(s->objective) || !isfinite(s->primal_infeasibility)) {
        s->result = SK_RESULT_NUMERIC_FAILURE;
    } else if (m->Q) {
        if (sk_verify(m, s) == SK_OK && s->primal_infeasibility <= 100.0 * o->primal_tol &&
            s->dual_infeasibility <= 100.0 * o->dual_tol && s->complementarity <= 100.0 * o->dual_tol)
            converged = 1;
    }
    if (s->result != SK_RESULT_NUMERIC_FAILURE && converged && (!m->Q || (s->primal_infeasibility <= 100.0 * o->primal_tol &&
        s->dual_infeasibility <= 100.0 * o->dual_tol && s->complementarity <= 100.0 * o->dual_tol))) {
        s->result = SK_RESULT_OPTIMAL;
    } else {
        s->result = o->time_limit > 0.0 && s->solve_seconds >= o->time_limit
            ? SK_RESULT_TIME_LIMIT : SK_RESULT_ITERATION_LIMIT;
    }
    free(x_new); free(x_bar); free(y_new); free(gradient); free(qx); free(qdiag); free(tau_vec); free(sigma_vec);
    return SK_OK;

memory_failure:
    free(x); free(x_new); free(x_bar); free(y); free(y_new); free(activity); free(gradient); free(qx);
    free(qdiag); free(tau_vec); free(sigma_vec);
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
