/* Bounded-variable revised simplex.
 *
 * Formulation.  The model  min c'x, rlow <= Ax <= rupp, clow <= x <= cupp  is
 * put in equality form  M z = 0  with  M = [A  -I]  and  z = (x; s).  The
 * logical s_i then *is* the activity of row i and carries the row's bounds, so
 * the starting basis is the whole set of logicals and B = -I is trivially
 * factorizable.  Nonbasic variables sit at one of their own bounds instead of
 * being split into positive and negative parts, which keeps the basis m x m
 * regardless of how many two-sided bounds the model has.
 *
 * Phase 1 does not add artificial columns.  Instead each infeasible basic
 * variable is given a unit cost pointing back toward its violated bound and a
 * one-sided effective bound for the ratio test; minimizing that piecewise
 * objective drives the sum of primal infeasibilities to zero.  This keeps the
 * basis dimension and the sparsity of M unchanged between the two phases.
 *
 * Stability.  The ratio test is Harris' two-pass rule: pass one computes the
 * largest step allowed when every bound is relaxed by the feasibility
 * tolerance, pass two then picks, among rows admitted by that step, the one
 * with the largest pivot magnitude.  Trading a tiny bound violation for a
 * bigger pivot is what keeps the basis factorization well conditioned on
 * degenerate LPs.  Anti-cycling is by fallback to Bland's rule once a stall is
 * detected, which is finite-terminating.
 */
#include "sk_simplex.h"
#include "sk_lu.h"
#include "sk_scale.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPX_PIVOT_ZERO   1e-11
#define SPX_DROP_TOL     1e-14

typedef struct spx {
    const sk_model *m;
    int nrow, ncol, ntot;

    sk_csc M;              /* [A  -I], nrow x ntot                        */
    double *lb, *ub;       /* working bounds, length ntot                 */
    double *cost;          /* phase-2 cost, length ntot                   */

    int    *basis;         /* nrow: variable basic at each position       */
    signed char *stat;     /* ntot: SK_BASIC / SK_AT_LOWER / ...          */
    double *xval;          /* ntot: current value of every variable       */
    double *xB;            /* nrow: values of the basic variables         */

    sk_lu   lu;
    sk_csc  B;             /* scratch basis matrix for refactorization    */

    double *y;             /* nrow: duals                                 */
    double *cB;            /* nrow: basic costs of the active phase       */
    double *alpha;         /* nrow: FTRAN of the entering column          */
    double *rhs;           /* nrow: scratch                               */
    double *d;             /* ntot: reduced costs                         */

    double *R, *C;         /* row / column scale factors, powers of two   */
    double *w;             /* ntot: Devex reference weights               */
    double *rho;           /* nrow: pivot row scratch for the Devex update */
    int     devex_age;
    double  primal_tol, dual_tol, pivot_tol;
    int     refactor_interval;
    sk_spx_stats st;
} spx;

/* ---------------------------------------------------------------- helpers */

static void spx_free(spx *S)
{
    free(S->M.p); free(S->M.i); free(S->M.x);
    free(S->lb); free(S->ub); free(S->cost);
    free(S->basis); free(S->stat); free(S->xval); free(S->xB);
    free(S->y); free(S->cB); free(S->alpha); free(S->rhs); free(S->d);
    free(S->R); free(S->C);
    free(S->w); free(S->rho);
    free(S->B.p); free(S->B.i); free(S->B.x);
    sk_lu_free(&S->lu);
    memset(S, 0, sizeof(*S));
}

/* Tighten bounds implied by rows with exactly one nonzero coefficient. This
 * is deliberately non-mutating: simplex receives a shallow model copy whose
 * bounds belong to this routine, while the caller's model remains the
 * authority for verification and postsolve. */
static sk_status spx_singleton_presolve(const sk_model *m, const sk_options *o,
                                        sk_model *work, double **lower_out,
                                        double **upper_out, int *infeasible)
{
    double *lower = NULL, *upper = NULL;
    int *count = NULL, *column = NULL;
    double *coefficient = NULL;
    int i, j, p;

    *lower_out = NULL; *upper_out = NULL; *infeasible = 0;
    *work = *m;
    if (!o->presolve || m->ncol == 0 || m->nrow == 0) return SK_OK;
    lower = (double *)malloc((size_t)m->ncol * sizeof(double));
    upper = (double *)malloc((size_t)m->ncol * sizeof(double));
    count = (int *)calloc((size_t)m->nrow, sizeof(int));
    column = (int *)malloc((size_t)m->nrow * sizeof(int));
    coefficient = (double *)malloc((size_t)m->nrow * sizeof(double));
    if (!lower || !upper || !count || !column || !coefficient) {
        free(lower); free(upper); free(count); free(column); free(coefficient);
        return SK_ERR_MEMORY;
    }
    memcpy(lower, m->clow, (size_t)m->ncol * sizeof(double));
    memcpy(upper, m->cupp, (size_t)m->ncol * sizeof(double));
    for (j = 0; j < m->ncol; ++j) for (p = m->A.p[j]; p < m->A.p[j + 1]; ++p) {
        const int row = m->A.i[p];
        if (fabs(m->A.x[p]) <= SPX_DROP_TOL) continue;
        if (count[row]++ == 0) { column[row] = j; coefficient[row] = m->A.x[p]; }
    }
    for (i = 0; i < m->nrow; ++i) if (count[i] == 1) {
        const int col = column[i];
        const double a = coefficient[i];
        if (!SK_IS_NEG_INF(m->rlow[i])) {
            const double bound = m->rlow[i] / a;
            if (a > 0.0) { if (bound > lower[col]) lower[col] = bound; }
            else { if (bound < upper[col]) upper[col] = bound; }
        }
        if (!SK_IS_INF(m->rupp[i])) {
            const double bound = m->rupp[i] / a;
            if (a > 0.0) { if (bound < upper[col]) upper[col] = bound; }
            else { if (bound > lower[col]) lower[col] = bound; }
        }
        if (lower[col] > upper[col] + (o->primal_tol > 0.0 ? o->primal_tol : 1e-7)) {
            *infeasible = 1;
            break;
        }
    }
    free(count); free(column); free(coefficient);
    if (*infeasible) { free(lower); free(upper); return SK_OK; }
    work->clow = lower; work->cupp = upper;
    *lower_out = lower; *upper_out = upper;
    return SK_OK;
}

/* Build M = [A  -I] and the working bound/cost arrays. */
static sk_status spx_build(spx *S, const sk_model *m, const sk_options *o)
{
    int j, p, q = 0, nnz;
    const int nr = m->nrow, nc = m->ncol, nt = m->ncol + m->nrow;

    memset(S, 0, sizeof(*S));
    S->m = m; S->nrow = nr; S->ncol = nc; S->ntot = nt;
    S->primal_tol = o->primal_tol > 0 ? o->primal_tol : 1e-7;
    S->dual_tol   = o->dual_tol   > 0 ? o->dual_tol   : 1e-7;
    S->pivot_tol  = o->pivot_tol  > 0 ? o->pivot_tol  : 0.1;
    S->refactor_interval = o->refactor_interval > 0 ? o->refactor_interval : 100;
    sk_lu_init(&S->lu);

    nnz = m->A.p[nc] + nr;
    S->M.nrow = nr; S->M.ncol = nt; S->M.nzmax = nnz;
    S->M.p = (int *)malloc(((size_t)nt + 1) * sizeof(int));
    S->M.i = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    S->M.x = (double *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    S->lb   = (double *)malloc((size_t)nt * sizeof(double));
    S->ub   = (double *)malloc((size_t)nt * sizeof(double));
    S->cost = (double *)calloc((size_t)nt, sizeof(double));
    S->basis = (int *)malloc((size_t)(nr > 0 ? nr : 1) * sizeof(int));
    S->stat  = (signed char *)malloc((size_t)nt);
    S->xval  = (double *)calloc((size_t)nt, sizeof(double));
    S->xB    = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    S->y     = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    S->cB    = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    S->alpha = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    S->rhs   = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    S->d     = (double *)calloc((size_t)nt, sizeof(double));
    if (!S->M.p || !S->M.i || !S->M.x || !S->lb || !S->ub || !S->cost ||
        !S->basis || !S->stat || !S->xval || !S->xB || !S->y || !S->cB ||
        !S->alpha || !S->rhs || !S->d) { spx_free(S); return SK_ERR_MEMORY; }

    /* Work on the scaled model  (R A C) x~ = s~,  x = C x~,  y = R y~. */
    S->R = (double *)malloc((size_t)(nr > 0 ? nr : 1) * sizeof(double));
    S->C = (double *)malloc((size_t)(nc > 0 ? nc : 1) * sizeof(double));
    S->w = (double *)malloc((size_t)nt * sizeof(double));
    S->rho = (double *)calloc((size_t)(nr > 0 ? nr : 1), sizeof(double));
    if (!S->R || !S->C || !S->w || !S->rho) { spx_free(S); return SK_ERR_MEMORY; }
    for (j = 0; j < nt; j++) S->w[j] = 1.0;
    S->devex_age = 0;
    if (o->scaling) sk_scale_model(m, S->R, S->C, 4);
    else { for (j = 0; j < nr; j++) S->R[j] = 1.0; for (j = 0; j < nc; j++) S->C[j] = 1.0; }

    for (j = 0; j < nc; j++) {
        double cj = S->C[j];
        S->M.p[j] = q;
        for (p = m->A.p[j]; p < m->A.p[j + 1]; p++) {
            S->M.i[q] = m->A.i[p];
            S->M.x[q] = m->A.x[p] * S->R[m->A.i[p]] * cj;
            q++;
        }
        S->lb[j] = SK_IS_NEG_INF(m->clow[j]) ? -SK_INFINITY : m->clow[j] / cj;
        S->ub[j] = SK_IS_INF(m->cupp[j])     ?  SK_INFINITY : m->cupp[j] / cj;
        S->cost[j] = m->c[j] * cj;
    }
    for (j = 0; j < nr; j++) {
        double rj = S->R[j];
        S->M.p[nc + j] = q;
        S->M.i[q] = j; S->M.x[q] = -1.0; q++;
        S->lb[nc + j] = SK_IS_NEG_INF(m->rlow[j]) ? -SK_INFINITY : m->rlow[j] * rj;
        S->ub[nc + j] = SK_IS_INF(m->rupp[j])     ?  SK_INFINITY : m->rupp[j] * rj;
        S->cost[nc + j] = 0.0;
    }
    S->M.p[nt] = q;

    /* start: logicals basic, structurals at the bound nearest zero */
    for (j = 0; j < nc; j++) {
        if (!SK_IS_NEG_INF(S->lb[j]) && !SK_IS_INF(S->ub[j]))
            S->stat[j] = (fabs(S->lb[j]) <= fabs(S->ub[j])) ? SK_AT_LOWER : SK_AT_UPPER;
        else if (!SK_IS_NEG_INF(S->lb[j])) S->stat[j] = SK_AT_LOWER;
        else if (!SK_IS_INF(S->ub[j]))     S->stat[j] = SK_AT_UPPER;
        else                               S->stat[j] = SK_FREE_ZERO;
        S->xval[j] = (S->stat[j] == SK_AT_LOWER) ? S->lb[j]
                   : (S->stat[j] == SK_AT_UPPER) ? S->ub[j] : 0.0;
    }
    for (j = 0; j < nr; j++) { S->basis[j] = nc + j; S->stat[nc + j] = SK_BASIC; }

    /* room for the densest basis we might factorize */
    S->B.nrow = nr; S->B.ncol = nr;
    S->B.nzmax = nnz + nr + 1;
    S->B.p = (int *)malloc(((size_t)nr + 1) * sizeof(int));
    S->B.i = (int *)malloc((size_t)S->B.nzmax * sizeof(int));
    S->B.x = (double *)malloc((size_t)S->B.nzmax * sizeof(double));
    if (!S->B.p || !S->B.i || !S->B.x) { spx_free(S); return SK_ERR_MEMORY; }
    return SK_OK;
}

static void spx_basis_matrix(spx *S)
{
    int k, p, q = 0;
    for (k = 0; k < S->nrow; k++) {
        int v = S->basis[k];
        S->B.p[k] = q;
        for (p = S->M.p[v]; p < S->M.p[v + 1]; p++) { S->B.i[q] = S->M.i[p]; S->B.x[q] = S->M.x[p]; q++; }
    }
    S->B.p[S->nrow] = q;
}

/* xB = -B^{-1} * (sum over nonbasic j of M_j * x_j) */
static sk_status spx_recompute_xB(spx *S)
{
    int j, p, k;
    memset(S->rhs, 0, (size_t)S->nrow * sizeof(double));
    for (j = 0; j < S->ntot; j++) {
        double v;
        if (S->stat[j] == SK_BASIC) continue;
        v = S->xval[j];
        if (v == 0.0) continue;
        for (p = S->M.p[j]; p < S->M.p[j + 1]; p++) S->rhs[S->M.i[p]] -= S->M.x[p] * v;
    }
    if (sk_lu_ftran(&S->lu, S->rhs) != SK_OK) return SK_ERR_SINGULAR;
    for (k = 0; k < S->nrow; k++) { S->xB[k] = S->rhs[k]; S->xval[S->basis[k]] = S->rhs[k]; }
    return SK_OK;
}

static sk_status spx_refactorize(spx *S)
{
    sk_status st;
    spx_basis_matrix(S);
    st = sk_lu_factorize(&S->lu, &S->B, S->pivot_tol);
    if (st != SK_OK) return st;
    S->st.refactorizations++;
    return spx_recompute_xB(S);
}

/* Total primal infeasibility of the basic variables. */
static double spx_infeasibility(const spx *S)
{
    int k;
    double sum = 0.0;
    for (k = 0; k < S->nrow; k++) {
        int v = S->basis[k];
        if (!SK_IS_NEG_INF(S->lb[v]) && S->xB[k] < S->lb[v] - S->primal_tol) sum += S->lb[v] - S->xB[k];
        if (!SK_IS_INF(S->ub[v])     && S->xB[k] > S->ub[v] + S->primal_tol) sum += S->xB[k] - S->ub[v];
    }
    return sum;
}

/* Basic costs for the active phase. Phase 1 points each infeasible basic
 * variable back toward the bound it violates; feasible ones cost nothing. */
static void spx_build_cB(spx *S, int phase1)
{
    int k;
    for (k = 0; k < S->nrow; k++) {
        int v = S->basis[k];
        if (!phase1) { S->cB[k] = S->cost[v]; continue; }
        if (!SK_IS_NEG_INF(S->lb[v]) && S->xB[k] < S->lb[v] - S->primal_tol)      S->cB[k] = -1.0;
        else if (!SK_IS_INF(S->ub[v]) && S->xB[k] > S->ub[v] + S->primal_tol)     S->cB[k] =  1.0;
        else                                                                      S->cB[k] =  0.0;
    }
}

/* d_j = c_j - y' M_j for every nonbasic j. */
static void spx_price(spx *S, int phase1)
{
    int j, p;
    for (j = 0; j < S->ntot; j++) {
        double s;
        if (S->stat[j] == SK_BASIC) { S->d[j] = 0.0; continue; }
        s = phase1 ? 0.0 : S->cost[j];
        for (p = S->M.p[j]; p < S->M.p[j + 1]; p++) s -= S->M.x[p] * S->y[S->M.i[p]];
        S->d[j] = s;
    }
}

/* Choose the entering variable. Returns -1 when the phase is optimal.
 * pricing_mode 0 is Devex, 1 is Bland, and 2 is Dantzig.  Dantzig is a
 * useful intermediate anti-degeneracy policy: it restores raw reduced-cost
 * progress before resorting to Bland's finite, but often very slow, rule. */
static int spx_choose_entering(spx *S, int pricing_mode, int *dir_out)
{
    int j, best = -1, bestdir = 0;
    double bestval = 0.0;
    for (j = 0; j < S->ntot; j++) {
        signed char t = S->stat[j];
        double dj = S->d[j];
        int dir;
        if (t == SK_BASIC) continue;
        if (S->lb[j] == S->ub[j]) continue;              /* fixed: never enters */
        if (t == SK_AT_LOWER)      { if (dj >= -S->dual_tol) continue; dir =  1; }
        else if (t == SK_AT_UPPER) { if (dj <=  S->dual_tol) continue; dir = -1; }
        else                       { if (fabs(dj) <= S->dual_tol) continue; dir = (dj < 0.0) ? 1 : -1; }
        if (pricing_mode == 1) return (*dir_out = dir), j;
        if (pricing_mode == 2) {
            double score = fabs(dj);
            if (score > bestval) { bestval = score; best = j; bestdir = dir; }
            continue;
        }
        /* Devex: score d_j^2 / w_j rather than |d_j|.  Dantzig's rule measures
         * the objective gain per unit *step in x_j*, which says nothing about
         * how far the ratio test will actually let x_j move.  On degenerate
         * models it therefore keeps picking columns that yield a zero step.
         * The reference weight w_j estimates the length of the move, so the
         * score approximates gain per unit distance travelled in the basis. */
        {
            double w = S->w ? S->w[j] : 1.0;
            double score = (dj * dj) / (w > 1e-12 ? w : 1e-12);
            if (score > bestval) { bestval = score; best = j; bestdir = dir; }
        }
    }
    *dir_out = bestdir;
    return best;
}

/* Devex weight update (Forrest-Goldfarb).
 *
 * Needs the pivot row of B^{-1}N, obtained by one extra BTRAN of e_r followed
 * by a pass over the nonbasic columns - the same order of cost as pricing
 * itself, and it routinely pays for itself several times over in iterations
 * saved.  Weights are reset to a fresh reference framework when they grow
 * large enough that the estimate has drifted. */
static void spx_devex_update(spx *S, int enter, int leave, double pivot)
{
    int j, p;
    double wq, inv;

    if (!S->w || !S->rho) return;
    if (fabs(pivot) < 1e-12) return;

    memset(S->rho, 0, (size_t)S->nrow * sizeof(double));
    S->rho[leave] = 1.0;
    if (sk_lu_btran(&S->lu, S->rho) != SK_OK) return;

    wq = S->w[enter];
    inv = 1.0 / pivot;
    for (j = 0; j < S->ntot; j++) {
        double arj = 0.0, cand;
        if (S->stat[j] == SK_BASIC || j == enter) continue;
        for (p = S->M.p[j]; p < S->M.p[j + 1]; p++) arj += S->M.x[p] * S->rho[S->M.i[p]];
        if (arj == 0.0) continue;
        cand = (arj * inv) * (arj * inv) * wq;
        if (cand > S->w[j]) S->w[j] = cand;
    }
    {
        double wl = wq * inv * inv;
        S->w[S->basis[leave]] = wl > 1.0 ? wl : 1.0;
    }
    S->w[enter] = 1.0;

    if (++S->devex_age > 2000) {                 /* refresh the framework */
        for (j = 0; j < S->ntot; j++) S->w[j] = 1.0;
        S->devex_age = 0;
    }
}

/* Effective ratio-test bounds for basic position k. */
static void spx_eff_bounds(const spx *S, int k, int phase1, double *lo, double *hi)
{
    int v = S->basis[k];
    double l = S->lb[v], u = S->ub[v];
    if (phase1) {
        if (!SK_IS_NEG_INF(l) && S->xB[k] < l - S->primal_tol) { *lo = -SK_INFINITY; *hi = l; return; }
        if (!SK_IS_INF(u)     && S->xB[k] > u + S->primal_tol) { *lo = u; *hi = SK_INFINITY; return; }
    }
    *lo = l; *hi = u;
}

/* Harris two-pass ratio test.
 * Returns the leaving position, -1 for a bound flip, -2 for unboundedness. */
static int spx_ratio_test(spx *S, int enter, int dir, int phase1,
                          double *step_out, int *to_upper_out)
{
    int k, leave = -1, to_upper = 0;
    double range = S->ub[enter] - S->lb[enter];
    double tmax, best_pivot = 0.0, step = 0.0;
    const double relax = S->primal_tol;

    if (SK_IS_INF(S->ub[enter]) || SK_IS_NEG_INF(S->lb[enter])) range = SK_INFINITY;
    tmax = range;

    /* pass 1: largest step tolerating a relaxed bound violation */
    for (k = 0; k < S->nrow; k++) {
        double delta = -dir * S->alpha[k], lo, hi, r;
        if (fabs(delta) <= SPX_PIVOT_ZERO) continue;
        spx_eff_bounds(S, k, phase1, &lo, &hi);
        if (delta > 0.0) {
            if (SK_IS_INF(hi)) continue;
            r = (hi + relax - S->xB[k]) / delta;
        } else {
            if (SK_IS_NEG_INF(lo)) continue;
            r = (lo - relax - S->xB[k]) / delta;
        }
        if (r < 0.0) r = 0.0;
        if (r < tmax) tmax = r;
    }

    /* pass 2: among rows admitted by tmax, take the largest pivot */
    for (k = 0; k < S->nrow; k++) {
        double delta = -dir * S->alpha[k], lo, hi, r;
        if (fabs(delta) <= SPX_PIVOT_ZERO) continue;
        spx_eff_bounds(S, k, phase1, &lo, &hi);
        if (delta > 0.0) {
            if (SK_IS_INF(hi)) continue;
            r = (hi - S->xB[k]) / delta;
        } else {
            if (SK_IS_NEG_INF(lo)) continue;
            r = (lo - S->xB[k]) / delta;
        }
        if (r < 0.0) r = 0.0;
        if (r <= tmax && fabs(delta) > best_pivot) {
            best_pivot = fabs(delta);
            leave = k;
            step = r;
            to_upper = (delta > 0.0);
        }
    }

    if (leave < 0) {
        if (SK_IS_INF(range)) return -2;      /* nothing blocks: unbounded */
        *step_out = range;
        return -1;                            /* bound flip */
    }
    if (!SK_IS_INF(range) && range <= step) { *step_out = range; return -1; }
    *step_out = step;
    *to_upper_out = to_upper;
    return leave;
}

/* ------------------------------------------------------------------ solve */

sk_status sk_simplex_solve(const sk_model *m, const sk_options *o,
                           sk_solution *s, sk_spx_stats *stats)
{
    spx S;
    sk_model work;
    sk_options defaults;
    sk_status st;
    double *presolve_lower = NULL, *presolve_upper = NULL;
    int presolve_infeasible = 0;
    int phase1, bland = 0, j, k, p;
    long long stall = 0, itlimit;
    double last_infeas = INFINITY, t0, tphase;
    sk_result result = SK_RESULT_NUMERIC_FAILURE;

    if (!m || !s) return SK_ERR_ARG;
    if (!o) { sk_options_default(&defaults); o = &defaults; }
    st = spx_singleton_presolve(m, o, &work, &presolve_lower, &presolve_upper,
                                &presolve_infeasible);
    if (st != SK_OK) return st;
    if (presolve_infeasible) {
        s->result = SK_RESULT_INFEASIBLE;
        s->ncol = m->ncol; s->nrow = m->nrow;
        s->primal_infeasibility = INFINITY;
        return SK_OK;
    }
    itlimit = (o->iteration_limit > 0) ? o->iteration_limit
                                       : (long long)20000 + 200LL * (m->nrow + m->ncol);

    st = spx_build(&S, &work, o);
    if (st != SK_OK) { free(presolve_lower); free(presolve_upper); return st; }
    t0 = sk_wall_seconds();
    tphase = t0;

    st = spx_refactorize(&S);
    if (st != SK_OK) { spx_free(&S); free(presolve_lower); free(presolve_upper); return st; }

    phase1 = (spx_infeasibility(&S) > S.primal_tol);

    for (;;) {
        int enter, dir, leave, to_upper = 0;
        double step;


        if (S.st.iterations >= itlimit) { result = SK_RESULT_ITERATION_LIMIT; break; }
        if (o->time_limit > 0.0 && sk_wall_seconds() - t0 > o->time_limit) {
            result = SK_RESULT_TIME_LIMIT; break;
        }

        /* refresh factorization periodically to bound drift and eta growth */
        if (sk_lu_needs_refactor(&S.lu, S.refactor_interval)) {
            st = spx_refactorize(&S);
            if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
        }

        if (phase1 && spx_infeasibility(&S) <= S.primal_tol) {
            S.st.phase1_iterations = S.st.iterations;
            S.st.phase1_seconds = sk_wall_seconds() - tphase;
            tphase = sk_wall_seconds();
            phase1 = 0;
            /* Phase 1 can finish immediately after a basis exchange. Start
               Phase 2 from a freshly factorized basis rather than carrying
               an eta update across the change in objective. */
            st = spx_refactorize(&S);
            if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
            bland = 0;
            stall = 0;
            last_infeas = INFINITY;
        }

        spx_build_cB(&S, phase1);
        memcpy(S.y, S.cB, (size_t)S.nrow * sizeof(double));
        if (sk_lu_btran(&S.lu, S.y) != SK_OK) {
            st = spx_refactorize(&S);
            if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
            continue;
        }
        spx_price(&S, phase1);

        enter = spx_choose_entering(&S, bland, &dir);
        if (enter < 0) {
            if (phase1) {
                /* phase-1 optimum with residual infeasibility: infeasible LP */
                if (spx_infeasibility(&S) > S.primal_tol * 100.0) { result = SK_RESULT_INFEASIBLE; break; }
                S.st.phase1_iterations = S.st.iterations;
                S.st.phase1_seconds = sk_wall_seconds() - tphase;
                tphase = sk_wall_seconds();
                phase1 = 0; bland = 0; stall = 0; last_infeas = INFINITY;
                st = spx_refactorize(&S);
                if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
                continue;
            }
            result = SK_RESULT_OPTIMAL;
            break;
        }

        /* alpha = B^{-1} M_enter */
        memset(S.alpha, 0, (size_t)S.nrow * sizeof(double));
        for (p = S.M.p[enter]; p < S.M.p[enter + 1]; p++) S.alpha[S.M.i[p]] = S.M.x[p];
        if (sk_lu_ftran(&S.lu, S.alpha) != SK_OK) {
            st = spx_refactorize(&S);
            if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
            continue;
        }

        leave = spx_ratio_test(&S, enter, dir, phase1, &step, &to_upper);
        if (leave == -2) {
            if (phase1) { result = SK_RESULT_NUMERIC_FAILURE; break; }
            result = SK_RESULT_UNBOUNDED;
            break;
        }

        if (leave == -1) {
            /* bound flip: the entering variable crosses to its other bound */
            double delta = dir * step;
            for (k = 0; k < S.nrow; k++) S.xB[k] -= delta * S.alpha[k];
            for (k = 0; k < S.nrow; k++) S.xval[S.basis[k]] = S.xB[k];
            S.xval[enter] += delta;
            S.stat[enter] = (S.stat[enter] == SK_AT_LOWER) ? SK_AT_UPPER : SK_AT_LOWER;
            S.st.bound_flips++;
            S.st.iterations++;
            continue;
        }

        {
            int lv = S.basis[leave];
            double delta = dir * step;
            double lo, hi;

            /* Refresh the Devex weights while the old factorization and the
               old basis are both still in place - the update needs the pivot
               row of the basis we are about to leave. */
            if (bland != 1) spx_devex_update(&S, enter, leave, S.alpha[leave]);

            /* Capture the phase-1 effective bound before xB is updated.
               After the update the basic value becomes feasible and the
               special one-sided Phase-1 bound is no longer detectable. */
            spx_eff_bounds(&S, leave, phase1, &lo, &hi);

            for (k = 0; k < S.nrow; k++) S.xB[k] -= delta * S.alpha[k];
            for (k = 0; k < S.nrow; k++) S.xval[S.basis[k]] = S.xB[k];

            /* leaving variable is pinned to the bound it reached */
            /* In phase 1 an infeasible-below basic has effective (-inf, lb)
               and an infeasible-above basic has effective (ub, +inf). The
               ratio-test side is therefore an artificial side, not always
               the physical variable side used by phase 2. */
            if (phase1 && to_upper && !SK_IS_NEG_INF(S.lb[lv]) && hi == S.lb[lv]) {
                S.stat[lv] = SK_AT_LOWER; S.xval[lv] = S.lb[lv];
            } else if (phase1 && !to_upper && !SK_IS_INF(S.ub[lv]) && lo == S.ub[lv]) {
                S.stat[lv] = SK_AT_UPPER; S.xval[lv] = S.ub[lv];
            } else if (to_upper) {
                S.stat[lv] = SK_AT_UPPER; S.xval[lv] = SK_IS_INF(hi) ? S.xval[lv] : hi;
            } else {
                S.stat[lv] = SK_AT_LOWER; S.xval[lv] = SK_IS_NEG_INF(lo) ? S.xval[lv] : lo;
            }
            if (S.lb[lv] == S.ub[lv]) { S.stat[lv] = SK_AT_LOWER; S.xval[lv] = S.lb[lv]; }

            S.xval[enter] += delta;
            S.basis[leave] = enter;
            S.stat[enter] = SK_BASIC;
            S.xB[leave] = S.xval[enter];

            if (sk_lu_update(&S.lu, leave, S.alpha, SPX_DROP_TOL) != SK_OK) {
                st = spx_refactorize(&S);
                if (st != SK_OK) { result = SK_RESULT_NUMERIC_FAILURE; break; }
            }
        }

        S.st.iterations++;

        /* Stall detection: Devex -> Dantzig -> Bland.  Dantzig restores
         * objective progress on degenerate Netlib bases before the finite but
         * slower Bland fallback is engaged. */
        {
            double now = phase1 ? spx_infeasibility(&S) : 0.0;
            if (phase1) {
                if (now >= last_infeas - 1e-12) stall++; else stall = 0;
                last_infeas = now;
            } else {
                if (step <= 1e-12) stall++; else stall = 0;
            }
            if (stall > 1000 && bland == 2) {
                bland = 1;
                S.st.bland_episodes++;
            } else if (stall > 200 && bland == 0) {
                bland = 2;
            } else if (stall == 0 && bland) {
                bland = 0;
            }
        }
    }

    S.st.phase2_seconds = sk_wall_seconds() - tphase;
    S.st.final_primal_infeasibility = spx_infeasibility(&S);

    /* ---- publish the solution ---- */
    if (!s->x)      s->x = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    if (!s->y)      s->y = (double *)calloc((size_t)m->nrow + 1, sizeof(double));
    if (!s->rc)     s->rc = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    if (!s->rowact) s->rowact = (double *)calloc((size_t)m->nrow + 1, sizeof(double));
    if (!s->x || !s->y || !s->rc || !s->rowact) { spx_free(&S); free(presolve_lower); free(presolve_upper); return SK_ERR_MEMORY; }
    s->ncol = m->ncol; s->nrow = m->nrow;

    /* back to the original variable space: x = C x~ */
    for (j = 0; j < m->ncol; j++) s->x[j] = S.xval[j] * S.C[j];

    /* duals from the phase-2 basis */
    spx_build_cB(&S, 0);
    memcpy(S.y, S.cB, (size_t)S.nrow * sizeof(double));
    if (sk_lu_btran(&S.lu, S.y) == SK_OK) {
        /* The simplex equality is [A -I](x;s)=0.  Its multiplier has
           stationarity c-A'y=0, whereas the public API uses c+A'y=0. */
        /* and the original dual space: y = R y~ */
        for (k = 0; k < m->nrow; k++) s->y[k] = -S.y[k] * S.R[k];
    } else {
        for (k = 0; k < m->nrow; k++) s->y[k] = 0.0;
    }

    s->result = result;
    s->iterations = S.st.iterations;
    s->solve_seconds = sk_wall_seconds() - t0;
    if (stats) *stats = S.st;

    spx_free(&S);
    free(presolve_lower); free(presolve_upper);

    /* Residuals and reduced costs come from the independent checker, never
     * from solver-internal state. */
    sk_verify(m, s);
    if (s->result == SK_RESULT_OPTIMAL && s->primal_infeasibility > 1e-6)
        s->result = SK_RESULT_NUMERIC_FAILURE;
    return SK_OK;
}
