/* Core services: model ownership, timing, MPS ingestion and the independent
 * primal-dual certificate checker.
 *
 * The checker deliberately shares no state with the solvers.  It recomputes
 * A*x and c + A'y from the original model and re-derives every residual, so a
 * bug inside the simplex cannot make an infeasible point look optimal.
 */
#include "sankhya.h"
#include "sankhya_lp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define sk_strcasecmp _stricmp
#else
#include <strings.h>
#define sk_strcasecmp strcasecmp
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

const char *sk_version(void) { return "SANKHYA 0.2.0"; }

const char *sk_result_name(sk_result r)
{
    switch (r) {
    case SK_RESULT_OPTIMAL:         return "optimal";
    case SK_RESULT_INFEASIBLE:      return "infeasible";
    case SK_RESULT_UNBOUNDED:       return "unbounded";
    case SK_RESULT_ITERATION_LIMIT: return "iteration_limit";
    case SK_RESULT_TIME_LIMIT:      return "time_limit";
    case SK_RESULT_NUMERIC_FAILURE: return "numeric_failure";
    case SK_RESULT_GAP_LIMIT:       return "gap_limit";
    default:                        return "not_run";
    }
}

const char *sk_status_name(sk_status s)
{
    switch (s) {
    case SK_OK:              return "ok";
    case SK_ERR_ARG:         return "invalid_argument";
    case SK_ERR_MEMORY:      return "out_of_memory";
    case SK_ERR_STRUCTURE:   return "invalid_structure";
    case SK_ERR_IO:          return "io_error";
    case SK_ERR_SINGULAR:    return "singular";
    case SK_ERR_UNSUPPORTED: return "unsupported";
    case SK_ERR_NUMERIC:     return "numeric_error";
    default:                 return "unknown";
    }
}

double sk_wall_seconds(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
#endif
}

/* ------------------------------------------------------------------ model */

void sk_model_init(sk_model *m)
{
    memset(m, 0, sizeof(*m));
    m->objsense = 1.0;
}

static void free_names(char **v, int n)
{
    int i;
    if (!v) return;
    for (i = 0; i < n; i++) free(v[i]);
    free(v);
}

void sk_model_free(sk_model *m)
{
    if (!m) return;
    free(m->A.p); free(m->A.i); free(m->A.x);
    if (m->Q) { free(m->Q->p); free(m->Q->i); free(m->Q->x); free(m->Q); }
    free(m->c); free(m->rlow); free(m->rupp);
    free(m->clow); free(m->cupp); free(m->vartype);
    free_names(m->rowname, m->nrow);
    free_names(m->colname, m->ncol);
    free(m->name);
    sk_model_init(m);
}

sk_status sk_model_alloc(sk_model *m, int nrow, int ncol, int nzmax)
{
    if (!m || nrow < 0 || ncol < 0 || nzmax < 0) return SK_ERR_ARG;
    sk_model_init(m);
    m->nrow = nrow;
    m->ncol = ncol;
    m->A.nrow = nrow;
    m->A.ncol = ncol;
    m->A.nzmax = nzmax;
    m->A.p = (int *)calloc((size_t)ncol + 1, sizeof(int));
    m->A.i = (int *)malloc((size_t)(nzmax > 0 ? nzmax : 1) * sizeof(int));
    m->A.x = (double *)malloc((size_t)(nzmax > 0 ? nzmax : 1) * sizeof(double));
    m->c    = (double *)calloc((size_t)ncol + 1, sizeof(double));
    m->clow = (double *)calloc((size_t)ncol + 1, sizeof(double));
    m->cupp = (double *)calloc((size_t)ncol + 1, sizeof(double));
    m->rlow = (double *)calloc((size_t)nrow + 1, sizeof(double));
    m->rupp = (double *)calloc((size_t)nrow + 1, sizeof(double));
    m->vartype = (unsigned char *)calloc((size_t)ncol + 1, 1);
    if (!m->A.p || !m->A.i || !m->A.x || !m->c || !m->clow || !m->cupp ||
        !m->rlow || !m->rupp || !m->vartype) { sk_model_free(m); return SK_ERR_MEMORY; }
    return SK_OK;
}

int sk_model_num_integer(const sk_model *m)
{
    int j, n = 0;
    for (j = 0; j < m->ncol; j++) if (m->vartype[j] == SK_INTEGER) n++;
    return n;
}

/* --------------------------------------------------------------- solution */

void sk_solution_init(sk_solution *s)
{
    memset(s, 0, sizeof(*s));
    s->result = SK_RESULT_NOT_RUN;
    s->objective = NAN;
    s->dual_bound = -SK_INFINITY;
    s->mip_gap = INFINITY;
}

void sk_solution_free(sk_solution *s)
{
    if (!s) return;
    free(s->x); free(s->y); free(s->rc); free(s->rowact);
    sk_solution_init(s);
}

/* ---------------------------------------------------------------- options */

void sk_options_default(sk_options *o)
{
    memset(o, 0, sizeof(*o));
    o->primal_tol = 1e-7;
    o->dual_tol   = 1e-7;
    o->pivot_tol  = 0.1;
    o->time_limit = 0.0;
    o->iteration_limit = 0;
    o->node_limit = 0;
    o->presolve = 1;
    o->scaling = 1;
    o->refactor_interval = 100;
    o->verbosity = 1;
    o->threads = 1;
    o->mip_gap_abs = 1e-9;
    o->mip_gap_rel = 1e-6;
    o->mip_cuts = 1;
    o->mip_heuristics = 1;
    o->dual_simplex = 1;
    o->random_seed = 20260825ULL;
}

/* ------------------------------------------------------------ MPS bridge */

/* The legacy reader is retained: it already covers RANGES, BOUNDS, OBJSENSE
 * and MARKER/INTORG.  It normalizes every instance to a minimization and uses
 * IEEE infinity for free bounds; this bridge converts to the solver's int
 * indexing and 1e30 infinity convention. */
static double conv_lo(double v) { return (v <= -1e29 || isinf(v)) ? -SK_INFINITY : v; }
static double conv_hi(double v) { return (v >=  1e29 || isinf(v)) ?  SK_INFINITY : v; }

static char *dupstr(const char *s)
{
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s) + 1;
    d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

typedef struct sk_q_entry {
    int row;
    int column;
    double value;
} sk_q_entry;

static int q_entry_compare(const void *left, const void *right)
{
    const sk_q_entry *a = (const sk_q_entry *)left;
    const sk_q_entry *b = (const sk_q_entry *)right;
    if (a->column != b->column) return a->column < b->column ? -1 : 1;
    if (a->row != b->row) return a->row < b->row ? -1 : 1;
    return 0;
}

static int column_index(const sk_model *m, const char *name)
{
    int j;
    for (j = 0; j < m->ncol; ++j)
        if (m->colname && m->colname[j] && strcmp(m->colname[j], name) == 0) return j;
    return -1;
}

static int append_q_entry(sk_q_entry **entries, size_t *count, size_t *capacity,
                          int row, int column, double value)
{
    sk_q_entry *next;
    if (*count == *capacity) {
        size_t next_capacity = *capacity ? 2 * *capacity : 32;
        next = (sk_q_entry *)realloc(*entries, next_capacity * sizeof(*next));
        if (!next) return 0;
        *entries = next;
        *capacity = next_capacity;
    }
    (*entries)[*count].row = row;
    (*entries)[*count].column = column;
    (*entries)[*count].value = value;
    ++*count;
    return 1;
}

/* Read the quadratic portion after the established LP/MILP MPS reader has
 * built the variable map. QPS stores Q for 0.5*x'Q*x. Input may contain one
 * triangle, so every off-diagonal coefficient is mirrored into the full
 * symmetric CSC representation used by sk_model. */
static sk_status read_qps_quadratic(const char *path, sk_model *m)
{
    FILE *file = NULL;
    char line[8192];
    sk_q_entry *entries = NULL;
    size_t count = 0, capacity = 0, p;
    int in_quadratic = 0;
    sk_csc *q = NULL;

    file = fopen(path, "r");
    if (!file) return SK_ERR_IO;
    while (fgets(line, sizeof(line), file)) {
        char *token[4];
        char *current;
        int ntoken = 0, left, right;
        char *end = NULL;
        double value;
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r') continue;
        current = strtok(line, " \t\r\n");
        while (current && ntoken < 4) { token[ntoken++] = current; current = strtok(NULL, " \t\r\n"); }
        if (!ntoken) continue;
        if (sk_strcasecmp(token[0], "QUADOBJ") == 0 || sk_strcasecmp(token[0], "QMATRIX") == 0 ||
            sk_strcasecmp(token[0], "QSECTION") == 0) { in_quadratic = 1; continue; }
        if (sk_strcasecmp(token[0], "ENDATA") == 0) break;
        if (sk_strcasecmp(token[0], "ROWS") == 0 || sk_strcasecmp(token[0], "COLUMNS") == 0 ||
            sk_strcasecmp(token[0], "RHS") == 0 || sk_strcasecmp(token[0], "RANGES") == 0 ||
            sk_strcasecmp(token[0], "BOUNDS") == 0) { in_quadratic = 0; continue; }
        if (!in_quadratic) continue;
        if (ntoken != 3) { fclose(file); free(entries); return SK_ERR_STRUCTURE; }
        left = column_index(m, token[0]);
        right = column_index(m, token[1]);
        value = strtod(token[2], &end);
        if (left < 0 || right < 0 || !end || *end != '\0' || !isfinite(value)) {
            fclose(file); free(entries); return SK_ERR_STRUCTURE;
        }
        if (!append_q_entry(&entries, &count, &capacity, right, left, value) ||
            (left != right && !append_q_entry(&entries, &count, &capacity, left, right, value))) {
            fclose(file); free(entries); return SK_ERR_MEMORY;
        }
    }
    fclose(file);
    if (!count) { free(entries); return SK_OK; }
    qsort(entries, count, sizeof(*entries), q_entry_compare);
    q = (sk_csc *)calloc(1, sizeof(*q));
    if (!q) { free(entries); return SK_ERR_MEMORY; }
    q->nrow = q->ncol = m->ncol;
    q->p = (int *)calloc((size_t)m->ncol + 1, sizeof(int));
    q->i = (int *)malloc(count * sizeof(int));
    q->x = (double *)malloc(count * sizeof(double));
    if (!q->p || !q->i || !q->x) { free(q->p); free(q->i); free(q->x); free(q); free(entries); return SK_ERR_MEMORY; }
    for (p = 0; p < count;) {
        const size_t begin = p;
        double sum = 0.0;
        while (p < count && entries[p].column == entries[begin].column && entries[p].row == entries[begin].row)
            sum += entries[p++].value;
        if (sum != 0.0) { q->i[q->nzmax] = entries[begin].row; q->x[q->nzmax++] = sum; q->p[entries[begin].column + 1]++; }
    }
    for (p = 0; p < (size_t)m->ncol; ++p) q->p[p + 1] += q->p[p];
    free(entries);
    m->Q = q;
    return SK_OK;
}

sk_status sk_read_mps(const char *path, sk_model *m)
{
    SankhyaLPModel raw;
    sk_status st = SK_OK;
    int i, j, nnz;

    if (!path || !m) return SK_ERR_ARG;
    sankhya_lp_model_init(&raw);
    if (sankhya_lp_read_mps(path, &raw) != SANKHYA_OK) return SK_ERR_IO;

    nnz = (int)raw.A.nonzeros;
    st = sk_model_alloc(m, (int)raw.A.rows, (int)raw.A.columns, nnz);
    if (st != SK_OK) { sankhya_lp_model_destroy(&raw); return st; }

    for (j = 0; j <= m->ncol; j++) m->A.p[j] = (int)raw.A.column_offsets[j];
    for (i = 0; i < nnz; i++) { m->A.i[i] = raw.A.row_indices[i]; m->A.x[i] = raw.A.values[i]; }
    for (i = 0; i < m->nrow; i++) {
        m->rlow[i] = conv_lo(raw.row_lower[i]);
        m->rupp[i] = conv_hi(raw.row_upper[i]);
    }
    for (j = 0; j < m->ncol; j++) {
        m->c[j]    = raw.objective[j];
        m->clow[j] = conv_lo(raw.column_lower[j]);
        m->cupp[j] = conv_hi(raw.column_upper[j]);
        m->vartype[j] = (raw.variable_type[j] == SANKHYA_CONTINUOUS) ? SK_CONTINUOUS : SK_INTEGER;
    }
    m->objshift = raw.objective_offset;
    m->objsense = 1.0;   /* the reader has already normalized to minimize */

    if (raw.row_names) {
        m->rowname = (char **)calloc((size_t)m->nrow + 1, sizeof(char *));
        if (m->rowname) for (i = 0; i < m->nrow; i++) m->rowname[i] = dupstr(raw.row_names[i]);
    }
    if (raw.column_names) {
        m->colname = (char **)calloc((size_t)m->ncol + 1, sizeof(char *));
        if (m->colname) for (j = 0; j < m->ncol; j++) m->colname[j] = dupstr(raw.column_names[j]);
    }
    sankhya_lp_model_destroy(&raw);
    st = read_qps_quadratic(path, m);
    if (st != SK_OK) { sk_model_free(m); return st; }
    return SK_OK;
}

/* --------------------------------------------------- certificate checking */

/* Recomputes every residual from the model alone.
 *
 * Primal:  max violation of rlow <= Ax <= rupp and clow <= x <= cupp.
 * Dual:    with rc = c + A'y, the sign conditions
 *              x_j at clow  =>  rc_j >= 0        r_i at rlow  =>  y_i <= 0
 *              x_j at cupp  =>  rc_j <= 0        r_i at rupp  =>  y_i >= 0
 *              x_j interior =>  rc_j == 0        r_i interior =>  y_i == 0
 *          A multiplier whose sign points at an infinite bound is itself a
 *          dual infeasibility, and is reported as such rather than ignored.
 * For convex QPs, the same stationarity-sign and complementary-slackness
 * tests form a KKT residual. LPs additionally report the primal/dual gap.
 */
sk_status sk_verify(const sk_model *m, sk_solution *s)
{
    int i, j, p;
    double *act = NULL, *rc = NULL, *qx = NULL;
    double pinf = 0.0, dinf = 0.0, comp = 0.0;
    double pobj, dobj;
    const double bt = 1e-9;   /* how close counts as "at" a bound */

    if (!m || !s || !s->x) return SK_ERR_ARG;

    act = (double *)calloc((size_t)m->nrow + 1, sizeof(double));
    rc  = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    if (m->Q) qx = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    if (!act || !rc || (m->Q && !qx)) { free(act); free(rc); free(qx); return SK_ERR_MEMORY; }

    for (j = 0; j < m->ncol; j++)
        for (p = m->A.p[j]; p < m->A.p[j + 1]; p++)
            act[m->A.i[p]] += m->A.x[p] * s->x[j];

    pobj = m->objshift;
    for (j = 0; j < m->ncol; j++) pobj += m->c[j] * s->x[j];
    if (m->Q) {
        for (j = 0; j < m->Q->ncol; j++)
            for (p = m->Q->p[j]; p < m->Q->p[j + 1]; p++) qx[m->Q->i[p]] += m->Q->x[p] * s->x[j];
        for (j = 0; j < m->ncol; j++) pobj += 0.5 * s->x[j] * qx[j];
    }

    /* ---- primal residuals ---- */
    for (i = 0; i < m->nrow; i++) {
        if (!SK_IS_NEG_INF(m->rlow[i]) && m->rlow[i] - act[i] > pinf) pinf = m->rlow[i] - act[i];
        if (!SK_IS_INF(m->rupp[i])     && act[i] - m->rupp[i] > pinf) pinf = act[i] - m->rupp[i];
    }
    for (j = 0; j < m->ncol; j++) {
        if (!isfinite(s->x[j])) { pinf = INFINITY; break; }
        if (!SK_IS_NEG_INF(m->clow[j]) && m->clow[j] - s->x[j] > pinf) pinf = m->clow[j] - s->x[j];
        if (!SK_IS_INF(m->cupp[j])     && s->x[j] - m->cupp[j] > pinf) pinf = s->x[j] - m->cupp[j];
    }

    /* ---- dual residuals ---- */
    dobj = m->objshift;
    if (s->y) {
        for (j = 0; j < m->ncol; j++) {
            double r = m->c[j] + (m->Q ? qx[j] : 0.0);
            for (p = m->A.p[j]; p < m->A.p[j + 1]; p++) r += m->A.x[p] * s->y[m->A.i[p]];
            rc[j] = r;
        }
        for (j = 0; j < m->ncol; j++) {
            int atlo = !SK_IS_NEG_INF(m->clow[j]) && fabs(s->x[j] - m->clow[j]) <= bt;
            int athi = !SK_IS_INF(m->cupp[j])     && fabs(s->x[j] - m->cupp[j]) <= bt;
            double r = rc[j];
            if (atlo && athi) {           /* fixed: any sign is dual feasible */
            } else if (atlo) {
                if (-r > dinf) dinf = -r;
            } else if (athi) {
                if ( r > dinf) dinf =  r;
            } else {
                if (fabs(r) > dinf) dinf = fabs(r);
            }
            /* dual objective contribution, at the bound the sign selects */
            if (r > 0.0) {
                if (SK_IS_NEG_INF(m->clow[j])) { if (r > dinf) dinf = r; }
                else dobj += r * m->clow[j];
            } else if (r < 0.0) {
                if (SK_IS_INF(m->cupp[j])) { if (-r > dinf) dinf = -r; }
                else dobj += r * m->cupp[j];
            }
            if (r > 0.0 && !SK_IS_NEG_INF(m->clow[j])) {
                double product = r * fabs(s->x[j] - m->clow[j]);
                if (product > comp) comp = product;
            } else if (r < 0.0 && !SK_IS_INF(m->cupp[j])) {
                double product = -r * fabs(m->cupp[j] - s->x[j]);
                if (product > comp) comp = product;
            }
        }
        for (i = 0; i < m->nrow; i++) {
            int atlo = !SK_IS_NEG_INF(m->rlow[i]) && fabs(act[i] - m->rlow[i]) <= bt;
            int athi = !SK_IS_INF(m->rupp[i])     && fabs(act[i] - m->rupp[i]) <= bt;
            double yv = s->y[i];
            if (atlo && athi) {
            } else if (atlo) {
                if ( yv > dinf) dinf =  yv;
            } else if (athi) {
                if (-yv > dinf) dinf = -yv;
            } else {
                if (fabs(yv) > dinf) dinf = fabs(yv);
            }
            if (yv < 0.0) {
                if (SK_IS_NEG_INF(m->rlow[i])) { if (-yv > dinf) dinf = -yv; }
                else dobj += yv * m->rlow[i];
                if (!SK_IS_NEG_INF(m->rlow[i])) {
                    double product = -yv * fabs(act[i] - m->rlow[i]);
                    if (product > comp) comp = product;
                }
            } else if (yv > 0.0) {
                if (SK_IS_INF(m->rupp[i])) { if (yv > dinf) dinf = yv; }
                else dobj += yv * m->rupp[i];
                if (!SK_IS_INF(m->rupp[i])) {
                    double product = yv * fabs(m->rupp[i] - act[i]);
                    if (product > comp) comp = product;
                }
            }
        }
        s->complementarity = m->Q ? comp / (1.0 + fabs(pobj)) : fabs(pobj - dobj) / (1.0 + fabs(pobj));
    } else {
        dinf = INFINITY;
        s->complementarity = INFINITY;
    }

    if (s->rowact) for (i = 0; i < m->nrow; i++) s->rowact[i] = act[i];
    if (s->rc)     for (j = 0; j < m->ncol; j++) s->rc[j] = rc[j];
    s->objective = pobj;
    s->primal_infeasibility = pinf > 0.0 ? pinf : 0.0;
    s->dual_infeasibility   = dinf > 0.0 ? dinf : 0.0;
    free(act); free(rc); free(qx);
    return SK_OK;
}
