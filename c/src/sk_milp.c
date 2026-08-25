/* Branch-and-bound for mixed-integer linear programming.
 *
 * Search strategy is depth-first plunging with best-bound backtracking: after
 * branching we descend immediately into the preferred child, which finds
 * incumbents early and keeps the LP bases similar from node to node, and when
 * a dive dies we resume from the open node with the weakest bound, which is
 * what actually proves optimality.  Pure best-first would prove the bound with
 * the fewest nodes but finds its first feasible solution far too late; pure
 * depth-first finds solutions quickly but can wander indefinitely.
 *
 * Branching is by pseudocost: for each integer variable we accumulate the
 * observed objective degradation per unit of fractionality moved, separately
 * for the up and down directions, and branch on the variable whose predicted
 * degradation is largest in both directions (a product score, which is the
 * standard robust choice).  Until a variable has been branched on often enough
 * to trust its history, its pseudocost is initialised from the root LP.
 *
 * Every node is preceded by bound propagation, which is where most of the
 * search tree is actually removed: tightening integer bounds from row activity
 * limits often makes a subtree infeasible without solving its relaxation.
 */
#include "sk_milp.h"
#include "sk_simplex.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MILP_INT_TOL   1e-6
#define MILP_INF       SK_INFINITY

/* ------------------------------------------------------------------ nodes */

typedef struct bchg {
    int    var;
    unsigned char upper;      /* 1 => this change set an upper bound */
    double value;
} bchg;

typedef struct node {
    bchg  *chg;               /* cumulative bound changes from the root */
    int    nchg;
    double bound;             /* parent relaxation value: a valid dual bound */
    int    depth;
    double frac;              /* fractionality of the branched variable */
} node;

typedef struct heap {
    node **a;
    int n, cap;
} heap;

static int heap_push(heap *h, node *v)
{
    int i;
    if (h->n >= h->cap) {
        int cap = h->cap ? h->cap * 2 : 256;
        node **na = (node **)realloc(h->a, (size_t)cap * sizeof(node *));
        if (!na) return 0;
        h->a = na; h->cap = cap;
    }
    i = h->n++;
    h->a[i] = v;
    while (i > 0) {                       /* sift up on weakest bound */
        int p = (i - 1) / 2;
        if (h->a[p]->bound <= h->a[i]->bound) break;
        { node *t = h->a[p]; h->a[p] = h->a[i]; h->a[i] = t; }
        i = p;
    }
    return 1;
}

static node *heap_pop(heap *h)
{
    node *top, *last;
    int i = 0;
    if (h->n == 0) return NULL;
    top = h->a[0];
    last = h->a[--h->n];
    if (h->n == 0) return top;
    h->a[0] = last;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < h->n && h->a[l]->bound < h->a[m]->bound) m = l;
        if (r < h->n && h->a[r]->bound < h->a[m]->bound) m = r;
        if (m == i) break;
        { node *t = h->a[m]; h->a[m] = h->a[i]; h->a[i] = t; }
        i = m;
    }
    return top;
}

static void node_free(node *v) { if (v) { free(v->chg); free(v); } }

static node *node_child(const node *parent, int var, int upper, double value,
                        double bound, double frac)
{
    node *v = (node *)calloc(1, sizeof(node));
    int n = parent ? parent->nchg : 0;
    if (!v) return NULL;
    v->chg = (bchg *)malloc((size_t)(n + 1) * sizeof(bchg));
    if (!v->chg) { free(v); return NULL; }
    if (n) memcpy(v->chg, parent->chg, (size_t)n * sizeof(bchg));
    v->chg[n].var = var;
    v->chg[n].upper = (unsigned char)upper;
    v->chg[n].value = value;
    v->nchg = n + 1;
    v->bound = bound;
    v->depth = parent ? parent->depth + 1 : 0;
    v->frac = frac;
    return v;
}

/* ------------------------------------------------------------------ solver */

typedef struct milp {
    const sk_model *m;
    sk_model work;            /* shares A/c/rows with m, owns clow/cupp */
    double *clow0, *cupp0;    /* root bounds after integer rounding */
    double *nlow, *nupp;      /* bounds of the node being solved */

    double *pc_down, *pc_up;  /* pseudocosts, per column */
    int    *nc_down, *nc_up;

    double *incumbent;
    double  incumbent_obj;
    int     have_incumbent;

    sk_options lp_opt;
    sk_solution ls;           /* reusable scratch for node relaxations */
    sk_milp_stats st;
    double  t0, time_limit;
    double  gap_abs, gap_rel;
} milp;

static int is_int_var(const sk_model *m, int j) { return m->vartype[j] == SK_INTEGER; }

static double frac_of(double v) { return v - floor(v); }

static int is_integral(double v)
{
    return fabs(v - floor(v + 0.5)) <= MILP_INT_TOL;
}

/* Apply a node's cumulative bound changes on top of the root bounds. */
static void milp_apply(milp *M, const node *v)
{
    int i;
    memcpy(M->nlow, M->clow0, (size_t)M->m->ncol * sizeof(double));
    memcpy(M->nupp, M->cupp0, (size_t)M->m->ncol * sizeof(double));
    for (i = 0; v && i < v->nchg; i++) {
        const bchg *c = &v->chg[i];
        if (c->upper) { if (c->value < M->nupp[c->var]) M->nupp[c->var] = c->value; }
        else          { if (c->value > M->nlow[c->var]) M->nlow[c->var] = c->value; }
    }
}

/* Interval bound propagation.
 *
 * For each row compute the activity range implied by the current column
 * bounds, then invert the row to tighten each column in turn.  Integer columns
 * are rounded inward after tightening, which is what makes the technique
 * strong on MILP: a bound of 2.3 on an integer becomes 2, often cascading.
 * Returns 0 if the node is proved infeasible. */
static int milp_propagate(milp *M, int rounds)
{
    const sk_model *m = M->m;
    int i, j, p, it, changed = 1;
    double *amin = (double *)malloc((size_t)m->nrow * sizeof(double));
    double *amax = (double *)malloc((size_t)m->nrow * sizeof(double));
    int *ninf_min = (int *)malloc((size_t)m->nrow * sizeof(int));
    int *ninf_max = (int *)malloc((size_t)m->nrow * sizeof(int));
    int ok = 1;

    if (!amin || !amax || !ninf_min || !ninf_max) { ok = 1; goto done; }

    for (it = 0; it < rounds && changed && ok; it++) {
        changed = 0;
        for (i = 0; i < m->nrow; i++) { amin[i] = amax[i] = 0.0; ninf_min[i] = ninf_max[i] = 0; }
        for (j = 0; j < m->ncol && ok; j++) {
            double lo = M->nlow[j], up = M->nupp[j];
            if (lo > up + 1e-9) { ok = 0; break; }
            for (p = m->A.p[j]; p < m->A.p[j + 1]; p++) {
                int r = m->A.i[p];
                double a = m->A.x[p];
                double clo = (a >= 0.0) ? lo : up;
                double chi = (a >= 0.0) ? up : lo;
                if (SK_IS_NEG_INF(clo) || SK_IS_INF(clo)) ninf_min[r]++; else amin[r] += a * clo;
                if (SK_IS_INF(chi) || SK_IS_NEG_INF(chi)) ninf_max[r]++; else amax[r] += a * chi;
            }
        }
        if (!ok) break;

        for (i = 0; i < m->nrow && ok; i++) {
            if (ninf_max[i] == 0 && !SK_IS_NEG_INF(m->rlow[i]) && amax[i] < m->rlow[i] - 1e-7) { ok = 0; break; }
            if (ninf_min[i] == 0 && !SK_IS_INF(m->rupp[i])     && amin[i] > m->rupp[i] + 1e-7) { ok = 0; break; }
        }
        if (!ok) break;

        for (j = 0; j < m->ncol && ok; j++) {
            for (p = m->A.p[j]; p < m->A.p[j + 1] && ok; p++) {
                int r = m->A.i[p];
                double a = m->A.x[p], nl, nu, rest_min, rest_max;
                double lo = M->nlow[j], up = M->nupp[j];
                double clo = (a >= 0.0) ? lo : up;
                double chi = (a >= 0.0) ? up : lo;
                int mn_inf = ninf_min[r], mx_inf = ninf_max[r];
                double mn = amin[r], mx = amax[r];

                if (fabs(a) < 1e-12) continue;
                /* residual activity with column j removed */
                if (SK_IS_NEG_INF(clo) || SK_IS_INF(clo)) mn_inf--; else mn -= a * clo;
                if (SK_IS_INF(chi) || SK_IS_NEG_INF(chi)) mx_inf--; else mx -= a * chi;
                rest_min = mn; rest_max = mx;

                nl = -MILP_INF; nu = MILP_INF;
                if (!SK_IS_INF(m->rupp[r]) && mn_inf == 0) {
                    double t = (m->rupp[r] - rest_min) / a;
                    if (a > 0.0) { if (t < nu) nu = t; } else { if (t > nl) nl = t; }
                }
                if (!SK_IS_NEG_INF(m->rlow[r]) && mx_inf == 0) {
                    double t = (m->rlow[r] - rest_max) / a;
                    if (a > 0.0) { if (t > nl) nl = t; } else { if (t < nu) nu = t; }
                }
                if (is_int_var(m, j)) {
                    if (!SK_IS_NEG_INF(nl)) nl = ceil(nl - 1e-7);
                    if (!SK_IS_INF(nu))     nu = floor(nu + 1e-7);
                }
                if (!SK_IS_NEG_INF(nl) && nl > M->nlow[j] + 1e-9) {
                    M->nlow[j] = nl; changed = 1; M->st.propagations++;
                }
                if (!SK_IS_INF(nu) && nu < M->nupp[j] - 1e-9) {
                    M->nupp[j] = nu; changed = 1; M->st.propagations++;
                }
                if (M->nlow[j] > M->nupp[j] + 1e-9) { ok = 0; break; }
            }
        }

        /* Incumbent-driven objective propagation.  For a minimization model,
           if every other variable has a finite best contribution, the
           incumbent immediately bounds the contribution of this variable.
           This is safe node presolve (not a heuristic cutoff) and is useful
           on knapsack-like MIPLIB rows where feasibility propagation alone is
           weak. */
        if (ok && M->have_incumbent) {
            double obj_min = m->objshift;
            int finite = 1;
            for (j = 0; j < m->ncol; ++j) {
                double term;
                if (m->c[j] >= 0.0) {
                    if (SK_IS_NEG_INF(M->nlow[j])) { finite = 0; break; }
                    term = m->c[j] * M->nlow[j];
                } else {
                    if (SK_IS_INF(M->nupp[j])) { finite = 0; break; }
                    term = m->c[j] * M->nupp[j];
                }
                if (!isfinite(term) || !isfinite(obj_min += term)) { finite = 0; break; }
            }
            if (finite) for (j = 0; j < m->ncol && ok; ++j) {
                double base, bound;
                if (m->c[j] == 0.0) continue;
                base = (m->c[j] >= 0.0) ? m->c[j] * M->nlow[j] : m->c[j] * M->nupp[j];
                bound = (M->incumbent_obj - M->gap_abs - (obj_min - base)) / m->c[j];
                if (m->c[j] > 0.0 && bound < M->nupp[j] - 1e-9) {
                    if (is_int_var(m, j)) bound = floor(bound + 1e-7);
                    M->nupp[j] = bound; changed = 1; M->st.propagations++;
                } else if (m->c[j] < 0.0 && bound > M->nlow[j] + 1e-9) {
                    if (is_int_var(m, j)) bound = ceil(bound - 1e-7);
                    M->nlow[j] = bound; changed = 1; M->st.propagations++;
                }
                if (M->nlow[j] > M->nupp[j] + 1e-9) { ok = 0; break; }
            }
        }
    }

done:
    free(amin); free(amax); free(ninf_min); free(ninf_max);
    return ok;
}

/* Solve the relaxation with the bounds currently in nlow/nupp. */
static sk_result milp_solve_lp(milp *M, double *obj_out)
{
    sk_spx_stats ss;
    double remaining;
    M->work.clow = M->nlow;
    M->work.cupp = M->nupp;
    remaining = (M->time_limit > 0.0) ? M->time_limit - (sk_wall_seconds() - M->t0) : 0.0;
    if (M->time_limit > 0.0 && remaining <= 0.0) return SK_RESULT_TIME_LIMIT;
    M->lp_opt.time_limit = (M->time_limit > 0.0) ? remaining : 0.0;

    memset(&ss, 0, sizeof(ss));
    if (sk_simplex_solve(&M->work, &M->lp_opt, &M->ls, &ss) != SK_OK)
        return SK_RESULT_NUMERIC_FAILURE;
    M->st.lp_solves++;
    M->st.simplex_iterations += ss.iterations;
    if (obj_out) *obj_out = M->ls.objective;
    return M->ls.result;
}

static void milp_accept(milp *M, const double *x, double obj)
{
    memcpy(M->incumbent, x, (size_t)M->m->ncol * sizeof(double));
    M->incumbent_obj = obj;
    M->have_incumbent = 1;
    M->st.solutions_found++;
}

/* Round the integers of the current relaxation, fix them, and re-solve for the
 * continuous variables.  One LP for a decent chance at an early incumbent,
 * which is what makes the bound pruning effective. */
static void milp_heuristic_round(milp *M, const double *xrel)
{
    const sk_model *m = M->m;
    double *savelo = NULL, *savehi = NULL;
    double obj;
    sk_result r;
    int j, feasible = 1;

    savelo = (double *)malloc((size_t)m->ncol * sizeof(double));
    savehi = (double *)malloc((size_t)m->ncol * sizeof(double));
    if (!savelo || !savehi) { free(savelo); free(savehi); return; }
    memcpy(savelo, M->nlow, (size_t)m->ncol * sizeof(double));
    memcpy(savehi, M->nupp, (size_t)m->ncol * sizeof(double));

    for (j = 0; j < m->ncol; j++) {
        double v;
        if (!is_int_var(m, j)) continue;
        v = floor(xrel[j] + 0.5);
        if (v < M->nlow[j]) v = ceil(M->nlow[j] - 1e-7);
        if (v > M->nupp[j]) v = floor(M->nupp[j] + 1e-7);
        if (v < M->nlow[j] - 1e-9 || v > M->nupp[j] + 1e-9) { feasible = 0; break; }
        M->nlow[j] = M->nupp[j] = v;
    }
    if (feasible) {
        r = milp_solve_lp(M, &obj);
        if (r == SK_RESULT_OPTIMAL && M->ls.primal_infeasibility <= 1e-6 &&
            (!M->have_incumbent || obj < M->incumbent_obj - 1e-12)) {
            milp_accept(M, M->ls.x, obj);
            M->st.heuristic_hits++;
        }
    }
    memcpy(M->nlow, savelo, (size_t)m->ncol * sizeof(double));
    memcpy(M->nupp, savehi, (size_t)m->ncol * sizeof(double));
    free(savelo); free(savehi);
}

/* Pseudocost branching. */
static int milp_select_branch(milp *M, const double *x, double *frac_out)
{
    const sk_model *m = M->m;
    int j, best = -1;
    double bestscore = -1.0;

    for (j = 0; j < m->ncol; j++) {
        double f, sd, su, score;
        if (!is_int_var(m, j)) continue;
        if (M->nlow[j] >= M->nupp[j] - 1e-9) continue;
        f = frac_of(x[j]);
        if (f <= MILP_INT_TOL || f >= 1.0 - MILP_INT_TOL) continue;

        sd = M->pc_down[j] * f;
        su = M->pc_up[j] * (1.0 - f);
        if (sd < 1e-8) sd = 1e-8;
        if (su < 1e-8) su = 1e-8;
        score = sd * su;                     /* product rule */
        if (score > bestscore) { bestscore = score; best = j; *frac_out = f; }
    }
    return best;
}

static void milp_update_pseudocost(milp *M, int var, int up, double dobj, double dfrac)
{
    double rate;
    if (var < 0 || dfrac <= 1e-9 || dobj < 0.0) return;
    rate = dobj / dfrac;
    if (up) {
        M->pc_up[var] = (M->pc_up[var] * M->nc_up[var] + rate) / (M->nc_up[var] + 1);
        M->nc_up[var]++;
    } else {
        M->pc_down[var] = (M->pc_down[var] * M->nc_down[var] + rate) / (M->nc_down[var] + 1);
        M->nc_down[var]++;
    }
}

/* ------------------------------------------------------------------ driver */

sk_status sk_milp_solve(const sk_model *m, const sk_options *o,
                        sk_solution *s, sk_milp_stats *stats)
{
    milp M;
    sk_options defaults;
    heap open;
    node *current = NULL;
    double root_obj = 0.0, best_bound;
    int j, nint;
    sk_status rc = SK_OK;
    sk_result final = SK_RESULT_NOT_RUN;

    if (!m || !s) return SK_ERR_ARG;
    if (!o) { sk_options_default(&defaults); o = &defaults; }

    nint = sk_model_num_integer(m);
    if (nint == 0) {                       /* pure LP: nothing to branch on */
        sk_spx_stats ss;
        sk_status st = sk_simplex_solve(m, o, s, &ss);
        if (stats) { memset(stats, 0, sizeof(*stats)); stats->simplex_iterations = ss.iterations;
                     stats->root_bound = s->objective; stats->best_bound = s->objective; }
        return st;
    }

    memset(&M, 0, sizeof(M));
    memset(&open, 0, sizeof(open));
    M.m = m;
    M.t0 = sk_wall_seconds();
    M.time_limit = o->time_limit;
    M.gap_abs = o->mip_gap_abs > 0 ? o->mip_gap_abs : 1e-9;
    M.gap_rel = o->mip_gap_rel > 0 ? o->mip_gap_rel : 1e-6;
    M.incumbent_obj = MILP_INF;
    M.lp_opt = *o;
    M.lp_opt.verbosity = 0;
    M.lp_opt.time_limit = 0.0;
    sk_solution_init(&M.ls);

    /* the work model shares every array with the caller's except the bounds */
    M.work = *m;
    M.clow0 = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.cupp0 = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.nlow  = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.nupp  = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.pc_down = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.pc_up   = (double *)malloc((size_t)m->ncol * sizeof(double));
    M.nc_down = (int *)calloc((size_t)m->ncol, sizeof(int));
    M.nc_up   = (int *)calloc((size_t)m->ncol, sizeof(int));
    M.incumbent = (double *)calloc((size_t)m->ncol, sizeof(double));
    if (!M.clow0 || !M.cupp0 || !M.nlow || !M.nupp || !M.pc_down || !M.pc_up ||
        !M.nc_down || !M.nc_up || !M.incumbent) { rc = SK_ERR_MEMORY; goto cleanup; }

    for (j = 0; j < m->ncol; j++) {
        double lo = m->clow[j], up = m->cupp[j];
        if (is_int_var(m, j)) {
            if (!SK_IS_NEG_INF(lo)) lo = ceil(lo - 1e-7);
            if (!SK_IS_INF(up))     up = floor(up + 1e-7);
        }
        M.clow0[j] = lo; M.cupp0[j] = up;
        M.pc_down[j] = 1.0; M.pc_up[j] = 1.0;
    }

    /* ---- root ---- */
    milp_apply(&M, NULL);
    if (!milp_propagate(&M, 8)) { final = SK_RESULT_INFEASIBLE; goto finish; }
    {
        sk_result r = milp_solve_lp(&M, &root_obj);
        if (r == SK_RESULT_INFEASIBLE) { final = SK_RESULT_INFEASIBLE; goto finish; }
        if (r == SK_RESULT_UNBOUNDED)  { final = SK_RESULT_UNBOUNDED;  goto finish; }
        if (r != SK_RESULT_OPTIMAL)    { final = r; goto finish; }
    }
    M.st.root_bound = root_obj;
    best_bound = root_obj;

    /* pseudocosts seeded from the root objective magnitude */
    for (j = 0; j < m->ncol; j++) {
        double seed = fabs(m->c[j]);
        if (seed < 1e-6) seed = 1e-6;
        M.pc_down[j] = seed; M.pc_up[j] = seed;
    }
    if (o->mip_heuristics) milp_heuristic_round(&M, M.ls.x);

    current = node_child(NULL, 0, 0, 0.0, root_obj, 0.0);
    if (!current) { rc = SK_ERR_MEMORY; goto cleanup; }
    current->nchg = 0;                     /* root carries no bound changes */

    /* ---- search ---- */
    for (;;) {
        double obj = 0.0, cutoff;
        sk_result r;
        int bvar;
        double bfrac = 0.0;

        if (!current) {
            current = heap_pop(&open);
            if (!current) { final = M.have_incumbent ? SK_RESULT_OPTIMAL : SK_RESULT_INFEASIBLE; break; }
        }
        if (M.time_limit > 0.0 && sk_wall_seconds() - M.t0 > M.time_limit) {
            final = SK_RESULT_TIME_LIMIT; node_free(current); current = NULL; break;
        }
        if (o->node_limit > 0 && M.st.nodes >= o->node_limit) {
            final = SK_RESULT_ITERATION_LIMIT; node_free(current); current = NULL; break;
        }

        /* weakest open bound, including the node in hand */
        best_bound = current->bound;
        for (j = 0; j < open.n; j++) if (open.a[j]->bound < best_bound) best_bound = open.a[j]->bound;
        if (M.have_incumbent) {
            double gap = fabs(M.incumbent_obj - best_bound);
            if (gap <= M.gap_abs || gap / (1e-10 + fabs(M.incumbent_obj)) <= M.gap_rel) {
                final = SK_RESULT_OPTIMAL; node_free(current); current = NULL; break;
            }
        }

        cutoff = M.have_incumbent ? M.incumbent_obj - M.gap_abs : MILP_INF;
        if (current->bound >= cutoff) { node_free(current); current = NULL; continue; }

        milp_apply(&M, current);
        if (current->depth > M.st.max_depth) M.st.max_depth = current->depth;
        if (!milp_propagate(&M, 4)) { node_free(current); current = NULL; continue; }

        M.st.nodes++;
        r = milp_solve_lp(&M, &obj);
        if (r == SK_RESULT_TIME_LIMIT) { final = SK_RESULT_TIME_LIMIT; node_free(current); current = NULL; break; }
        if (r != SK_RESULT_OPTIMAL) { node_free(current); current = NULL; continue; }
        if (obj >= cutoff) { node_free(current); current = NULL; continue; }

        /* pseudocost feedback from the branch that produced this node */
        if (current->nchg > 0) {
            const bchg *last = &current->chg[current->nchg - 1];
            milp_update_pseudocost(&M, last->var, !last->upper,
                                   obj - current->bound, current->frac);
        }

        bvar = milp_select_branch(&M, M.ls.x, &bfrac);
        if (bvar < 0) {                     /* integral: a new incumbent */
            if (!M.have_incumbent || obj < M.incumbent_obj - 1e-12) milp_accept(&M, M.ls.x, obj);
            node_free(current); current = NULL; continue;
        }

        if (o->mip_heuristics && (M.st.nodes % 64) == 1) milp_heuristic_round(&M, M.ls.x);

        {
            double xv = M.ls.x[bvar];
            node *down = node_child(current, bvar, 1, floor(xv), obj, bfrac);
            node *up   = node_child(current, bvar, 0, ceil(xv),  obj, 1.0 - bfrac);
            if (!down || !up) { node_free(down); node_free(up); node_free(current); rc = SK_ERR_MEMORY; goto cleanup; }
            node_free(current);
            /* dive toward the side the pseudocosts say is cheaper */
            if (M.pc_down[bvar] * bfrac <= M.pc_up[bvar] * (1.0 - bfrac)) {
                if (!heap_push(&open, up)) { rc = SK_ERR_MEMORY; goto cleanup; }
                current = down;
            } else {
                if (!heap_push(&open, down)) { rc = SK_ERR_MEMORY; goto cleanup; }
                current = up;
            }
        }
    }

finish:
    if (final == SK_RESULT_NOT_RUN) final = SK_RESULT_NUMERIC_FAILURE;
    {
        double bb = M.have_incumbent ? M.incumbent_obj : MILP_INF;
        for (j = 0; j < open.n; j++) if (open.a[j]->bound < bb) bb = open.a[j]->bound;
        if (!M.have_incumbent && final != SK_RESULT_INFEASIBLE) bb = M.st.root_bound;
        M.st.best_bound = bb;
    }

    if (!s->x) s->x = (double *)calloc((size_t)m->ncol + 1, sizeof(double));
    if (!s->x) { rc = SK_ERR_MEMORY; goto cleanup; }
    s->ncol = m->ncol; s->nrow = m->nrow;
    if (M.have_incumbent) {
        memcpy(s->x, M.incumbent, (size_t)m->ncol * sizeof(double));
        s->objective = M.incumbent_obj;
        s->dual_bound = M.st.best_bound;
        s->mip_gap = fabs(M.incumbent_obj - M.st.best_bound) / (1e-10 + fabs(M.incumbent_obj));
        if (final != SK_RESULT_OPTIMAL && final != SK_RESULT_TIME_LIMIT &&
            final != SK_RESULT_ITERATION_LIMIT) final = SK_RESULT_GAP_LIMIT;
    } else {
        s->objective = NAN;
        s->dual_bound = M.st.best_bound;
        s->mip_gap = INFINITY;
    }
    s->result = final;
    s->nodes = M.st.nodes;
    s->iterations = M.st.simplex_iterations;
    s->solve_seconds = sk_wall_seconds() - M.t0;
    M.st.gap_abs = M.have_incumbent ? fabs(M.incumbent_obj - M.st.best_bound) : INFINITY;
    M.st.gap_rel = s->mip_gap;
    if (stats) *stats = M.st;

    /* independent certificate on the reported point */
    if (M.have_incumbent) {
        double *savelo = M.work.clow, *savehi = M.work.cupp;
        M.work.clow = (double *)m->clow; M.work.cupp = (double *)m->cupp;
        sk_verify(m, s);
        M.work.clow = savelo; M.work.cupp = savehi;
        s->result = final;                  /* verify must not restate status */
    }

cleanup:
    for (j = 0; j < open.n; j++) node_free(open.a[j]);
    free(open.a);
    node_free(current);
    free(M.clow0); free(M.cupp0); free(M.nlow); free(M.nupp);
    free(M.pc_down); free(M.pc_up); free(M.nc_down); free(M.nc_up);
    free(M.incumbent);
    sk_solution_free(&M.ls);
    return rc;
}
