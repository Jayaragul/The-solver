/* SANKHYA - sovereign mathematical optimization core.
 *
 * Public C API.  The solver is self-contained: no external LP/QP library is
 * linked, and every numerical kernel in this tree is written from scratch.
 */
#ifndef SANKHYA_H
#define SANKHYA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ status */

typedef enum sk_status {
    SK_OK = 0,
    SK_ERR_ARG = 1,
    SK_ERR_MEMORY = 2,
    SK_ERR_STRUCTURE = 3,
    SK_ERR_IO = 4,
    SK_ERR_SINGULAR = 5,
    SK_ERR_UNSUPPORTED = 6,
    SK_ERR_NUMERIC = 7
} sk_status;

/* Terminal state of an optimization run. */
typedef enum sk_result {
    SK_RESULT_OPTIMAL = 0,
    SK_RESULT_INFEASIBLE = 1,
    SK_RESULT_UNBOUNDED = 2,
    SK_RESULT_ITERATION_LIMIT = 3,
    SK_RESULT_TIME_LIMIT = 4,
    SK_RESULT_NUMERIC_FAILURE = 5,
    SK_RESULT_NOT_RUN = 6,
    SK_RESULT_GAP_LIMIT = 7
} sk_result;

const char *sk_result_name(sk_result r);
const char *sk_status_name(sk_status s);

/* --------------------------------------------------------------- variables */

typedef enum sk_vartype {
    SK_CONTINUOUS = 0,
    SK_INTEGER    = 1
} sk_vartype;

/* Continuous solve engines, for sk_options.lp_engine. */
enum {
    SK_LP_AUTO        = 0,   /* simplex for LP, first-order for QP */
    SK_LP_SIMPLEX     = 1,   /* force the revised simplex          */
    SK_LP_FIRST_ORDER = 2    /* force the PDHG path                */
};

#define SK_INFINITY 1.0e30
#define SK_IS_INF(x)     ((x) >=  SK_INFINITY)
#define SK_IS_NEG_INF(x) ((x) <= -SK_INFINITY)

/* ------------------------------------------------------------------- model */

/* Column-compressed sparse matrix (CSC). */
typedef struct sk_csc {
    int     nrow;
    int     ncol;
    int    *p;      /* ncol + 1 column starts                */
    int    *i;      /* nnz row indices                       */
    double *x;      /* nnz values                            */
    int     nzmax;
} sk_csc;

/* An optimization instance in the canonical SANKHYA form:
 *
 *     minimize     c'x + 0.5 x'Qx + objshift
 *     subject to   rlow <= A x <= rupp
 *                  clow <=   x <= cupp
 *                  x_j integral for j with vartype[j] == SK_INTEGER
 *
 * Q is optional (NULL for LP/MILP) and is stored as the full symmetric matrix.
 */
typedef struct sk_model {
    int         nrow;
    int         ncol;
    sk_csc      A;
    sk_csc     *Q;          /* NULL unless quadratic                        */
    double     *c;
    double     *rlow;
    double     *rupp;
    double     *clow;
    double     *cupp;
    unsigned char *vartype;
    double      objshift;
    double      objsense;   /* +1 minimize, -1 maximize (applied to c and Q) */
    char      **rowname;
    char      **colname;
    char       *name;
} sk_model;

void      sk_model_init(sk_model *m);
void      sk_model_free(sk_model *m);
sk_status sk_model_alloc(sk_model *m, int nrow, int ncol, int nzmax);
/* Validate dimensions, canonical CSC structure, finite coefficients/costs,
 * bounds, and optional-Q dimensions before handing a model to a solver. */
sk_status sk_model_validate(const sk_model *m);
int       sk_model_num_integer(const sk_model *m);

/* MPS reader: fixed and free format, RANGES / BOUNDS / OBJSENSE, and the
 * QUADOBJ / QMATRIX sections used by QPS files. */
sk_status sk_read_mps(const char *path, sk_model *m);

/* ---------------------------------------------------------------- solution */

typedef struct sk_solution {
    sk_result result;
    double    objective;
    double   *x;        /* ncol primal values          */
    double   *y;        /* nrow duals (row activities' multipliers) */
    double   *rc;       /* ncol reduced costs          */
    double   *rowact;   /* nrow row activities         */
    int       ncol;
    int       nrow;
    /* diagnostics */
    long long iterations;
    long long nodes;
    double    solve_seconds;
    double    primal_infeasibility;
    double    dual_infeasibility;
    double    complementarity;
    double    mip_gap;
    double    dual_bound;
} sk_solution;

void sk_solution_init(sk_solution *s);
void sk_solution_free(sk_solution *s);

/* ---------------------------------------------------------------- options */

typedef struct sk_options {
    double primal_tol;       /* primal feasibility tolerance            */
    double dual_tol;         /* dual feasibility / optimality tolerance */
    double pivot_tol;        /* relative pivot threshold in LU          */
    double time_limit;       /* seconds, <=0 for none                   */
    long long iteration_limit;
    long long node_limit;
    int    presolve;         /* 0 off, 1 on                             */
    int    scaling;          /* 0 off, 1 geometric+equilibration        */
    int    refactor_interval;
    int    verbosity;        /* 0 silent, 1 summary, 2 iteration log    */
    int    threads;
    double mip_gap_abs;
    double mip_gap_rel;
    int    mip_cuts;
    int    mip_heuristics;
    int    dual_simplex;     /* prefer dual simplex where applicable    */
    /* Continuous engine selection.  SK_LP_AUTO uses the revised simplex for
       LPs, which is the exact, certificate-bearing path; the first-order
       method is retained for QP and for LPs too large for a basis factor. */
    int    lp_engine;
    unsigned long long random_seed;
} sk_options;

void sk_options_default(sk_options *o);

/* ------------------------------------------------------------------ solve */

/* Current native dispatch solves continuous LPs and convex QPs (Q optional)
 * by guarded exact paths or a primal-dual first-order method. Bounded MILPs
 * use exact-simplex branch-and-bound over certified LP relaxations. A guarded
 * MIQP slice admits only small positive-semidefinite Q in the active-set
 * certificate regime (or bounds-only closed form). Larger MIQP remains
 * unsupported. */
sk_status sk_solve(const sk_model *m, const sk_options *o, sk_solution *s);

/* Independently recomputes primal/dual residuals for a claimed solution.
 * This never consults solver-internal state - it is the certificate checker. */
sk_status sk_verify(const sk_model *m, sk_solution *s);

double sk_wall_seconds(void);
const char *sk_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SANKHYA_H */
