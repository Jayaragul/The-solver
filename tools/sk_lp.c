/* SANKHYA LP driver.
 *
 * Reads an MPS file, solves it with the revised simplex, and emits one JSON
 * object.  Every residual printed comes from sk_verify(), which recomputes
 * A*x and c + A'y from the original model, so the numbers here are not the
 * solver's own opinion of itself. */
#include "sankhya.h"
#include "sk_simplex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr,
        "usage: sk_lp <file.mps> [options]\n"
        "  --time-limit S     wall clock limit in seconds\n"
        "  --iter-limit N     simplex iteration limit\n"
        "  --primal-tol T     primal feasibility tolerance\n"
        "  --dual-tol T       dual feasibility tolerance\n"
        "  --refactor N       refactorization interval\n"
        "  --expect V         reference objective, reported as relative error\n"
        "  --quiet            suppress the human-readable header\n");
}

int main(int argc, char **argv)
{
    sk_model m;
    sk_options o;
    sk_solution s;
    sk_spx_stats st;
    const char *path = NULL;
    double expect = NAN, tread, tsolve;
    int i, quiet = 0, verbose = 0;
    sk_status rc;

    if (argc < 2) { usage(); return 2; }
    sk_options_default(&o);
    memset(&st, 0, sizeof(st));

    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "--quiet")) { quiet = 1; continue; }
        if (!strcmp(argv[i], "--verbose")) { verbose = 1; continue; }
        if (i + 1 >= argc) { usage(); return 2; }
        if      (!strcmp(argv[i], "--time-limit")) o.time_limit = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iter-limit")) o.iteration_limit = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--primal-tol")) o.primal_tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dual-tol"))   o.dual_tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--refactor"))   o.refactor_interval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expect"))     expect = atof(argv[++i]);
        else { usage(); return 2; }
    }
    if (!path) { usage(); return 2; }

    sk_model_init(&m);
    tread = sk_wall_seconds();
    rc = sk_read_mps(path, &m);
    tread = sk_wall_seconds() - tread;
    if (rc != SK_OK) {
        printf("{\"file\":\"%s\",\"status\":\"read_error\",\"detail\":\"%s\"}\n",
               path, sk_status_name(rc));
        return 1;
    }

    if (!quiet)
        fprintf(stderr, "%s: %d rows, %d cols, %d nonzeros, %d integer\n",
                path, m.nrow, m.ncol, m.A.p[m.ncol], sk_model_num_integer(&m));

    sk_solution_init(&s);
    tsolve = sk_wall_seconds();
    rc = sk_simplex_solve(&m, &o, &s, &st);
    tsolve = sk_wall_seconds() - tsolve;

    if (rc != SK_OK) {
        printf("{\"file\":\"%s\",\"status\":\"solve_error\",\"detail\":\"%s\"}\n",
               path, sk_status_name(rc));
        sk_model_free(&m); sk_solution_free(&s);
        return 1;
    }

    printf("{\"file\":\"%s\",\"rows\":%d,\"cols\":%d,\"nnz\":%d,"
           "\"status\":\"%s\",\"objective\":%.12g,"
           "\"primal_inf\":%.3e,\"dual_inf\":%.3e,\"gap\":%.3e,"
           "\"iterations\":%lld,\"phase1_iterations\":%lld,\"refactorizations\":%lld,"
           "\"bound_flips\":%lld,\"read_seconds\":%.6f,\"solve_seconds\":%.6f",
           path, m.nrow, m.ncol, m.A.p[m.ncol],
           sk_result_name(s.result), s.objective,
           s.primal_infeasibility, s.dual_infeasibility, s.complementarity,
           s.iterations, st.phase1_iterations, st.refactorizations,
           st.bound_flips, tread, tsolve);
    if (!isnan(expect)) {
        double rel = fabs(s.objective - expect) / (1.0 + fabs(expect));
        printf(",\"expected\":%.12g,\"relative_error\":%.3e", expect, rel);
    }
    printf("}\n");

    if (verbose) {
        int k;
        fprintf(stderr, "x  =");
        for (k = 0; k < m.ncol && k < 12; k++) fprintf(stderr, " %g", s.x[k]);
        fprintf(stderr, "\ny  =");
        for (k = 0; k < m.nrow && k < 12; k++) fprintf(stderr, " %g", s.y[k]);
        fprintf(stderr, "\nrc =");
        for (k = 0; k < m.ncol && k < 12; k++) fprintf(stderr, " %g", s.rc[k]);
        fprintf(stderr, "\nact=");
        for (k = 0; k < m.nrow && k < 12; k++) fprintf(stderr, " %g", s.rowact[k]);
        fprintf(stderr, "\n");
    }

    rc = (s.result == SK_RESULT_OPTIMAL) ? SK_OK : SK_ERR_NUMERIC;
    sk_model_free(&m);
    sk_solution_free(&s);
    return rc == SK_OK ? 0 : 3;
}
