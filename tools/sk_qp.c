/* SANKHYA native QP driver.
 *
 * Reads a QPS/MPS model, solves the continuous convex-QP path, and emits one
 * machine-readable record.  The reported residuals are independently
 * recomputed by sk_verify() from the original sparse model. */
#include "sankhya.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr,
        "usage: sk_qp <file.qps> [options]\n"
        "  --time-limit S    wall clock limit in seconds\n"
        "  --iter-limit N    PDHG iteration limit\n"
        "  --threads N       OpenMP worker count for transpose-SpMV\n"
        "  --expect V        reference objective, reported as relative error\n"
        "  --quiet           suppress the human-readable header\n");
}

int main(int argc, char **argv)
{
    sk_model m;
    sk_options o;
    sk_solution s;
    const char *path = NULL;
    double expect = NAN, tread, tsolve;
    int i, quiet = 0;
    sk_status rc;

    if (argc < 2) { usage(); return 2; }
    sk_options_default(&o);
    for (i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "--quiet")) { quiet = 1; continue; }
        if (i + 1 >= argc) { usage(); return 2; }
        if      (!strcmp(argv[i], "--time-limit")) o.time_limit = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iter-limit")) o.iteration_limit = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--threads"))    o.threads = atoi(argv[++i]);
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
    if (!m.Q) {
        printf("{\"file\":\"%s\",\"status\":\"not_qp\"}\n", path);
        sk_model_free(&m);
        return 1;
    }
    if (!quiet)
        fprintf(stderr, "%s: %d rows, %d cols, %d A-nnz, %d Q-nnz\n",
                path, m.nrow, m.ncol, m.A.p[m.ncol], m.Q->p[m.Q->ncol]);

    sk_solution_init(&s);
    tsolve = sk_wall_seconds();
    rc = sk_solve(&m, &o, &s);
    tsolve = sk_wall_seconds() - tsolve;
    if (rc != SK_OK) {
        printf("{\"file\":\"%s\",\"status\":\"solve_error\",\"detail\":\"%s\"}\n",
               path, sk_status_name(rc));
        sk_model_free(&m); sk_solution_free(&s);
        return 1;
    }

    printf("{\"file\":\"%s\",\"rows\":%d,\"cols\":%d,\"a_nnz\":%d,\"q_nnz\":%d,"
           "\"status\":\"%s\",\"objective\":%.12g,\"primal_inf\":%.3e,"
           "\"dual_inf\":%.3e,\"complementarity\":%.3e,\"iterations\":%lld,"
           "\"read_seconds\":%.6f,\"solve_seconds\":%.6f",
           path, m.nrow, m.ncol, m.A.p[m.ncol], m.Q->p[m.Q->ncol],
           sk_result_name(s.result), s.objective, s.primal_infeasibility,
           s.dual_infeasibility, s.complementarity, s.iterations, tread, tsolve);
    if (!isnan(expect)) {
        double rel = fabs(s.objective - expect) / (1.0 + fabs(expect));
        printf(",\"expected\":%.12g,\"relative_error\":%.3e", expect, rel);
    }
    printf("}\n");

    rc = (s.result == SK_RESULT_OPTIMAL) ? SK_OK : SK_ERR_NUMERIC;
    sk_model_free(&m);
    sk_solution_free(&s);
    return rc == SK_OK ? 0 : 3;
}
