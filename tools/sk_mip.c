/* SANKHYA MILP driver.
 *
 * Reads an MPS file, runs branch-and-bound over simplex relaxations, and emits
 * one JSON object.  Residuals come from sk_verify(), which recomputes A*x from
 * the original model, and integrality is re-checked here rather than trusted
 * from the search. */
#include "sankhya.h"
#include "sk_milp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* JSON has no literal for NaN or infinity, and C's %g prints bare `nan` and
   `inf`, which no strict parser accepts.  A MILP legitimately produces both:
   an infeasible model has no objective, and an unexplored tree has an infinite
   dual bound.  Emit `null` so the record stays machine-readable and the
   consumer can distinguish "no value" from a real number. */
static void json_num(const char *key, double v, const char *fmt)
{
    char buf[64];
    printf("\"%s\":", key);
    if (v != v || v > 1e300 || v < -1e300) { printf("null"); return; }
    snprintf(buf, sizeof buf, fmt, v);
    printf("%s", buf);
}

static void usage(void)
{
    fprintf(stderr,
        "usage: sk_mip <file.mps> [options]\n"
        "  --time-limit S    wall clock limit in seconds\n"
        "  --node-limit N    branch-and-bound node limit\n"
        "  --gap-rel G       relative MIP gap tolerance\n"
        "  --gap-abs G       absolute MIP gap tolerance\n"
        "  --no-cuts         disable validated MILP cover cuts\n"
        "  --no-heuristics   disable primal heuristics\n"
        "  --expect V        reference objective, reported as relative error\n"
        "  --quiet           suppress the human-readable header\n");
}

int main(int argc, char **argv)
{
    sk_model m;
    sk_options o;
    sk_solution s;
    sk_milp_stats st;
    const char *path = NULL;
    double expect = NAN, tread, tsolve, maxint = 0.0;
    int i, j, quiet = 0;
    sk_status rc;

    if (argc < 2) { usage(); return 2; }
    sk_options_default(&o);
    memset(&st, 0, sizeof(st));

    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "--quiet")) { quiet = 1; continue; }
        if (!strcmp(argv[i], "--no-heuristics")) { o.mip_heuristics = 0; continue; }
        if (!strcmp(argv[i], "--no-cuts")) { o.mip_cuts = 0; continue; }
        if (i + 1 >= argc) { usage(); return 2; }
        if      (!strcmp(argv[i], "--time-limit")) o.time_limit = atof(argv[++i]);
        else if (!strcmp(argv[i], "--node-limit")) o.node_limit = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gap-rel"))    o.mip_gap_rel = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gap-abs"))    o.mip_gap_abs = atof(argv[++i]);
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
    rc = sk_milp_solve(&m, &o, &s, &st);
    tsolve = sk_wall_seconds() - tsolve;
    if (rc != SK_OK) {
        printf("{\"file\":\"%s\",\"status\":\"solve_error\",\"detail\":\"%s\"}\n",
               path, sk_status_name(rc));
        sk_model_free(&m); sk_solution_free(&s);
        return 1;
    }

    /* re-check integrality here, independently of the search */
    if (s.x)
        for (j = 0; j < m.ncol; j++)
            if (m.vartype[j] == SK_INTEGER) {
                double d = fabs(s.x[j] - floor(s.x[j] + 0.5));
                if (d > maxint) maxint = d;
            }

    printf("{\"file\":\"%s\",\"rows\":%d,\"cols\":%d,\"nnz\":%d,\"integers\":%d,"
           "\"status\":\"%s\",",
           path, m.nrow, m.ncol, m.A.p[m.ncol], sk_model_num_integer(&m),
           sk_result_name(s.result));
    json_num("objective", s.objective, "%.12g");   printf(",");
    json_num("dual_bound", s.dual_bound, "%.12g"); printf(",");
    json_num("mip_gap", s.mip_gap, "%.3e");        printf(",");
    printf("\"primal_inf\":%.3e,\"integrality_inf\":%.3e,"
           "\"nodes\":%lld,\"lp_solves\":%lld,\"simplex_iterations\":%lld,"
           "\"propagations\":%lld,\"cuts_added\":%lld,\"heuristic_hits\":%lld,"
           "\"solutions\":%d,\"max_depth\":%d,",
           s.primal_infeasibility, maxint,
           st.nodes, st.lp_solves, st.simplex_iterations,
           st.propagations, st.cuts_added, st.heuristic_hits, st.solutions_found,
           st.max_depth);
    json_num("root_bound", st.root_bound, "%.12g");
    printf(",\"read_seconds\":%.6f,\"solve_seconds\":%.6f", tread, tsolve);
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
