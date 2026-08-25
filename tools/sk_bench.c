/* Native benchmark harness. One JSON object is emitted per MPS/QPS file so
 * benchmark runs can be archived and compared without a Python runtime. */
#include "sankhya.h"
#include "sk_milp.h"
#include "sk_simplex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define SK_BENCH_COMPILER "MSVC"
#elif defined(__clang__)
#define SK_BENCH_COMPILER "Clang"
#elif defined(__GNUC__)
#define SK_BENCH_COMPILER "GCC"
#else
#define SK_BENCH_COMPILER "unknown"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define SK_BENCH_MACHINE "x86_64"
#elif defined(_M_ARM64) || defined(__aarch64__)
#define SK_BENCH_MACHINE "aarch64"
#else
#define SK_BENCH_MACHINE "unknown"
#endif

static void usage(void)
{
    fprintf(stderr, "usage: sk_bench [options] file1.mps [file2.qps ...]\n"
                    "  --time-limit S    per-instance wall limit\n"
                    "  --iterations N    per-instance iteration limit\n"
                    "  --nodes N         per-instance node limit\n");
}

static void json_string(const char *text)
{
    const unsigned char *p = (const unsigned char *)text;
    putchar('"');
    for (; *p; ++p) {
        if (*p == '\\' || *p == '"') { putchar('\\'); putchar(*p); }
        else if (*p == '\n') fputs("\\n", stdout);
        else if (*p == '\r') fputs("\\r", stdout);
        else if (*p == '\t') fputs("\\t", stdout);
        else if (*p < 0x20) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}

static void json_number(double value)
{
    if (isfinite(value) && fabs(value) < SK_INFINITY) printf("%.17g", value);
    else fputs("null", stdout);
}

int main(int argc, char **argv)
{
    sk_options o;
    int i, files = 0, failures = 0;
    sk_options_default(&o);
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--time-limit") && i + 1 < argc) o.time_limit = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) o.iteration_limit = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--nodes") && i + 1 < argc) o.node_limit = atoll(argv[++i]);
        else if (argv[i][0] == '-') { usage(); return 2; }
        else ++files;
    }
    if (!files) { usage(); return 2; }

    for (i = 1; i < argc; ++i) {
        sk_model m;
        sk_solution s;
        sk_milp_stats ms;
        sk_spx_stats ss;
        sk_status rc;
        double t0, seconds, read_seconds;
        int integers;
        if (argv[i][0] == '-') {
            if (i + 1 < argc && (!strcmp(argv[i], "--time-limit") ||
                !strcmp(argv[i], "--iterations") || !strcmp(argv[i], "--nodes"))) ++i;
            continue;
        }
        sk_model_init(&m); sk_solution_init(&s);
        memset(&ms, 0, sizeof(ms)); memset(&ss, 0, sizeof(ss));
        t0 = sk_wall_seconds(); rc = sk_read_mps(argv[i], &m); read_seconds = sk_wall_seconds() - t0;
        if (rc != SK_OK) {
            fputs("{\"file\":", stdout); json_string(argv[i]);
            printf(",\"status\":\"read_error\",\"detail\":\"%s\"}\n", sk_status_name(rc));
            sk_model_free(&m); sk_solution_free(&s); ++failures; continue;
        }
        integers = sk_model_num_integer(&m);
        t0 = sk_wall_seconds();
        if (m.Q || integers) rc = sk_solve(&m, &o, &s);
        else rc = sk_simplex_solve(&m, &o, &s, &ss);
        seconds = sk_wall_seconds() - t0;
        if (rc != SK_OK) {
            fputs("{\"file\":", stdout); json_string(argv[i]);
            printf(",\"status\":\"solve_error\",\"detail\":\"%s\"}\n", sk_status_name(rc));
            sk_model_free(&m); sk_solution_free(&s); ++failures; continue;
        }
        if (s.x) sk_verify(&m, &s);
        fputs("{\"file\":", stdout); json_string(argv[i]);
        fputs(",\"solver_version\":", stdout); json_string(sk_version());
        fputs(",\"compiler\":", stdout); json_string(SK_BENCH_COMPILER);
        fputs(",\"machine\":", stdout); json_string(SK_BENCH_MACHINE);
        printf(",\"rows\":%d,\"cols\":%d,\"nnz\":%d,\"integers\":%d,\"quadratic\":%s,\"status\":\"%s\",\"objective\":",
               m.nrow, m.ncol, m.A.p[m.ncol], integers, m.Q ? "true" : "false", sk_result_name(s.result));
        json_number(s.objective); fputs(",\"dual_bound\":", stdout); json_number(s.dual_bound);
        fputs(",\"mip_gap\":", stdout); json_number(s.mip_gap);
        fputs(",\"primal_inf\":", stdout); json_number(s.primal_infeasibility);
        fputs(",\"dual_inf\":", stdout); json_number(s.dual_infeasibility);
        fputs(",\"complementarity\":", stdout); json_number(s.complementarity);
        printf(",\"iterations\":%lld,\"nodes\":%lld,\"read_seconds\":", s.iterations, s.nodes);
        json_number(read_seconds); fputs(",\"solve_seconds\":", stdout); json_number(seconds); puts("}");
        sk_model_free(&m); sk_solution_free(&s);
    }
    return failures ? 1 : 0;
}
