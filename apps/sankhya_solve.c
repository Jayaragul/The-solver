#include "sankhya.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_long_long(const char *text, long long *value)
{
    char *end = NULL;
    long long parsed = strtoll(text, &end, 10);
    if (!end || *end != '\0' || parsed <= 0) return 0;
    *value = parsed;
    return 1;
}

int main(int argc, char **argv)
{
    sk_model model;
    sk_solution solution;
    sk_options options;
    int i;
    if (argc < 2) {
        fprintf(stderr, "usage: sankhya_solve <model.mps|model.qps> [--iterations N] [--nodes N] [--time seconds]\n");
        return 2;
    }
    sk_options_default(&options);
    for (i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!parse_long_long(argv[++i], &options.iteration_limit)) return 2;
        } else if (strcmp(argv[i], "--nodes") == 0 && i + 1 < argc) {
            if (!parse_long_long(argv[++i], &options.node_limit)) return 2;
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            char *end = NULL;
            options.time_limit = strtod(argv[++i], &end);
            if (!end || *end != '\0' || options.time_limit <= 0.0) return 2;
        } else {
            fprintf(stderr, "unrecognized option: %s\n", argv[i]);
            return 2;
        }
    }
    sk_model_init(&model);
    sk_solution_init(&solution);
    if (sk_read_mps(argv[1], &model) != SK_OK) {
        fprintf(stderr, "read_error=%s\n", sk_status_name(SK_ERR_IO));
        return 3;
    }
    if (sk_solve(&model, &options, &solution) != SK_OK) {
        fprintf(stderr, "solve_error\n");
        sk_solution_free(&solution); sk_model_free(&model);
        return 4;
    }
    if (solution.x) (void)sk_verify(&model, &solution);
    printf("status=%s iterations=%lld nodes=%lld objective=%.17g solve_seconds=%.9g\n",
        sk_result_name(solution.result), solution.iterations, solution.nodes,
        solution.objective, solution.solve_seconds);
    printf("primal_infeasibility=%.17g mip_gap=%.17g\n",
        solution.primal_infeasibility, solution.mip_gap);
    if (model.Q) {
        printf("kkt_dual_infeasibility=%.17g kkt_complementarity=%.17g\n",
            solution.dual_infeasibility, solution.complementarity);
    } else {
        printf("certificate_status=lp_pdhg_primal_only\n");
    }
    for (i = 0; i < solution.ncol; ++i)
        printf("x[%s]=%.17g\n", model.colname && model.colname[i] ? model.colname[i] : "?", solution.x[i]);
    {
        const int exit_code = (solution.result == SK_RESULT_OPTIMAL || solution.result == SK_RESULT_GAP_LIMIT) ? 0 : 5;
        sk_solution_free(&solution); sk_model_free(&model);
        return exit_code;
    }
}
