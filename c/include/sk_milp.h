/* Branch-and-bound for mixed-integer linear programming and the guarded
 * diagonal-MIQP slice.
 *
 * LP node relaxations use the revised simplex; admitted diagonal-QP node
 * relaxations use the exact continuous-QP path. This module owns only the
 * search: bound propagation, branching, heuristics and incumbent/bound
 * bookkeeping that proves optimality.
 */
#ifndef SK_MILP_H
#define SK_MILP_H

#include "sankhya.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_milp_stats {
    long long nodes;              /* nodes whose relaxation was solved      */
    long long lp_solves;          /* includes heuristic and probing solves  */
    long long simplex_iterations;
    long long cuts_added;
    long long heuristic_hits;     /* incumbents found by a heuristic        */
    long long propagations;       /* variable bounds tightened              */
    double    root_bound;         /* LP relaxation value at the root        */
    double    best_bound;         /* proven bound at termination            */
    double    gap_abs;
    double    gap_rel;
    int       solutions_found;
    int       max_depth;
} sk_milp_stats;

/* Solve the mixed-integer problem.  Continuous models are passed straight to
 * the simplex.  On return `s` holds the incumbent and `s->dual_bound` the
 * proven bound; `s->result` is SK_RESULT_OPTIMAL only when the gap closed. */
sk_status sk_milp_solve(const sk_model *m, const sk_options *o,
                        sk_solution *s, sk_milp_stats *st);

#ifdef __cplusplus
}
#endif

#endif
