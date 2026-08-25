/* Bounded-variable revised simplex.
 *
 * Solves   min c'x   s.t.  rlow <= Ax <= rupp,  clow <= x <= cupp
 * by working on the equality form  [A  -I] (x; s) = 0  with  rlow <= s <= rupp,
 * so every row contributes one logical variable and the starting basis is the
 * full set of logicals.  Bounds are handled directly on the nonbasic variables
 * rather than by splitting variables, which keeps the basis at m x m.
 */
#ifndef SK_SIMPLEX_H
#define SK_SIMPLEX_H

#include "sankhya.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Nonbasic position of a variable. */
enum {
    SK_BASIC     = 0,
    SK_AT_LOWER  = 1,
    SK_AT_UPPER  = 2,
    SK_FREE_ZERO = 3
};

typedef struct sk_spx_stats {
    long long iterations;
    long long phase1_iterations;
    long long refactorizations;
    long long bound_flips;
    long long bland_episodes;   /* times the anti-cycling rule engaged */
    double    phase1_seconds;
    double    phase2_seconds;
    double    final_primal_infeasibility;
} sk_spx_stats;

/* Solve the LP relaxation of `m` (integrality is ignored here).
 * On success `s` carries x, y, reduced costs and row activities. */
sk_status sk_simplex_solve(const sk_model *m, const sk_options *o,
                           sk_solution *s, sk_spx_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
