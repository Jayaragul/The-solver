/* Row/column equilibration used by the simplex. */
#ifndef SK_SCALE_H
#define SK_SCALE_H

#include "sankhya.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fill R (length nrow) and C (length ncol) with power-of-two scale factors
 * that drive the entries of R*A*C toward unit magnitude.  The caller solves
 * the scaled model and recovers x = C*x~ and y = R*y~. */
void sk_scale_model(const sk_model *m, double *R, double *C, int passes);

#ifdef __cplusplus
}
#endif

#endif
