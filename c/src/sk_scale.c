/* Matrix equilibration for the simplex.
 *
 * Netlib carries several models whose coefficients span ten or more orders of
 * magnitude (greenbea, pilot, 80bau3b).  Unscaled, the ratio test and the
 * pivot threshold lose their meaning: what counts as an acceptable pivot in
 * one row is numerical noise in another, and the objective converges to a
 * handful of correct digits instead of twelve.  Repeatedly dividing each row
 * and column by sqrt(min*max) of its entries drives every coefficient toward
 * unit magnitude.
 *
 * The resulting factors are rounded to powers of two, so applying them is
 * exact in binary floating point and scaling contributes no rounding error of
 * its own.  The solver works on (R A C), recovering x = C x~ and y = R y~.
 */
#include "sk_scale.h"

#include <math.h>
#include <stdlib.h>

void sk_scale_model(const sk_model *m, double *R, double *C, int passes)
{
    int i, j, p, it;
    double *rmin = NULL, *rmax = NULL;

    if (!m || !R || !C) return;
    for (i = 0; i < m->nrow; i++) R[i] = 1.0;
    for (j = 0; j < m->ncol; j++) C[j] = 1.0;
    if (m->nrow == 0 || m->ncol == 0 || passes <= 0) return;

    rmin = (double *)malloc((size_t)m->nrow * sizeof(double));
    rmax = (double *)malloc((size_t)m->nrow * sizeof(double));
    if (!rmin || !rmax) { free(rmin); free(rmax); return; }

    for (it = 0; it < passes; it++) {
        for (j = 0; j < m->ncol; j++) {
            double mn = 0.0, mx = 0.0;
            for (p = m->A.p[j]; p < m->A.p[j + 1]; p++) {
                double v = fabs(m->A.x[p]) * R[m->A.i[p]] * C[j];
                if (v <= 0.0) continue;
                if (mn == 0.0 || v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (mx > 0.0) C[j] /= sqrt(mn * mx);
        }
        for (i = 0; i < m->nrow; i++) { rmin[i] = 0.0; rmax[i] = 0.0; }
        for (j = 0; j < m->ncol; j++) {
            for (p = m->A.p[j]; p < m->A.p[j + 1]; p++) {
                int r = m->A.i[p];
                double v = fabs(m->A.x[p]) * R[r] * C[j];
                if (v <= 0.0) continue;
                if (rmin[r] == 0.0 || v < rmin[r]) rmin[r] = v;
                if (v > rmax[r]) rmax[r] = v;
            }
        }
        for (i = 0; i < m->nrow; i++)
            if (rmax[i] > 0.0) R[i] /= sqrt(rmin[i] * rmax[i]);
    }

    for (i = 0; i < m->nrow; i++) {
        int e;
        if (!(R[i] > 0.0) || !isfinite(R[i])) { R[i] = 1.0; continue; }
        frexp(R[i], &e);
        R[i] = ldexp(1.0, e - 1);
    }
    for (j = 0; j < m->ncol; j++) {
        int e;
        if (!(C[j] > 0.0) || !isfinite(C[j])) { C[j] = 1.0; continue; }
        frexp(C[j], &e);
        C[j] = ldexp(1.0, e - 1);
    }
    free(rmin); free(rmax);
}
