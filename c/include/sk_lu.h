/* Sparse LU factorization of a simplex basis, with product-form updates.
 *
 * Factorization is left-looking Gilbert-Peierls with threshold partial
 * pivoting, run on a column ordering that first peels off the triangular
 * part of the basis so that only the residual "bump" incurs fill-in.
 *
 * The factor is used through FTRAN (B x = b) and BTRAN (B' x = b).  Basis
 * changes between refactorizations are absorbed by an eta file (product-form
 * update); the caller refactorizes when the file grows or accuracy degrades.
 */
#ifndef SK_LU_H
#define SK_LU_H

#include "sankhya.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_eta {
    int    r;        /* pivot position replaced          */
    int    nz;       /* stored entries of the eta column */
    int   *idx;
    double *val;
    double pivot;    /* alpha[r]                          */
} sk_eta;

typedef struct sk_lu {
    int  n;

    /* L is unit lower triangular, U is upper triangular, both CSC. */
    int    *Lp, *Li;  double *Lx;  int Lnz, Lmax;
    int    *Up, *Ui;  double *Ux;  int Unz, Umax;
    int    *pinv;     /* pinv[original row] = pivot position, size n */
    int    *q;        /* q[pivot position] = original column, size n */

    /* scratch */
    double *work;
    int    *xi;
    int    *pstack;
    int    *marker;

    /* eta file */
    sk_eta *eta;
    int     neta, etamax;
    int     eta_nz;

    double  pivot_tol;
    int     nfactor;      /* how many times factorized       */
    int     nupdate;      /* updates since last factorization */
    double  growth;       /* max |U| / max |B| after factorization */
} sk_lu;

void      sk_lu_init(sk_lu *lu);
void      sk_lu_free(sk_lu *lu);

/* Factorize the square matrix B (n x n, CSC).  B is not retained. */
sk_status sk_lu_factorize(sk_lu *lu, const sk_csc *B, double pivot_tol);

/* x := B^{-1} b  (b and x may alias; both are dense length-n arrays). */
sk_status sk_lu_ftran(sk_lu *lu, double *x);
/* x := B^{-T} b */
sk_status sk_lu_btran(sk_lu *lu, double *x);

/* Record a basis change: position r leaves, and `alpha` is the FTRAN of the
 * entering column (already computed with the *current* factor). */
sk_status sk_lu_update(sk_lu *lu, int r, const double *alpha, double drop_tol);

int    sk_lu_needs_refactor(const sk_lu *lu, int interval);
double sk_lu_fill(const sk_lu *lu);

#ifdef __cplusplus
}
#endif

#endif
