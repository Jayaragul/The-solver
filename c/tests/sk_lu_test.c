/* Regression tests for the sparse LU: FTRAN, BTRAN and product-form updates.
 *
 * Every check is a residual against the original matrix, so a wrong
 * permutation or a mis-ordered eta file cannot pass. */
#include "sk_lu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng_state = 88172645463325252ULL;
static double rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)((rng_state >> 11) & 0xFFFFFFFFULL) / 4294967296.0;
}

/* Dense-times-vector against the CSC form, used only by the tests. */
static void csc_mv(const sk_csc *A, const double *x, double *y)
{
    int i, j, p;
    for (i = 0; i < A->nrow; i++) y[i] = 0.0;
    for (j = 0; j < A->ncol; j++)
        for (p = A->p[j]; p < A->p[j + 1]; p++) y[A->i[p]] += A->x[p] * x[j];
}

static void csc_mtv(const sk_csc *A, const double *x, double *y)
{
    int j, p;
    for (j = 0; j < A->ncol; j++) {
        double s = 0.0;
        for (p = A->p[j]; p < A->p[j + 1]; p++) s += A->x[p] * x[A->i[p]];
        y[j] = s;
    }
}

static double maxdiff(const double *a, const double *b, int n)
{
    int i;
    double m = 0.0;
    for (i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

/* Build a random n x n matrix with a guaranteed-nonzero diagonal so it is
 * nonsingular with probability 1, plus `extra` off-diagonal entries. */
static int build_random(sk_csc *A, int n, int extra)
{
    int j, k, p = 0, cap;
    int *cnt = (int *)calloc((size_t)n + 1, sizeof(int));
    int **rows = (int **)calloc((size_t)n + 1, sizeof(int *));
    double **vals = (double **)calloc((size_t)n + 1, sizeof(double *));
    int per = extra / (n > 0 ? n : 1) + 1;

    /* Each column has its diagonal plus up to `per` distinct off-diagonal
       entries.  The old n+extra allocation was too small whenever `per`
       rounded up, corrupting the regression matrix at n=40 and beyond. */
    cap = n * (per + 1) + 1;

    A->nrow = A->ncol = n;
    A->nzmax = cap;
    A->p = (int *)malloc(((size_t)n + 1) * sizeof(int));
    A->i = (int *)malloc((size_t)cap * sizeof(int));
    A->x = (double *)malloc((size_t)cap * sizeof(double));
    if (!cnt || !rows || !vals || !A->p || !A->i || !A->x) return 0;

    for (j = 0; j < n; j++) {
        rows[j] = (int *)malloc((size_t)(per + 2) * sizeof(int));
        vals[j] = (double *)malloc((size_t)(per + 2) * sizeof(double));
        if (!rows[j] || !vals[j]) return 0;
        rows[j][0] = j;                       /* diagonal, dominant */
        vals[j][0] = 4.0 + rnd();
        cnt[j] = 1;
        for (k = 0; k < per && cnt[j] < per + 1; k++) {
            int r = (int)(rnd() * n);
            int d, dup = 0;
            if (r < 0 || r >= n) continue;
            for (d = 0; d < cnt[j]; d++) if (rows[j][d] == r) { dup = 1; break; }
            if (dup) continue;
            rows[j][cnt[j]] = r;
            vals[j][cnt[j]] = rnd() * 2.0 - 1.0;
            cnt[j]++;
        }
    }
    for (j = 0; j < n; j++) {
        A->p[j] = p;
        for (k = 0; k < cnt[j] && p < cap; k++) { A->i[p] = rows[j][k]; A->x[p] = vals[j][k]; p++; }
        free(rows[j]); free(vals[j]);
    }
    A->p[n] = p;
    free(cnt); free(rows); free(vals);
    return 1;
}

static void free_csc(sk_csc *A) { free(A->p); free(A->i); free(A->x); memset(A, 0, sizeof(*A)); }

static int fail(const char *what, double got, double tol)
{
    printf("  FAIL %-28s residual %.3e (tol %.1e)\n", what, got, tol);
    return 1;
}

int main(void)
{
    int bad = 0;
    int sizes[] = { 1, 2, 5, 40, 300, 1200 };
    int t;

    printf("sparse LU regression\n");

    for (t = 0; t < (int)(sizeof(sizes) / sizeof(sizes[0])); t++) {
        int n = sizes[t];
        sk_csc A;
        sk_lu lu;
        double *b, *x, *chk;
        double r1, r2;
        int i;

        memset(&A, 0, sizeof(A));
        if (!build_random(&A, n, n * 4)) { printf("  alloc failed\n"); return 1; }
        sk_lu_init(&lu);
        if (sk_lu_factorize(&lu, &A, 0.1) != SK_OK) { printf("  factorize failed n=%d\n", n); return 1; }

        b   = (double *)malloc((size_t)n * sizeof(double));
        x   = (double *)malloc((size_t)n * sizeof(double));
        chk = (double *)malloc((size_t)n * sizeof(double));
        for (i = 0; i < n; i++) b[i] = rnd() * 2.0 - 1.0;

        /* FTRAN: B x = b */
        memcpy(x, b, (size_t)n * sizeof(double));
        if (sk_lu_ftran(&lu, x) != SK_OK) { printf("  ftran failed\n"); return 1; }
        csc_mv(&A, x, chk);
        r1 = maxdiff(chk, b, n);

        /* BTRAN: B' x = b */
        memcpy(x, b, (size_t)n * sizeof(double));
        if (sk_lu_btran(&lu, x) != SK_OK) { printf("  btran failed\n"); return 1; }
        csc_mtv(&A, x, chk);
        r2 = maxdiff(chk, b, n);

        printf("  n=%-5d nnz=%-7d fill=%-8.0f ftran=%.2e btran=%.2e\n",
               n, A.p[n], sk_lu_fill(&lu), r1, r2);
        if (r1 > 1e-9) bad += fail("ftran", r1, 1e-9);
        if (r2 > 1e-9) bad += fail("btran", r2, 1e-9);

        free(b); free(x); free(chk);
        sk_lu_free(&lu);
        free_csc(&A);
    }

    /* ---- product-form update: replace a column and re-solve ---- */
    {
        int n = 200, k, rep;
        sk_csc A;
        sk_lu lu;
        double *b, *x, *chk, *alpha, *newcol;
        double worst = 0.0;
        int i, p;

        memset(&A, 0, sizeof(A));
        if (!build_random(&A, n, n * 3)) return 1;
        sk_lu_init(&lu);
        if (sk_lu_factorize(&lu, &A, 0.1) != SK_OK) { printf("  factorize failed\n"); return 1; }

        b      = (double *)malloc((size_t)n * sizeof(double));
        x      = (double *)malloc((size_t)n * sizeof(double));
        chk    = (double *)malloc((size_t)n * sizeof(double));
        alpha  = (double *)malloc((size_t)n * sizeof(double));
        newcol = (double *)malloc((size_t)n * sizeof(double));

        for (rep = 0; rep < 12; rep++) {
            /* choose a position to replace and a dense-ish replacement column */
            k = (int)(rnd() * n);
            if (k < 0 || k >= n) k = rep % n;
            for (i = 0; i < n; i++) newcol[i] = 0.0;
            newcol[k] = 3.0 + rnd();                 /* keep it well conditioned */
            for (i = 0; i < 5; i++) {
                int r = (int)(rnd() * n);
                if (r >= 0 && r < n && r != k) newcol[r] = rnd() * 2.0 - 1.0;
            }

            /* alpha = B^{-1} * newcol with the *current* factor, then update */
            memcpy(alpha, newcol, (size_t)n * sizeof(double));
            if (sk_lu_ftran(&lu, alpha) != SK_OK) { printf("  ftran failed\n"); return 1; }
            if (sk_lu_update(&lu, k, alpha, 1e-14) != SK_OK) { printf("  update failed\n"); return 1; }

            /* mutate the reference matrix the same way: column k := newcol */
            {
                int nzk = 0;
                for (i = 0; i < n; i++) if (newcol[i] != 0.0) nzk++;
                if (nzk <= A.p[k + 1] - A.p[k]) {
                    p = A.p[k];
                    for (i = 0; i < n; i++) if (newcol[i] != 0.0) { A.i[p] = i; A.x[p] = newcol[i]; p++; }
                    while (p < A.p[k + 1]) { A.i[p] = 0; A.x[p] = 0.0; p++; }
                } else {
                    /* rebuild with room */
                    int *ni = (int *)malloc((size_t)(A.p[n] + n) * sizeof(int));
                    double *nx = (double *)malloc((size_t)(A.p[n] + n) * sizeof(double));
                    int *np = (int *)malloc(((size_t)n + 1) * sizeof(int));
                    int j, q = 0;
                    for (j = 0; j < n; j++) {
                        np[j] = q;
                        if (j == k) { for (i = 0; i < n; i++) if (newcol[i] != 0.0) { ni[q] = i; nx[q] = newcol[i]; q++; } }
                        else for (p = A.p[j]; p < A.p[j + 1]; p++) { ni[q] = A.i[p]; nx[q] = A.x[p]; q++; }
                    }
                    np[n] = q;
                    free(A.p); free(A.i); free(A.x);
                    A.p = np; A.i = ni; A.x = nx;
                }
            }

            for (i = 0; i < n; i++) b[i] = rnd() * 2.0 - 1.0;
            memcpy(x, b, (size_t)n * sizeof(double));
            if (sk_lu_ftran(&lu, x) != SK_OK) return 1;
            csc_mv(&A, x, chk);
            if (maxdiff(chk, b, n) > worst) worst = maxdiff(chk, b, n);

            memcpy(x, b, (size_t)n * sizeof(double));
            if (sk_lu_btran(&lu, x) != SK_OK) return 1;
            csc_mtv(&A, x, chk);
            if (maxdiff(chk, b, n) > worst) worst = maxdiff(chk, b, n);
        }
        printf("  eta updates (12 replacements, n=200): worst residual %.2e\n", worst);
        if (worst > 1e-8) bad += fail("eta update", worst, 1e-8);

        free(b); free(x); free(chk); free(alpha); free(newcol);
        sk_lu_free(&lu);
        free_csc(&A);
    }

    /* ---- singular matrix must be detected, not silently accepted ---- */
    {
        sk_csc A;
        sk_lu lu;
        int p[4] = { 0, 2, 4, 6 };
        int ri[6] = { 0, 1, 0, 1, 0, 1 };       /* rank 2, third column dependent */
        double v[6] = { 1.0, 2.0, 2.0, 4.0, 3.0, 6.0 };
        A.nrow = A.ncol = 3; A.nzmax = 6; A.p = p; A.i = ri; A.x = v;
        sk_lu_init(&lu);
        if (sk_lu_factorize(&lu, &A, 0.1) != SK_ERR_SINGULAR) {
            printf("  FAIL singular matrix not detected\n");
            bad++;
        } else {
            printf("  singular detection: ok\n");
        }
        sk_lu_free(&lu);
    }

    printf(bad ? "FAILED (%d)\n" : "all sparse LU checks passed\n", bad);
    return bad ? 1 : 0;
}
