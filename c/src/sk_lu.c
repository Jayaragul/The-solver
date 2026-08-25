/* Sparse LU factorization with threshold partial pivoting.
 *
 * The numeric kernel is left-looking (Gilbert-Peierls): column k of the
 * factor comes from a sparse triangular solve L x = B(:,q[k]) whose nonzero
 * pattern is found by depth-first search on the graph of L.  Total work is
 * then proportional to the flop count of the factorization rather than to
 * n^2, which is what makes it affordable to refactorize the basis of a large
 * sparse LP thousands of times during a solve.
 *
 * Pivots are drawn from the numerically acceptable set { |x_i| >= tol*max|x| }
 * and, within it, from the sparsest row.  That is a one-sided Markowitz rule:
 * it suppresses fill-in without ever accepting a pivot that is small relative
 * to its column, so the growth factor stays bounded.
 */
#include "sk_lu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ memory */

static int sk_grow(int **idx, double **val, int need, int *cap)
{
    int newcap = *cap;
    int *ni;
    double *nv;
    if (need <= *cap) return 1;
    while (newcap < need) newcap = (newcap < 1024) ? 2048 : newcap * 2;
    ni = (int *)realloc(*idx, (size_t)newcap * sizeof(int));
    if (!ni) return 0;
    *idx = ni;
    nv = (double *)realloc(*val, (size_t)newcap * sizeof(double));
    if (!nv) return 0;
    *val = nv;
    *cap = newcap;
    return 1;
}

void sk_lu_init(sk_lu *lu)
{
    memset(lu, 0, sizeof(*lu));
    lu->pivot_tol = 0.1;
}

static void sk_lu_free_etas(sk_lu *lu)
{
    int i;
    for (i = 0; i < lu->neta; i++) {
        free(lu->eta[i].idx);
        free(lu->eta[i].val);
    }
    lu->neta = 0;
    lu->eta_nz = 0;
}

void sk_lu_free(sk_lu *lu)
{
    sk_lu_free_etas(lu);
    free(lu->eta);
    free(lu->Lp); free(lu->Li); free(lu->Lx);
    free(lu->Up); free(lu->Ui); free(lu->Ux);
    free(lu->pinv); free(lu->q);
    free(lu->work); free(lu->xi); free(lu->pstack); free(lu->marker);
    memset(lu, 0, sizeof(*lu));
    lu->pivot_tol = 0.1;
}

/* --------------------------------------------------------------- ordering */

/* Order the columns of B so the triangular part is peeled off first.
 *
 * Column singletons of the active submatrix are pivots that create no fill and
 * are emitted from the front; row singletons are emitted from the back.  What
 * survives is the "bump", ordered by increasing column count so that its
 * sparse columns are eliminated first.  A simplex basis is usually 80-95%
 * triangular, so this keeps the fill-in confined to a small kernel.
 *
 * Ordering only affects sparsity: partial pivoting keeps the factorization
 * correct for any permutation handed to it.
 */
static sk_status sk_lu_order(const sk_csc *B, int *q)
{
    int n = B->ncol;
    int *rowcnt = NULL, *colcnt = NULL, *stack = NULL;
    int *rowstart = NULL, *rowfill = NULL, *rowidx = NULL;
    unsigned char *rdone = NULL, *cdone = NULL;
    int front = 0, back = n - 1, i, j, p, top, nnz;
    sk_status st = SK_OK;

    nnz = B->p[n];
    rowcnt   = (int *)calloc((size_t)n + 1, sizeof(int));
    colcnt   = (int *)calloc((size_t)n + 1, sizeof(int));
    stack    = (int *)malloc(((size_t)n + 1) * sizeof(int));
    rowstart = (int *)malloc(((size_t)n + 2) * sizeof(int));
    rowfill  = (int *)malloc(((size_t)n + 2) * sizeof(int));
    rowidx   = (int *)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    rdone    = (unsigned char *)calloc((size_t)n + 1, 1);
    cdone    = (unsigned char *)calloc((size_t)n + 1, 1);
    if (!rowcnt || !colcnt || !stack || !rowstart || !rowfill || !rowidx ||
        !rdone || !cdone) { st = SK_ERR_MEMORY; goto done; }

    for (j = 0; j < n; j++) {
        colcnt[j] = B->p[j + 1] - B->p[j];
        for (p = B->p[j]; p < B->p[j + 1]; p++) rowcnt[B->i[p]]++;
    }
    /* row-wise pattern, so row singletons are cheap to detect */
    rowstart[0] = 0;
    for (i = 0; i < n; i++) rowstart[i + 1] = rowstart[i] + rowcnt[i];
    memcpy(rowfill, rowstart, (size_t)n * sizeof(int));
    for (j = 0; j < n; j++)
        for (p = B->p[j]; p < B->p[j + 1]; p++) rowidx[rowfill[B->i[p]]++] = j;

    /* --- column singletons to the front --- */
    top = 0;
    for (j = 0; j < n; j++) if (colcnt[j] == 1) stack[top++] = j;
    while (top > 0) {
        j = stack[--top];
        if (cdone[j] || colcnt[j] != 1) continue;
        i = -1;
        for (p = B->p[j]; p < B->p[j + 1]; p++)
            if (!rdone[B->i[p]]) { i = B->i[p]; break; }
        if (i < 0) continue;
        q[front++] = j;
        cdone[j] = 1; rdone[i] = 1;
        for (p = rowstart[i]; p < rowstart[i + 1]; p++) {
            int jj = rowidx[p];
            if (cdone[jj]) continue;
            if (--colcnt[jj] == 1 && top <= n) stack[top++] = jj;
        }
    }

    /* --- row singletons to the back --- */
    top = 0;
    for (i = 0; i < n; i++) {
        int live = 0;
        if (rdone[i]) { rowcnt[i] = 0; continue; }
        for (p = rowstart[i]; p < rowstart[i + 1]; p++) if (!cdone[rowidx[p]]) live++;
        rowcnt[i] = live;
        if (live == 1) stack[top++] = i;
    }
    while (top > 0) {
        i = stack[--top];
        if (rdone[i] || rowcnt[i] != 1) continue;
        j = -1;
        for (p = rowstart[i]; p < rowstart[i + 1]; p++)
            if (!cdone[rowidx[p]]) { j = rowidx[p]; break; }
        if (j < 0) continue;
        q[back--] = j;
        cdone[j] = 1; rdone[i] = 1;
        for (p = B->p[j]; p < B->p[j + 1]; p++) {
            int ii = B->i[p];
            if (rdone[ii]) continue;
            if (--rowcnt[ii] == 1 && top <= n) stack[top++] = ii;
        }
    }

    /* --- the bump, by increasing live column count (counting sort) --- */
    {
        int nb = back - front + 1;
        if (nb > 0) {
            int *order = (int *)malloc((size_t)nb * sizeof(int));
            int *tmp = (int *)malloc((size_t)nb * sizeof(int));
            int *bucket;
            int k = 0, c, maxc = 0, sum = 0;
            if (!order || !tmp) { free(order); free(tmp); st = SK_ERR_MEMORY; goto done; }
            for (j = 0; j < n; j++) {
                if (cdone[j]) continue;
                c = 0;
                for (p = B->p[j]; p < B->p[j + 1]; p++) if (!rdone[B->i[p]]) c++;
                colcnt[j] = c;
                if (c > maxc) maxc = c;
                if (k < nb) order[k++] = j;
            }
            nb = k;
            bucket = (int *)calloc((size_t)maxc + 2, sizeof(int));
            if (!bucket) { free(order); free(tmp); st = SK_ERR_MEMORY; goto done; }
            for (k = 0; k < nb; k++) bucket[colcnt[order[k]]]++;
            for (c = 0; c <= maxc; c++) { int t = bucket[c]; bucket[c] = sum; sum += t; }
            for (k = 0; k < nb; k++) tmp[bucket[colcnt[order[k]]]++] = order[k];
            for (k = 0; k < nb; k++) q[front + k] = tmp[k];
            free(order); free(tmp); free(bucket);
        }
    }

done:
    free(rowcnt); free(colcnt); free(stack);
    free(rowstart); free(rowfill); free(rowidx);
    free(rdone); free(cdone);
    return st;
}

/* ------------------------------------------------- depth-first reachability */

/* Iterative DFS from node j over the graph of L, pushing finished nodes onto
 * the output stack xi[top..n-1].  Recursion is unrolled explicitly because the
 * depth can reach n on a long dependency chain. */
static int sk_dfs(int j, const int *Lp, const int *Li, const int *pinv,
                  int top, int *xi, int *pstack, int *marker, int mark)
{
    int head = 0;
    xi[0] = j;
    while (head >= 0) {
        int jj = xi[head];
        int jnew = pinv[jj];
        int p, p2, done = 1;
        if (marker[jj] != mark) {
            marker[jj] = mark;
            pstack[head] = (jnew < 0) ? 0 : Lp[jnew];
        }
        p2 = (jnew < 0) ? 0 : Lp[jnew + 1];
        for (p = pstack[head]; p < p2; p++) {
            int i = Li[p];
            if (marker[i] == mark) continue;
            pstack[head] = p;
            xi[++head] = i;
            done = 0;
            break;
        }
        if (done) { head--; xi[--top] = jj; }
    }
    return top;
}

/* Solve L x = B(:,col) sparsely.  Returns `top`: the pattern is xi[top..n-1]
 * in topological order, values scattered into x.  L is indexed by original
 * row number here, since x lives in the unpermuted row space. */
static int sk_splsolve(const int *Lp, const int *Li, const double *Lx,
                       const sk_csc *B, int col, int *xi, double *x,
                       const int *pinv, int *pstack, int *marker, int mark)
{
    int n = B->nrow, top = n, p, px;
    for (p = B->p[col]; p < B->p[col + 1]; p++) {
        int i = B->i[p];
        if (marker[i] != mark)
            top = sk_dfs(i, Lp, Li, pinv, top, xi, pstack, marker, mark);
    }
    for (px = top; px < n; px++) x[xi[px]] = 0.0;
    for (p = B->p[col]; p < B->p[col + 1]; p++) x[B->i[p]] = B->x[p];
    for (px = top; px < n; px++) {
        int i = xi[px];
        int j = pinv[i];
        double xj;
        if (j < 0) continue;              /* row not yet pivotal */
        xj = x[i];
        if (xj == 0.0) continue;
        /* L has a unit diagonal stored first, so skip it */
        for (p = Lp[j] + 1; p < Lp[j + 1]; p++) x[Li[p]] -= Lx[p] * xj;
    }
    return top;
}

/* ---------------------------------------------------------- factorization */

sk_status sk_lu_factorize(sk_lu *lu, const sk_csc *B, double pivot_tol)
{
    int n, k, p, top, lnz = 0, unz = 0;
    int *rowcnt = NULL;
    double maxB = 0.0, maxU = 0.0;
    sk_status st;

    if (!lu || !B || B->nrow != B->ncol) return SK_ERR_ARG;
    n = B->ncol;
    sk_lu_free_etas(lu);
    lu->nupdate = 0;
    lu->pivot_tol = (pivot_tol > 0.0 && pivot_tol <= 1.0) ? pivot_tol : 0.1;

    if (lu->n != n) {
        free(lu->Lp); free(lu->Up); free(lu->pinv); free(lu->q);
        free(lu->work); free(lu->xi); free(lu->pstack); free(lu->marker);
        lu->Lp     = (int *)malloc(((size_t)n + 1) * sizeof(int));
        lu->Up     = (int *)malloc(((size_t)n + 1) * sizeof(int));
        lu->pinv   = (int *)malloc(((size_t)n + 1) * sizeof(int));
        lu->q      = (int *)malloc(((size_t)n + 1) * sizeof(int));
        lu->work   = (double *)calloc((size_t)n + 1, sizeof(double));
        lu->xi     = (int *)malloc((2 * (size_t)n + 2) * sizeof(int));
        lu->pstack = (int *)malloc(((size_t)n + 1) * sizeof(int));
        lu->marker = (int *)malloc(((size_t)n + 1) * sizeof(int));
        if (!lu->Lp || !lu->Up || !lu->pinv || !lu->q || !lu->work ||
            !lu->xi || !lu->pstack || !lu->marker) { lu->n = 0; return SK_ERR_MEMORY; }
        lu->n = n;
    }
    if (n == 0) { lu->Lp[0] = lu->Up[0] = 0; lu->Lnz = lu->Unz = 0; return SK_OK; }

    {
        int est = B->p[n] * 4 + 4 * n + 1024;
        if (!sk_grow(&lu->Li, &lu->Lx, est, &lu->Lmax)) return SK_ERR_MEMORY;
        if (!sk_grow(&lu->Ui, &lu->Ux, est, &lu->Umax)) return SK_ERR_MEMORY;
    }

    st = sk_lu_order(B, lu->q);
    if (st != SK_OK) return st;

    rowcnt = (int *)calloc((size_t)n + 1, sizeof(int));
    if (!rowcnt) return SK_ERR_MEMORY;
    for (p = 0; p < B->p[n]; p++) {
        double a = fabs(B->x[p]);
        rowcnt[B->i[p]]++;
        if (a > maxB) maxB = a;
    }
    for (k = 0; k < n; k++) { lu->pinv[k] = -1; lu->marker[k] = -1; }

    for (k = 0; k < n; k++) {
        int col = lu->q[k], ipiv = -1, bestrc = 0;
        double a = -1.0, pivot, thresh;
        double *x = lu->work;

        lu->Lp[k] = lnz;
        lu->Up[k] = unz;
        if (!sk_grow(&lu->Li, &lu->Lx, lnz + n + 1, &lu->Lmax)) { free(rowcnt); return SK_ERR_MEMORY; }
        if (!sk_grow(&lu->Ui, &lu->Ux, unz + n + 1, &lu->Umax)) { free(rowcnt); return SK_ERR_MEMORY; }

        top = sk_splsolve(lu->Lp, lu->Li, lu->Lx, B, col, lu->xi, x,
                          lu->pinv, lu->pstack, lu->marker, k);

        for (p = top; p < n; p++) {
            int i = lu->xi[p];
            if (lu->pinv[i] < 0) {
                double t = fabs(x[i]);
                if (t > a) a = t;
            }
        }
        if (a <= 0.0) { free(rowcnt); return SK_ERR_SINGULAR; }

        /* sparsest acceptable pivot row */
        thresh = lu->pivot_tol * a;
        for (p = top; p < n; p++) {
            int i = lu->xi[p];
            if (lu->pinv[i] >= 0) continue;
            if (fabs(x[i]) < thresh) continue;
            if (ipiv < 0 || rowcnt[i] < bestrc ||
                (rowcnt[i] == bestrc && fabs(x[i]) > fabs(x[ipiv]))) {
                ipiv = i; bestrc = rowcnt[i];
            }
        }
        if (ipiv < 0) { free(rowcnt); return SK_ERR_SINGULAR; }
        pivot = x[ipiv];

        /* U(:,k): strictly upper part, then the diagonal last */
        for (p = top; p < n; p++) {
            int i = lu->xi[p];
            if (lu->pinv[i] >= 0) { lu->Ui[unz] = lu->pinv[i]; lu->Ux[unz] = x[i]; unz++; }
        }
        lu->Ui[unz] = k; lu->Ux[unz] = pivot; unz++;
        if (fabs(pivot) > maxU) maxU = fabs(pivot);

        lu->pinv[ipiv] = k;
        lu->Li[lnz] = ipiv; lu->Lx[lnz] = 1.0; lnz++;   /* unit diagonal first */
        for (p = top; p < n; p++) {
            int i = lu->xi[p];
            if (lu->pinv[i] < 0) { lu->Li[lnz] = i; lu->Lx[lnz] = x[i] / pivot; lnz++; }
            x[i] = 0.0;
        }
        for (p = B->p[col]; p < B->p[col + 1]; p++)
            if (rowcnt[B->i[p]] > 0) rowcnt[B->i[p]]--;
    }
    lu->Lp[n] = lnz;  lu->Lnz = lnz;
    lu->Up[n] = unz;  lu->Unz = unz;
    /* switch L to permuted row indices so it is genuinely lower triangular */
    for (p = 0; p < lnz; p++) lu->Li[p] = lu->pinv[lu->Li[p]];

    free(rowcnt);
    lu->nfactor++;
    lu->growth = (maxB > 0.0) ? (maxU / maxB) : 1.0;
    return SK_OK;
}

/* ------------------------------------------------------------------ solves */

/* B = P' L U Q' with (P b)[pinv[i]] = b[i] and (Q y)[q[k]] = y[k]. */

static void sk_apply_eta_ftran(const sk_eta *e, double *x)
{
    double t = x[e->r] / e->pivot;
    int p;
    if (t != 0.0)
        for (p = 0; p < e->nz; p++) x[e->idx[p]] -= t * e->val[p];
    x[e->r] = t;
}

static void sk_apply_eta_btran(const sk_eta *e, double *x)
{
    double s = x[e->r];
    int p;
    for (p = 0; p < e->nz; p++) s -= e->val[p] * x[e->idx[p]];
    x[e->r] = s / e->pivot;
}

sk_status sk_lu_ftran(sk_lu *lu, double *x)
{
    int n = lu->n, i, k, p;
    double *w = lu->work;
    if (n == 0) return SK_OK;

    for (i = 0; i < n; i++) w[lu->pinv[i]] = x[i];
    for (k = 0; k < n; k++) {                       /* L y = Pb */
        double xk = w[k];
        if (xk == 0.0) continue;
        for (p = lu->Lp[k] + 1; p < lu->Lp[k + 1]; p++) w[lu->Li[p]] -= lu->Lx[p] * xk;
    }
    for (k = n - 1; k >= 0; k--) {                  /* U z = y */
        double d = lu->Ux[lu->Up[k + 1] - 1];
        double xk;
        if (d == 0.0) return SK_ERR_SINGULAR;
        w[k] /= d;
        xk = w[k];
        if (xk == 0.0) continue;
        for (p = lu->Up[k]; p < lu->Up[k + 1] - 1; p++) w[lu->Ui[p]] -= lu->Ux[p] * xk;
    }
    for (k = 0; k < n; k++) x[lu->q[k]] = w[k];
    memset(w, 0, (size_t)n * sizeof(double));

    for (i = 0; i < lu->neta; i++) sk_apply_eta_ftran(&lu->eta[i], x);
    return SK_OK;
}

sk_status sk_lu_btran(sk_lu *lu, double *x)
{
    int n = lu->n, i, k, p;
    double *w = lu->work;
    if (n == 0) return SK_OK;

    for (i = lu->neta - 1; i >= 0; i--) sk_apply_eta_btran(&lu->eta[i], x);

    for (k = 0; k < n; k++) w[k] = x[lu->q[k]];
    for (k = 0; k < n; k++) {                       /* U' v = w */
        double s = w[k], d;
        for (p = lu->Up[k]; p < lu->Up[k + 1] - 1; p++) s -= lu->Ux[p] * w[lu->Ui[p]];
        d = lu->Ux[lu->Up[k + 1] - 1];
        if (d == 0.0) return SK_ERR_SINGULAR;
        w[k] = s / d;
    }
    for (k = n - 1; k >= 0; k--) {                  /* L' z = v (unit diagonal) */
        double s = w[k];
        for (p = lu->Lp[k] + 1; p < lu->Lp[k + 1]; p++) s -= lu->Lx[p] * w[lu->Li[p]];
        w[k] = s;
    }
    for (i = 0; i < n; i++) x[i] = w[lu->pinv[i]];
    memset(w, 0, (size_t)n * sizeof(double));
    return SK_OK;
}

/* ------------------------------------------------------------ eta updates */

sk_status sk_lu_update(sk_lu *lu, int r, const double *alpha, double drop_tol)
{
    sk_eta *e;
    int i, nz = 0;

    if (r < 0 || r >= lu->n) return SK_ERR_ARG;
    if (fabs(alpha[r]) < 1e-11) return SK_ERR_SINGULAR;

    if (lu->neta >= lu->etamax) {
        int cap = lu->etamax ? lu->etamax * 2 : 64;
        sk_eta *ne = (sk_eta *)realloc(lu->eta, (size_t)cap * sizeof(sk_eta));
        if (!ne) return SK_ERR_MEMORY;
        lu->eta = ne;
        lu->etamax = cap;
    }
    e = &lu->eta[lu->neta];
    memset(e, 0, sizeof(*e));
    for (i = 0; i < lu->n; i++) if (i != r && fabs(alpha[i]) > drop_tol) nz++;
    e->idx = (int *)malloc(((size_t)nz + 1) * sizeof(int));
    e->val = (double *)malloc(((size_t)nz + 1) * sizeof(double));
    if (!e->idx || !e->val) { free(e->idx); free(e->val); return SK_ERR_MEMORY; }
    nz = 0;
    for (i = 0; i < lu->n; i++)
        if (i != r && fabs(alpha[i]) > drop_tol) { e->idx[nz] = i; e->val[nz] = alpha[i]; nz++; }
    e->nz = nz;
    e->r = r;
    e->pivot = alpha[r];
    lu->neta++;
    lu->nupdate++;
    lu->eta_nz += nz;
    return SK_OK;
}

int sk_lu_needs_refactor(const sk_lu *lu, int interval)
{
    if (lu->nupdate >= interval) return 1;
    if (lu->eta_nz > 4 * (lu->Lnz + lu->Unz) + 10000) return 1;
    return 0;
}

double sk_lu_fill(const sk_lu *lu)
{
    return (double)(lu->Lnz + lu->Unz);
}
