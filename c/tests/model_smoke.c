#include "sankhya_model.h"

#include <math.h>

int main(void) {
    const size_t offsets[] = {0, 2, 4};
    const int rows[] = {0, 1, 0, 1};
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    const double x[] = {2.0, 5.0};
    double y[2];
    SankhyaCSC matrix = {0};
    if (sankhya_csc_create(&matrix, 2, 2, 4, offsets, rows, values) != SANKHYA_OK) return 1;
    if (sankhya_csc_matvec(&matrix, x, y) != SANKHYA_OK) return 2;
    sankhya_csc_destroy(&matrix);
    return fabs(y[0] - 17.0) < 1e-12 && fabs(y[1] - 24.0) < 1e-12 ? 0 : 3;
}

