/* bandwidth.c — STREAM-style triad to measure effective memory bandwidth.
 *
 * a[i] = b[i] + s*c[i] moves 3 doubles per element (read b, read c, write a),
 * with ~zero arithmetic intensity, so it saturates the memory subsystem. This
 * is the number that pins the decode operating point on the roofline: decode is
 * bandwidth-bound, so no amount of i8mm peak can lift it above mem_bw * AI.
 */
#include <stdlib.h>

double now_sec(void);  /* defined in driver.c (monotonic clock) */

double triad_gbs(long n, int reps) {
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { free(a); free(b); free(c); return 0.0; }
    for (long i = 0; i < n; i++) { b[i] = 1.0; c[i] = 2.0; }
    const double s = 3.0;

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        for (long i = 0; i < n; i++) a[i] = b[i] + s * c[i];
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
        /* defeat dead-store elimination */
        if (a[n / 2] < 0) b[0] = a[n / 2];
    }
    double bytes = 3.0 * (double)n * sizeof(double);
    free(a); free(b); free(c);
    return bytes / best / 1e9;
}
