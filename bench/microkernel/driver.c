/* driver.c — sweep sizes, time the kernels, emit one JSON object to stdout.
 *
 * Emits the exact keys tokonomics/loaders.py feeds into roofline.py / economics.py:
 *   peak_int8_gops_on / peak_int8_gops_off  (the build sets which one this is)
 *   mem_bw_gbs                              (from the triad)
 * Plus a correctness flag: gemm() vs gemm_ref() must match bit-for-bit, else
 * the ablation is meaningless. The build label (i8mm on/off) comes from the
 * Makefile via -DABLATION_LABEL so the JSON is self-describing.
 */
#define _POSIX_C_SOURCE 199309L  /* expose clock_gettime/CLOCK_MONOTONIC under -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifndef ABLATION_LABEL
#  define ABLATION_LABEL "unknown"
#endif

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void gemm(const int8_t *, const int8_t *, int32_t *, int, int, int);
void gemm_ref(const int8_t *, const int8_t *, int32_t *, int, int, int);
const char *kernel_path(void);
double triad_gbs(long n, int reps);

static void fill_rand(int8_t *p, long n, unsigned *seed) {
    for (long i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        p[i] = (int8_t)((*seed >> 16) & 0x7f) - 64;  /* [-64,63] */
    }
}

/* time gemm at MxNxK, return GOP/s (2*M*N*K int8 MACs counted as 2 ops). */
static double bench_gemm(int M, int N, int K, int reps, int *ok) {
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    int32_t *R = malloc((size_t)M * N * sizeof(int32_t));
    unsigned seed = 0x1234u;
    fill_rand(A, (long)M * K, &seed);
    fill_rand(B, (long)N * K, &seed);

    gemm(A, B, C, M, N, K);
    gemm_ref(A, B, R, M, N, K);
    *ok = (memcmp(C, R, (size_t)M * N * sizeof(int32_t)) == 0);

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        gemm(A, B, C, M, N, K);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    double ops = 2.0 * (double)M * (double)N * (double)K;
    free(A); free(B); free(C); free(R);
    return ops / best / 1e9;
}

int main(void) {
    int reps = 20;
    int ok_all = 1, ok;
    /* prefill-shaped GEMM sweep; take the peak as the compute ceiling proxy.
     * The last shape {129,131,260} is deliberately *not* 16/8-aligned: M and N
     * are odd (exercises the edge-row / edge-col reference fallbacks) and K=260
     * leaves a tail (260 & ~15 = 256, 260 & ~7 = 256) that the SDOT/SMMLA fast
     * paths hand to gemm_ref. The bit-exact gate below thus covers the tails,
     * not just the aligned happy path. */
    int sizes[][3] = {{128,128,256},{256,256,512},{512,512,512},
                      {256,512,1024},{129,131,260}};
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    double peak = 0.0;
    for (int i = 0; i < n_sizes; i++) {
        double g = bench_gemm(sizes[i][0], sizes[i][1], sizes[i][2], reps, &ok);
        ok_all &= ok;
        if (g > peak) peak = g;
    }
    /* decode-shaped GEMV (M=1), reported for context (bandwidth-bound) */
    double gemv_gops = bench_gemm(1, 4096, 4096, reps, &ok);
    ok_all &= ok;

    double bw = triad_gbs(16L * 1000 * 1000, 10);

    /* The build is either the i8mm-ON or i8mm-OFF binary; we report the peak
     * into the matching key and leave the other for the sibling binary's JSON,
     * which the workflow merges. */
    int is_on = (strcmp(ABLATION_LABEL, "on") == 0);
    printf("{\n");
    /* Must match a row in econ/pricing.yaml: the free ubuntu-24.04-arm runner is
     * Azure Cobalt 100 / Neoverse-N2, whose pricing label is "cobalt100-n2".
     * (tokonomics/cli.py cmd_measured aborts if this label has no pricing row.) */
    printf("  \"label\": \"cobalt100-n2\",\n");
    printf("  \"kind\": \"measured\",\n");
    printf("  \"arch\": \"Neoverse N2 (ubuntu-24.04-arm)\",\n");
    printf("  \"ablation\": \"%s\",\n", ABLATION_LABEL);
    printf("  \"kernel_path\": \"%s\",\n", kernel_path());
    printf("  \"correct\": %s,\n", ok_all ? "true" : "false");
    printf("  \"peak_int8_gops_%s\": %.2f,\n", is_on ? "on" : "off", peak);
    printf("  \"gemv_gops\": %.2f,\n", gemv_gops);
    printf("  \"mem_bw_gbs\": %.2f\n", bw);
    printf("}\n");
    return ok_all ? 0 : 1;
}
