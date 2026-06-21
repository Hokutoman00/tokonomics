/* kernel.c — int8 GEMM (prefill proxy) + GEMV (decode proxy).
 *
 * One source, three code paths selected at compile time:
 *   - i8mm ON  : Armv8.6 MatMulInt8, SMMLA via vmmlaq_s32  (-DUSE_I8MM, +i8mm)
 *   - i8mm OFF : Armv8.2 dotprod,     SDOT  via vdotq_s32   (+dotprod)
 *   - scalar   : portable reference (x86 dev / correctness only, never timed)
 *
 * The whole point of the ablation is that ON and OFF must produce *bit-identical*
 * int32 accumulators on the same input — only the instruction (and thus the
 * GOP/s) changes. gemm_ref() is the oracle the driver checks both paths against.
 *
 * Layout: A is MxK row-major (int8), B is KxN but stored column-major (int8) so
 * the K dimension is contiguous for both SDOT and SMMLA. C is MxN int32 row-major.
 */
#include <stdint.h>
#include <string.h>

#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(USE_I8MM)
#  include <arm_neon.h>
#  define KERNEL_PATH "i8mm/smmla"
#elif defined(__ARM_FEATURE_DOTPROD)
#  include <arm_neon.h>
#  define KERNEL_PATH "dotprod/sdot"
#else
#  define KERNEL_PATH "scalar"
#endif

const char *kernel_path(void) { return KERNEL_PATH; }

/* Portable reference: the oracle. Always compiled, never used for timing. */
void gemm_ref(const int8_t *A, const int8_t *B, int32_t *C,
              int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * K + k] * (int32_t)B[j * K + k];
            C[i * N + j] = acc;
        }
    }
}

#if defined(__ARM_FEATURE_MATMUL_INT8) && defined(USE_I8MM)
/* SMMLA computes a 2x2 int32 tile from 2x8 * 8x2 int8 in one instruction.
 * We tile M,N by 2 and step K by 8. Tails fall back to the reference. */
void gemm(const int8_t *A, const int8_t *B, int32_t *C,
          int M, int N, int K) {
    int Mt = M & ~1, Nt = N & ~1, Kt = K & ~7;
    for (int i = 0; i < Mt; i += 2) {
        for (int j = 0; j < Nt; j += 2) {
            int32x4_t acc = vdupq_n_s32(0);  /* [c00 c01 c10 c11] */
            for (int k = 0; k < Kt; k += 8) {
                /* pack 2 rows of A (2x8) and 2 cols of B (2x8) */
                int8x16_t a = vcombine_s8(vld1_s8(&A[(i)   * K + k]),
                                          vld1_s8(&A[(i+1) * K + k]));
                int8x16_t b = vcombine_s8(vld1_s8(&B[(j)   * K + k]),
                                          vld1_s8(&B[(j+1) * K + k]));
                acc = vmmlaq_s32(acc, a, b);
            }
            int32_t t[4];
            vst1q_s32(t, acc);
            int32_t c00 = t[0], c01 = t[1], c10 = t[2], c11 = t[3];
            for (int k = Kt; k < K; k++) {
                c00 += (int32_t)A[i*K+k]     * (int32_t)B[j*K+k];
                c01 += (int32_t)A[i*K+k]     * (int32_t)B[(j+1)*K+k];
                c10 += (int32_t)A[(i+1)*K+k] * (int32_t)B[j*K+k];
                c11 += (int32_t)A[(i+1)*K+k] * (int32_t)B[(j+1)*K+k];
            }
            C[i*N+j]       = c00; C[i*N+j+1]     = c01;
            C[(i+1)*N+j]   = c10; C[(i+1)*N+j+1] = c11;
        }
    }
    /* edge rows/cols via reference so results match exactly */
    for (int i = Mt; i < M; i++)
        for (int j = 0; j < N; j++) { int32_t a=0; for(int k=0;k<K;k++) a+=(int32_t)A[i*K+k]*(int32_t)B[j*K+k]; C[i*N+j]=a; }
    for (int i = 0; i < Mt; i++)
        for (int j = Nt; j < N; j++) { int32_t a=0; for(int k=0;k<K;k++) a+=(int32_t)A[i*K+k]*(int32_t)B[j*K+k]; C[i*N+j]=a; }
}

#elif defined(__ARM_FEATURE_DOTPROD)
/* SDOT accumulates one int32 from 4 int8*int8 products. One dot per (i,j). */
void gemm(const int8_t *A, const int8_t *B, int32_t *C,
          int M, int N, int K) {
    int Kt = K & ~15;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < Kt; k += 16) {
                int8x16_t a = vld1q_s8(&A[i * K + k]);
                int8x16_t b = vld1q_s8(&B[j * K + k]);
                acc = vdotq_s32(acc, a, b);
            }
            int32_t c = vaddvq_s32(acc);
            for (int k = Kt; k < K; k++)
                c += (int32_t)A[i * K + k] * (int32_t)B[j * K + k];
            C[i * N + j] = c;
        }
    }
}

#else
void gemm(const int8_t *A, const int8_t *B, int32_t *C,
          int M, int N, int K) { gemm_ref(A, B, C, M, N, K); }
#endif

/* GEMV decode proxy: M=1 GEMM. Reuses the same gemm() so the only difference
 * between prefill and decode is the shape (and thus arithmetic intensity),
 * which is exactly the variable the roofline isolates. */
void gemv(const int8_t *A, const int8_t *B, int32_t *C, int N, int K) {
    gemm(A, B, C, 1, N, K);
}
