// SPDX-License-Identifier: Apache-2.0
#include <stddef.h>

#include "cochl/include/target/nchw/ncnn.h"

// Standalone GEMM for MatMul/Gemm in model_main_17.onnx
// transA/transB meaning:
//   transA=0 uses A as-is, transA=1 uses A^T
//   transB=0 uses B as-is, transB=1 uses B^T
// This implementation targets transA=0, transB=0.
// A: [M, K], B: [K, N], C: [M, N], row-major
int matmul_gemm_neon_standalone(const float* A,
                                const float* B,
                                const float* bias,
                                float* C,
                                int M,
                                int K,
                                int N)
{
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) return -1;

    for (int m = 0; m < M; ++m)
    {
        const float* a_row = A + (size_t)m * K;
        float* c_row = C + (size_t)m * N;

        int n = 0;
#if __ARM_NEON
        for (; n + 3 < N; n += 4)
        {
            float32x4_t acc = bias ? vld1q_f32(bias + n) : vdupq_n_f32(0.f);
            const float* b_col = B + n;
            for (int k = 0; k < K; ++k)
            {
                float a = a_row[k];
                float32x4_t b = vld1q_f32(b_col);
                acc = vfmaq_n_f32(acc, b, a);
                b_col += N;
            }
            vst1q_f32(c_row + n, acc);
        }
#endif
        for (; n < N; ++n)
        {
            float acc = bias ? bias[n] : 0.f;
            const float* b_col = B + n;
            for (int k = 0; k < K; ++k)
            {
                acc += a_row[k] * b_col[0];
                b_col += N;
            }
            c_row[n] = acc;
        }
    }
    return 0;
}
