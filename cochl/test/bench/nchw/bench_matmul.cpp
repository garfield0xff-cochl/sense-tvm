// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "cochl/include/target/nchw/ncnn.h"

static void* alloc_aligned(size_t size, size_t align)
{
    size_t size_aligned = (size + align - 1) & ~(align - 1);
    return aligned_alloc(align, size_aligned);
}

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

static void gemm_ref(const float* A, const float* B, const float* bias, float* C, int M, int K, int N)
{
    for (int m = 0; m < M; ++m)
    {
        const float* a_row = A + (size_t)m * K;
        float* c_row = C + (size_t)m * N;
        for (int n = 0; n < N; ++n)
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
}

int main()
{
    // Shapes derived from sense/onnx/model_main_17.onnx (Gemm)
    const int M = 1;
    const int K = 1280;
    const int N = 863;

    float* A = (float*)alloc_aligned((size_t)M * K * sizeof(float), 16);
    float* B = (float*)alloc_aligned((size_t)K * N * sizeof(float), 16);
    float* bias = (float*)alloc_aligned((size_t)N * sizeof(float), 16);
    float* C = (float*)alloc_aligned((size_t)M * N * sizeof(float), 16);
    float* C_ref = (float*)alloc_aligned((size_t)M * N * sizeof(float), 16);

    fill_random(A, (size_t)M * K);
    fill_random(B, (size_t)K * N);
    fill_random(bias, (size_t)N);

    gemm_ref(A, B, bias, C_ref, M, K, N);

    // warmup
    matmul_gemm_neon_standalone(A, B, bias, C, M, K, N);

    const int repeats = 200;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_gemm_neon_standalone(A, B, bias, C, M, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int i = 0; i < M * N; ++i)
    {
        double diff = std::fabs((double)C[i] - (double)C_ref[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    double mean_abs = sum_abs / (double)(M * N);

    std::printf("matmul_gemm_neon_standalone benchmark\n");
    std::printf("  shape: M=%d K=%d N=%d\n", M, K, N);
    std::printf("  mean: %.3f ms\n", total_ms / repeats);
    std::printf("  min:  %.3f ms\n", min_ms);
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", mean_abs);

    std::free(A);
    std::free(B);
    std::free(bias);
    std::free(C);
    std::free(C_ref);
    return 0;
}
