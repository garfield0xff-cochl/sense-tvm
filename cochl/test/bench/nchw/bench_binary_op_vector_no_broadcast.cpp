// SPDX-License-Identifier: Apache-2.0
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "cochl/include/target/nchw/ncnn.h"

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

using BinFn = int (*)(const float*, const float*, float*, int);

static void bench_case(const char* name, BinFn ncnn_fn, BinFn standalone_fn, int size)
{
    float* a = (float*)aligned_alloc(16, (size_t)size * sizeof(float));
    float* b = (float*)aligned_alloc(16, (size_t)size * sizeof(float));
    float* out_ncnn = (float*)aligned_alloc(16, (size_t)size * sizeof(float));
    float* out_standalone = (float*)aligned_alloc(16, (size_t)size * sizeof(float));

    fill_random(a, (size_t)size);
    fill_random(b, (size_t)size);

    const int repeats = 200;
    double total_ms = 0.0;
    double min_ms = 1e30;

    // warmup
    ncnn_fn(a, b, out_ncnn, size);
    standalone_fn(a, b, out_standalone, size);

    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn_fn(a, b, out_ncnn, size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double ncnn_mean = total_ms / repeats;
    double ncnn_min = min_ms;

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        standalone_fn(a, b, out_standalone, size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int i = 0; i < size; ++i)
    {
        double diff = std::fabs((double)out_ncnn[i] - (double)out_standalone[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    double mean_abs = sum_abs / (double)size;

    std::printf("%s benchmark\n", name);
    std::printf("  size: %d\n", size);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", mean_abs);

    std::free(a);
    std::free(b);
    std::free(out_ncnn);
    std::free(out_standalone);
}

int main()
{
    const int size = 1 << 20;
    bench_case(
        "binary_op_vector_no_broadcast_add",
        ncnn_binary_op_vector_no_broadcast_add,
        binary_op_vector_no_broadcast_add_standalone,
        size);
    bench_case(
        "binary_op_vector_no_broadcast_max",
        ncnn_binary_op_vector_no_broadcast_max,
        binary_op_vector_no_broadcast_max_standalone,
        size);
    bench_case(
        "binary_op_vector_no_broadcast_min",
        ncnn_binary_op_vector_no_broadcast_min,
        binary_op_vector_no_broadcast_min_standalone,
        size);
    return 0;
}
