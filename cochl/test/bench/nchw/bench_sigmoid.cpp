// SPDX-License-Identifier: Apache-2.0
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "cochl/include/target/nchw/ncnn.h"

static void fill_random(float* ptr, size_t total, float low = -3.f, float high = 3.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

static void sigmoid_ref(const float* in, float* out, int size)
{
    for (int i = 0; i < size; ++i) {
        float x = in[i];
        out[i] = 1.f / (1.f + std::exp(-x));
    }
}

int main()
{
    const int size = 1 << 20;
    float* in = (float*)aligned_alloc(16, (size_t)size * sizeof(float));
    float* out_ref = (float*)aligned_alloc(16, (size_t)size * sizeof(float));
    float* out = (float*)aligned_alloc(16, (size_t)size * sizeof(float));

    fill_random(in, (size_t)size);
    sigmoid_ref(in, out_ref, size);

    // warmup
    std::memcpy(out, in, (size_t)size * sizeof(float));
    sigmoid_neon_standalone(out, size);

    const int repeats = 200;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        std::memcpy(out, in, (size_t)size * sizeof(float));
        auto t0 = std::chrono::high_resolution_clock::now();
        sigmoid_neon_standalone(out, size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double mean_ms = total_ms / repeats;

    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int i = 0; i < size; ++i)
    {
        double diff = std::fabs((double)out[i] - (double)out_ref[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    double mean_abs = sum_abs / (double)size;

    std::printf("sigmoid_neon_standalone benchmark\n");
    std::printf("  size: %d\n", size);
    std::printf("  mean: %.3f ms\n", mean_ms);
    std::printf("  min:  %.3f ms\n", min_ms);
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", mean_abs);

    std::free(in);
    std::free(out_ref);
    std::free(out);
    return 0;
}
