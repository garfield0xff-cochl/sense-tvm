// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

#include "cochl/include/target/nchw/ncnn.h"
#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/paramdict.h"
#include "3rdparty/ncnn/src/layer/squeeze.h"

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

int main()
{
    // Expanded size for timing visibility: [1, 1280, 64, 64] -> [1, 1280, 64, 64] (no-op squeeze)
    int in_shape[4] = {1, 1280, 64, 64};
    int axes[2] = {2, 3};
    int in_dims = 4;

    size_t total = 1;
    for (int i = 0; i < in_dims; ++i) total *= (size_t)in_shape[i];

    float* in = (float*)aligned_alloc(16, total * sizeof(float));
    float* out = (float*)aligned_alloc(16, total * sizeof(float));
    float* out_ref = (float*)aligned_alloc(16, total * sizeof(float));

    fill_random(in, total);
    std::memcpy(out_ref, in, total * sizeof(float));

    ncnn::Squeeze squeeze;
    ncnn::ParamDict pd;
    int axes_data[2] = {2, 3};
    ncnn::Mat axes_mat(2, (void*)axes_data, (size_t)4u);
    pd.set(3, axes_mat);
    squeeze.load_param(pd);

    ncnn::Mat bottom_blob(64, 64, 1, 1280, (void*)in, (size_t)4u);

    const int repeats = 200;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn::Mat top_blob;
        squeeze.forward(bottom_blob, top_blob, ncnn::Option());
        std::memcpy(out, (const float*)top_blob, total * sizeof(float));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double ref_mean = total_ms / repeats;
    double ref_min = min_ms;

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        squeeze_nd(in, out, in_shape, in_dims, axes, 2);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    double max_abs = 0.0;
    for (size_t i = 0; i < total; ++i)
    {
        double diff = std::fabs((double)out[i] - (double)out_ref[i]);
        if (diff > max_abs) max_abs = diff;
    }

    std::printf("squeeze_nd benchmark\n");
    std::printf("  shape: [1,1280,64,64] -> [1,1280,64,64]\n");
    std::printf("  ref mean: %.3f ms\n", ref_mean);
    std::printf("  ref min:  %.3f ms\n", ref_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);
    std::printf("  max abs diff: %.6e\n", max_abs);

    std::free(in);
    std::free(out);
    std::free(out_ref);
    return 0;
}
