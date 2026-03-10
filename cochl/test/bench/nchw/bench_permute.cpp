// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cstring>

#include "cochl/include/target/nchw/ncnn.h"
#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/paramdict.h"
#include "3rdparty/ncnn/src/layer/permute.h"

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

static void permute_ref(const float* in, float* out, const int* in_shape, const int* perm, int dims)
{
    int out_shape[8];
    for (int i = 0; i < dims; ++i) out_shape[i] = in_shape[perm[i]];

    size_t in_stride[8];
    size_t out_stride[8];
    in_stride[dims - 1] = 1;
    out_stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
    {
        in_stride[i] = in_stride[i + 1] * (size_t)in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * (size_t)out_shape[i + 1];
    }

    size_t total = 1;
    for (int i = 0; i < dims; ++i) total *= (size_t)out_shape[i];

    for (size_t idx = 0; idx < total; ++idx)
    {
        size_t tmp = idx;
        size_t in_index = 0;
        for (int i = 0; i < dims; ++i)
        {
            size_t coord = tmp / out_stride[i];
            tmp -= coord * out_stride[i];
            in_index += coord * in_stride[perm[i]];
        }
        out[idx] = in[in_index];
    }
}

int main()
{
    // From model_main_17.onnx: [1,128,192,3] -> [1,3,128,192]
    // ncnn Mat has no batch dim, use dims=3 (w=192, h=128, c=3)
    // order_type=5 corresponds to c h w
    int in_shape[3] = {192, 128, 3};
    int perm[3] = {2, 1, 0};
    int dims = 3;

    size_t total = 1;
    for (int i = 0; i < dims; ++i) total *= (size_t)in_shape[i];

    float* in = (float*)aligned_alloc(16, total * sizeof(float));
    float* out = (float*)aligned_alloc(16, total * sizeof(float));
    float* out_ref = (float*)aligned_alloc(16, total * sizeof(float));

    fill_random(in, total);
    permute_ref(in, out_ref, in_shape, perm, dims);

    ncnn::Permute permute;
    ncnn::ParamDict pd;
    pd.set(0, 5);  // order_type = c h w
    permute.load_param(pd);

    ncnn::Mat bottom_blob(192, 128, 3, (void*)in, (size_t)4u);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn::Mat top_blob;
        permute.forward(bottom_blob, top_blob, ncnn::Option());
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
        permute_nd(in, out, in_shape, dims, perm);
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

    std::printf("permute_nd benchmark\n");
    std::printf("  perm: [0,3,1,2]\n");
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
