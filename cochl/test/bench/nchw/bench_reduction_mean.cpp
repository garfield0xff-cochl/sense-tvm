// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
#include "3rdparty/ncnn/src/paramdict.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/reduction.cpp"
#include "cochl/include/target/nchw/ncnn.h"

namespace ncnn {
Option::Option()
{
    std::memset(this, 0, sizeof(Option));
    num_threads = 1;
    lightmode = true;
    use_winograd_convolution = true;
    use_sgemm_convolution = true;
}
}  // namespace ncnn

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

static float max_abs_diff(const float* a, const float* b, size_t n)
{
    float maxv = 0.f;
    for (size_t i = 0; i < n; ++i)
    {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxv) maxv = d;
    }
    return maxv;
}

int main()
{
    int n = 1;
    int c = 64;
    int h = 32;
    int w = 48;
    size_t total = (size_t)n * c * h * w;
    size_t out_total = (size_t)n * c;

    float* input = (float*)aligned_alloc(16, total * sizeof(float));
    float* out_ncnn = (float*)aligned_alloc(16, out_total * sizeof(float));
    float* out_standalone = (float*)aligned_alloc(16, out_total * sizeof(float));

    fill_random(input, total);

    ncnn::Mat bottom(w, h, c, input, 4u, 0);
    ncnn::Mat top(1, 1, c, out_ncnn, 4u, 0);

    Reduction layer;
    ParamDict pd;
    pd.set(0, Reduction::ReductionOp_MEAN);
    pd.set(1, 1);
    pd.set(2, 0);
    pd.set(3, 1);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, 0);
    pd.set(7, 0);
    pd.set(8, 0);
    pd.set(9, 0);
    pd.set(10, 0);
    layer.load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs;
    bottom_blobs.push_back(bottom);
    std::vector<ncnn::Mat> top_blobs(1);
    top_blobs[0] = top;

    ncnn::Option opt;
    opt.num_threads = 1;

    layer.forward(bottom_blobs, top_blobs, opt);
    reduction_mean_hw_keepdims(input, out_standalone, n, c, h, w);

    float diff = max_abs_diff(out_ncnn, out_standalone, out_total);
    std::printf("reduction_mean_hw_keepdims accuracy max abs diff: %.6e\n", diff);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        layer.forward(bottom_blobs, top_blobs, opt);
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
        reduction_mean_hw_keepdims(input, out_standalone, n, c, h, w);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("reduction_mean_hw_keepdims benchmark\n");
    std::printf("  n=%d c=%d h=%d w=%d\n", n, c, h, w);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    std::free(input);
    std::free(out_ncnn);
    std::free(out_standalone);
    return 0;
}
