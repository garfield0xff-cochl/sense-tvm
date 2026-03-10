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
#include "3rdparty/ncnn/src/layer/binaryop.cpp"
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
    int outer = 1;
    int inner = 863;
    size_t total = (size_t)outer * inner;

    float* a = (float*)aligned_alloc(16, total * sizeof(float));
    float* b = (float*)aligned_alloc(16, inner * sizeof(float));
    float* out_ncnn = (float*)aligned_alloc(16, total * sizeof(float));
    float* out_standalone = (float*)aligned_alloc(16, total * sizeof(float));

    fill_random(a, total);
    fill_random(b, inner);

    ncnn::Mat A(inner, outer, 1, a, 4u, 0);
    ncnn::Mat B(inner, b, 4u, 1);
    ncnn::Mat C(inner, outer, 1, out_ncnn, 4u, 0);

    BinaryOp layer;
    ParamDict pd;
    pd.set(0, BinaryOp::Operation_ADD);
    pd.set(1, 0);
    pd.set(2, 0.f);
    layer.load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs;
    bottom_blobs.push_back(A);
    bottom_blobs.push_back(B);
    std::vector<ncnn::Mat> top_blobs(1);
    top_blobs[0] = C;

    ncnn::Option opt;
    opt.num_threads = 1;

    layer.forward(bottom_blobs, top_blobs, opt);
    binary_op_broadcast_add_standalone(a, b, out_standalone, outer, inner);

    float diff = max_abs_diff(out_ncnn, out_standalone, total);
    std::printf("binary_op_broadcast_add accuracy max abs diff: %.6e\n", diff);

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
        binary_op_broadcast_add_standalone(a, b, out_standalone, outer, inner);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("binary_op_broadcast_add benchmark\n");
    std::printf("  outer=%d inner=%d\n", outer, inner);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    std::free(a);
    std::free(b);
    std::free(out_ncnn);
    std::free(out_standalone);
    return 0;
}
