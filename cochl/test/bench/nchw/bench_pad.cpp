// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
#include "3rdparty/ncnn/src/paramdict.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/padding.cpp"
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
    int c = 32;
    int h = 64;
    int w = 96;
    int pad_top = 2;
    int pad_left = 3;
    int pad_bottom = 4;
    int pad_right = 1;
    float value = -0.75f;

    int out_h = h + pad_top + pad_bottom;
    int out_w = w + pad_left + pad_right;

    size_t total = (size_t)n * c * h * w;
    size_t out_total = (size_t)n * c * out_h * out_w;

    float* input = (float*)aligned_alloc(16, total * sizeof(float));
    float* out_ncnn = (float*)aligned_alloc(16, out_total * sizeof(float));
    float* out_standalone = (float*)aligned_alloc(16, out_total * sizeof(float));

    fill_random(input, total);

    ncnn::Mat bottom(w, h, c, input, 4u, 0);
    ncnn::Padding layer;
    ncnn::ParamDict pd;
    pd.set(0, pad_top);
    pd.set(1, pad_bottom);
    pd.set(2, pad_left);
    pd.set(3, pad_right);
    pd.set(4, 0);
    pd.set(5, value);
    layer.load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Mat top_blob;
    layer.forward(bottom, top_blob, opt);
    std::memcpy(out_ncnn, (const float*)top_blob, out_total * sizeof(float));

    pad2d_nchw(input, out_standalone, n, c, h, w, pad_top, pad_left, pad_bottom, pad_right, value);

    float diff = max_abs_diff(out_ncnn, out_standalone, out_total);
    std::printf("pad2d_nchw accuracy max abs diff: %.6e\n", diff);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn::Mat tmp;
        layer.forward(bottom, tmp, opt);
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
        pad2d_nchw(input, out_standalone, n, c, h, w, pad_top, pad_left, pad_bottom, pad_right, value);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("pad2d_nchw benchmark\n");
    std::printf("  n=%d c=%d h=%d w=%d\n", n, c, h, w);
    std::printf("  pads: top=%d left=%d bottom=%d right=%d\n", pad_top, pad_left, pad_bottom, pad_right);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    std::free(input);
    std::free(out_ncnn);
    std::free(out_standalone);
    return 0;
}
